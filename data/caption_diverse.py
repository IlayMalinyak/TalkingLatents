import numpy as np
import pandas as pd
import json
import os
os.system("pip install google-genai")
import time
import re
import warnings
from multiprocessing import Pool
from tqdm import tqdm
from google import genai
from typing import Dict, List, Any, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION (The Hybrid Strategy)
# ==========================================

# High Quality Model (Expensive/Slow) - For the "Gold Standard" subset
MODEL_HIGH_QUALITY = "models/gemini-3-pro-preview"
DELAY_HIGH_QUALITY = 0.5  # Slower to avoid rate limits (30 RPM)

# Bulk Model (Cheap/Fast) - For the massive training set
MODEL_BULK = "models/gemini-2.5-flash"
DELAY_BULK = 0.1  # Fast (can handle 100+ RPM)

# How many "Gold Standard" examples do you want?
GOLD_STANDARD_COUNT = 1000 

API_KEY_FILE = "/home/ilay.kamai/work/TalkingLatents/google_api.txt"
N_PROCESSES = 32  # Keep low to be safe with the Pro model

# ==========================================
# 2. Physics & Helper Functions
# ==========================================

def giant_cond(teff, logg):
    """Condition for red giants (Ciardi et al. 2011)."""
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4
    else:
        thresh = 5.2 - (2.8 * 1e-4 * teff)
    return logg <= thresh

def init_gemini_client():
    """Initialize Gemini client safely from file."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        try:
            if os.path.exists(API_KEY_FILE):
                with open(API_KEY_FILE, 'r') as f:
                    api_key = f.read().strip()
                os.environ["GOOGLE_API_KEY"] = api_key
            else:
                raise FileNotFoundError(f"Could not find {API_KEY_FILE}")
        except Exception as e:
            print(f"Auth Error: {e}")
            return None
    return genai.Client(api_key=api_key)

def parse_gemini_json_response(response_text):
    """Robust JSON parser."""
    text = response_text.strip()
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            text = re.sub(r',\s*([\]}])', r'\1', text)
            return json.loads(text)
        except:
            return None

# ==========================================
# 3. Prompt Engineering
# ==========================================

def create_star_prompt(example):
    teff = example.get('Teff', 0)
    logg = example.get('logg', 0)
    feh = example.get('FeH', 0)
    
    is_giant = giant_cond(teff, logg)
    evolutionary_phase = "Red Giant Branch / Evolved" if is_giant else "Main Sequence / Dwarf"

    data_context = f"""
    OBJECT DATA:
    - Effective Temperature (Teff): {teff:.0f} K
    - Surface Gravity (log g): {logg:.2f}
    - Metallicity ([Fe/H]): {feh:.2f}
    - Evolutionary Phase: {evolutionary_phase}
    """

    prompt = f"""
You are an expert astrophysicist creating a training dataset.
{data_context}

Create 3 distinct Question-Answer pairs based strictly on the data above.

REQUIRED OUTPUT FORMAT (JSON):
{{
  "qa_pairs": [
    {{
      "type": "retrieval",
      "question": "Ask for a specific value (e.g., 'What is the Teff?').",
      "answer": "The short answer."
    }},
    {{
      "type": "classification",
      "question": "Yes/No or Category question (e.g., 'Is this evolved?').",
      "answer": "Answer with brief justification."
    }},
    {{
      "type": "reasoning",
      "question": "Complex physical analysis question.",
      "answer": "Scientific explanation linking variables."
    }}
  ]
}}
"""
    return prompt

# ==========================================
# 4. Worker Process (Updated for Dynamic Model)
# ==========================================

client = None

def process_star_worker(args):
    """
    Args now includes 'model_id' and specific 'delay'
    """
    global client
    row_data, obsid_column, max_retries, delay, model_id = args
    
    if client is None:
        client = init_gemini_client()
        
    idx, row = row_data
    star_info = row.to_dict()
    
    if 'Teff' not in star_info or pd.isna(star_info['Teff']):
        return None

    prompt = create_star_prompt(star_info)
    result_json = None
    
    for attempt in range(max_retries):
        try:
            time.sleep(delay) # Dynamic delay based on model type
            
            response = client.models.generate_content(
                contents=[prompt],
                model=model_id,  # Uses the specific model assigned to this row
                config={"response_mime_type": "application/json"}
            )
            
            parsed = parse_gemini_json_response(response.text)
            if parsed and "qa_pairs" in parsed:
                result_json = parsed["qa_pairs"]
                break
                
        except Exception as e:
            # print(f"Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1) * 2) # Aggressive backoff

    if result_json:
        return {
            'index': idx,
            obsid_column: star_info.get(obsid_column, None),
            'model_used': model_id, # Track which model generated this
            'qa_pairs': result_json,
            'stellar_data': {k: v for k, v in star_info.items() if not pd.isna(v)}
        }
    return None

def generate_dataset(df, output_file, obsid_column='obsid', n_processes=4):
    print(f"Starting HYBRID generation for {len(df)} stars.")
    print(f" - First {GOLD_STANDARD_COUNT} stars: {MODEL_HIGH_QUALITY} (High Quality)")
    print(f" - Remaining stars: {MODEL_BULK} (Bulk/Fast)")
    
    # --- DYNAMIC ARGUMENT GENERATION ---
    args_list = []
    for i, (idx, row) in enumerate(df.iterrows()):
        
        # Decide which model to use based on index
        if i < GOLD_STANDARD_COUNT:
            # Use the smart teacher
            current_model = MODEL_HIGH_QUALITY
            current_delay = DELAY_HIGH_QUALITY
        else:
            # Use the fast worker
            current_model = MODEL_BULK
            current_delay = DELAY_BULK
            
        args_list.append(
            ((idx, row), obsid_column, 3, current_delay, current_model)
        )
    
    results = []
    
    # Run Multiprocessing
    if n_processes > 1:
        with Pool(processes=n_processes) as pool:
            # We use imap (ordered) or imap_unordered. 
            # Unordered is faster, but we want to track progress nicely.
            iterator = pool.imap_unordered(process_star_worker, args_list)
            
            with tqdm(total=len(df), desc="Generating QA") as pbar:
                for res in iterator:
                    if res:
                        results.append(res)
                    pbar.update(1)
                    
                    # Intermediate save every 500 items
                    if len(results) % 500 == 0:
                        pd.DataFrame(results).to_json(output_file, orient='records', indent=2)
    else:
        # Single process fallback
        for args in tqdm(args_list):
            res = process_star_worker(args)
            if res: results.append(res)

    # Final Save
    print(f"Finished. Success rate: {len(results)}/{len(df)}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

# ==========================================
# 5. Execution
# ==========================================

if __name__ == "__main__":
    # Input paths
    INPUT_CSV = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info.csv'
    OUTPUT_FILE = '/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_qa_hybrid.json'
    
    try:
        df = pd.read_csv(INPUT_CSV)
        # Fix Teff normalization if needed
        if df['Teff'].mean() < 100: 
            df['Teff'] = df['Teff'] * 5778
    except FileNotFoundError:
        # Synthetic test data
        print("Using synthetic data...")
        df = pd.DataFrame({
            'obsid': range(1050), # Enough to trigger both models
            'Teff': np.random.normal(5778, 1000, 1050),
            'logg': np.random.normal(4.4, 0.5, 1050),
            'FeH': np.random.normal(0.0, 0.2, 1050)
        })

    # Run
    generate_dataset(df, OUTPUT_FILE, n_processes=N_PROCESSES)