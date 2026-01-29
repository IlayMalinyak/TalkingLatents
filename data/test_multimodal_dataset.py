import pandas as pd
import numpy as np
import torch
import json
import os
from data.dataset_interpert import StellarQuestionsDataset
import random

# Mock data
OBSID = 12345
mock_json_data = [
    {
        "obsid": OBSID,
        "description": "Question: Describe the star.\nDescription: It is a star.",
        "stellar_data": {"Teff": 5000, "logg": 4.0, "FeH": 0.0}
    }
]

# Write mock JSON
json_path = "mock_stellar_data.json"
with open(json_path, 'w') as f:
    json.dump(mock_json_data, f)

# Mock DataFrame
# 1. Binarity YES
# 2. Binarity NO
# 3. Age Valid
# 4. Age NaN, fallback Age valid

test_scenarios = [
    {
        "name": "Binarity YES",
        "df_row": {'binarity_class_hard': 1, 'final_age': np.nan, 'age_ref': None, 'Age': np.nan},
        "expected_bin": "Yes"
    },
    {
        "name": "Binarity NO",
        "df_row": {'binarity_class_hard': 2, 'final_age': np.nan, 'age_ref': None, 'Age': np.nan},
        "expected_bin": "No"
    },
    {
        "name": "Binarity Probably NO",
        "df_row": {'binarity_class_hard': np.nan, 'final_age': np.nan, 'age_ref': None, 'Age': np.nan},
        "expected_bin": "Probably no"
    },
    {
        "name": "Age Final",
        "df_row": {'binarity_class_hard': np.nan, 'final_age': 5.5, 'age_ref': None, 'Age': np.nan},
        "expected_age": "5.50"
    },
    {
        "name": "Age Fallback",
        "df_row": {'binarity_class_hard': np.nan, 'final_age': np.nan, 'age_ref': None, 'Age': 3.2},
        "expected_age": "3.20"
    }
]

print("Starting verification...")

for scenario in test_scenarios:
    print(f"\nTraining Scenario: {scenario['name']}")
    df = pd.DataFrame([scenario['df_row']], index=[OBSID])
    
    # Init dataset
    ds = StellarQuestionsDataset(
        json_file=json_path,
        multimodal_df=df,
        enable_followup=True,
        followup_prob=1.0,
        tokenizer_path=None, # Fallback tokenizer
        split='train',
        # Force randomness to hit our target eventually, but we can inspect manually
        random_state=42
    )

    # We need to run multiple times to see different followups because it's random
    seen_questions = set()
    found_expected = False
    
    # Monkey patch RNG to force specific outcomes? 
    # Or just run loop enough times.
    
    for _ in range(30):
        sample = ds[0]
        # Parse text turns
        if 'followup_turns' in sample:
            for q, a in sample['followup_turns']:
                seen_questions.add(f"Q: {q} | A: {a}")
                
                if "is this star a binary" in q.lower():
                    if 'expected_bin' in scenario:
                        print(f"  Found Binarity: {a} (Expected: {scenario['expected_bin']})")
                        if a == scenario['expected_bin']:
                            found_expected = True
                
                if "what is the age" in q.lower():
                    if 'expected_age' in scenario:
                        print(f"  Found Age: {a} (Expected: {scenario['expected_age']})")
                        if a == scenario['expected_age']:
                            found_expected = True

    if not found_expected and ('expected_bin' in scenario or 'expected_age' in scenario):
         print("  WARNING: Did not find expected question/answer pair after 30 tries.")
    elif found_expected:
         print("  SUCCESS: Found expected answer.")
    else:
         print("  (No specific expectation for this scenario other than running)")

# Clean up
if os.path.exists(json_path):
    os.remove(json_path)

print("\nVerification Complete.")
