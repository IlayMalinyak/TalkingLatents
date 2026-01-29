"""
Fixed analysis for stellar type classification from followup questions.

This script properly analyzes the stellar type questions in followup_qa,
and creates visualizations showing example answers.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import seaborn as sns
import os
import sys
import math
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from src.snr_lookup import load_snr_lookup


# Stellar type keywords for classification
STELLAR_TYPE_KEYWORDS = {
    'dwarf': ['dwarf', 'ms star', 'a main sequence star', 'a main-sequence star', 
              'on the main sequence', 'near the main sequence', 'on or near the main sequence'],
    'subgiant': ['subgiant', 'sub-giant'],
    'giant': ['giant', 'rgb', 'a red giant', 'red giant branch'],
    'supergiant': ['supergiant', 'super-giant'],
    'white_dwarf': ['white dwarf', 'white-dwarf', 'wd'],
}

# Spectral type keywords
SPECTRAL_TYPES = ['O', 'B', 'A', 'F', 'G', 'K', 'M']


def extract_stellar_type_from_text(text: str) -> Optional[str]:
    """Extract stellar type classification from text with context awareness."""
    if not text:
        return None

    text_lower = text.lower()
    
    # Context modifiers that indicate comparison rather than direct classification
    comparison_phrases = [
        'compared to', 'similar to', 'like a', 'than a', 'not a', 
        'unlike a', 'rather than', 'instead of', 'versus'
    ]
    
    # Score each stellar type based on context
    scores = {}
    
    for stellar_type, keywords in STELLAR_TYPE_KEYWORDS.items():
        scores[stellar_type] = 0
        
        for keyword in keywords:
            # Find all occurrences of the keyword
            start = 0
            while True:
                pos = text_lower.find(keyword, start)
                if pos == -1:
                    break
                
                # Get context around the keyword (50 chars before and after)
                context_start = max(0, pos - 50)
                context_end = min(len(text_lower), pos + len(keyword) + 50)
                context = text_lower[context_start:context_end]
                
                # Check if it's in a comparison context
                is_comparison = any(phrase in context for phrase in comparison_phrases)
                
                # Check for direct classification indicators
                direct_indicators = [
                    'is a', 'is likely a', 'classified as', 'suggests', 
                    'likely a', 'appears to be', 'star is', 'is an',
                    'likely on', 'on or near', 'near the', 'on the'
                ]
                is_direct = any(indicator in context for indicator in direct_indicators)
                
                # Score based on context
                if is_direct and not is_comparison:
                    scores[stellar_type] += 3  # Strong positive signal
                elif not is_comparison:
                    scores[stellar_type] += 2  # Moderate positive signal
                elif is_comparison:
                    scores[stellar_type] -= 1  # Comparison context is negative
                else:
                    scores[stellar_type] += 1  # Weak positive signal
                
                start = pos + 1
    
    # Return the type with the highest score (if score > 0)
    if scores:
        best_type = max(scores.items(), key=lambda x: x[1])
        if best_type[1] > 0:
            return best_type[0]
    
    return None


def extract_spectral_type_from_text(text: str) -> List[str]:
    """Extract spectral type letters (O, B, A, F, G, K, M) from text. Returns list of all matches."""
    if not text:
        return []

    import re
    text_upper = text.upper()
    matches = set()

    # Priority 1: Strong explicit mentions like "G-type", "type G"
    # This avoids "A star" matching "a star" because we require "TYPE"
    for spec_type in SPECTRAL_TYPES:
        # Match: "G-type", "G type", "type G"
        pattern = rf'\b{spec_type}\s*[-]?\s*TYPE\b|\bTYPE\s+{spec_type}\b'
        if re.search(pattern, text_upper):
            matches.add(spec_type)
            
    # Priority 2: Look for "G star", "Class G"
    # Skip 'A' here because "A STAR" matches "a star" too easily
    for spec_type in SPECTRAL_TYPES:
        if spec_type == 'A':
            continue
            
        pattern = rf'\b{spec_type}\s+(?:STAR|CLASS)\b|\bCLASS\s+{spec_type}\b'
        if re.search(pattern, text_upper):
            matches.add(spec_type)
    
    return sorted(list(matches))


def extract_full_subclass_from_text(text: str) -> Optional[str]:
    """Extract full spectral subclass (e.g., 'G3', 'F0', 'K2') from text."""
    if not text:
        return None
    
    import re
    text_upper = text.upper()
    
    # Look for patterns like "G3", "F0", "K2 V", "M5 III", etc.
    # Pattern: spectral letter followed by a digit (0-9), optionally followed by space and luminosity class
    for spec_type in SPECTRAL_TYPES:
        # Match patterns like "G3", "G3 V", "G3-type", "G3 type star"
        pattern = rf'\b{spec_type}(\d)(?:\s*(?:V|IV|III|II|I|VI))?\b'
        match = re.search(pattern, text_upper)
        if match:
            # Return just the letter + number (e.g., "G3")
            return f"{spec_type}{match.group(1)}"
    
    return None


def extract_stellar_params_from_text(text: str) -> Dict[str, Optional[float]]:
    """Extract Teff, logg, and FeH values from generated text."""
    import re
    
    params = {'Teff': None, 'logg': None, 'FeH': None}
    
    if not text:
        return params
    
    # Teff patterns: "Teff of 4500 K", "temperature of 4500K", "temperature around 4500", "Teff = 4500 K"
    # Also handles "effective temperature (Teff) of..."
    teff_patterns = [
        r'(?:teff|temperature|effective temperature)(?:\s*\(.*?\))?(?:\s+of|\s+around|\s+is)?\s*[:=\s]*(\d{3,5})\s*k',
        r'(\d{3,5})\s*k.*?(?:teff|temperature)',
    ]
    for pattern in teff_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                val_str = match.group(1).replace(' ', '')
                params['Teff'] = float(val_str)
                break
            except (ValueError, IndexError):
                pass
    
    # logg patterns: "logg of 2.5", "log g of 2.5", "surface gravity of 2.5", "gravity around 4.0", "log g = 3. 70"
    # Also handles "surface gravity (logg) of..."
    logg_patterns = [
        r'(?:logg|log\s*g|surface gravity|gravity)(?:\s*\(.*?\))?(?:\s+of|\s+around|\s+is)?\s*[:=\s]*([0-5]\s*\.\s*\d{1,2})',
        r'(?:logg|log\s*g|gravity)(?:\s+of|\s+around)?\s*([0-5]\s*\.\s*\d{1,2})',
    ]
    for pattern in logg_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                val_str = match.group(1).replace(' ', '')
                params['logg'] = float(val_str)
                break
            except (ValueError, IndexError):
                pass
    
    # FeH patterns: "[Fe/H] of -0.5", "metallicity of -0.5", "FeH of -0.5", "FeH around -0.1", "[Fe/H] = 0. 11"
    # Also handles "metallicity ([Fe/H]) of..."
    feh_patterns = [
        r'\[fe/h\](?:\s+of|\s+around|\s+is)?\s*[:=\s]*(-?\s*\d+\s*\.\s*\d{1,2})',
        r'(?:metallicity|feh)(?:\s*\(.*?\))?(?:\s+of|\s+around|\s+is)?\s*[:=\s]*(-?\s*\d+\s*\.\s*\d{1,2})',
        r'feh(?:\s+content)?\s+around\s+(-?\s*\d+\s*\.\s*\d{1,2})',
    ]
    for pattern in feh_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                val_str = match.group(1).replace(' ', '')
                params['FeH'] = float(val_str)
                break
            except (ValueError, IndexError):
                pass
    
    return params



def is_valid_text(text: str) -> bool:
    """Check if text is valid (not gibberish token IDs)."""
    if not text:
        return False

    # Check if text is mostly numbers and commas (likely token IDs)
    text = text.strip()
    numeric_chars = sum(c.isdigit() or c in [',', ' ', '.'] for c in text)

    if len(text) > 0 and numeric_chars / len(text) > 0.7:
        return False

    # Check if text has reasonable word content
    words = text.split()
    if len(words) < 2:
        return False

    return True


def extract_value_from_text(text: str, unit: str = '') -> Optional[float]:
    """Extract a numeric value from text, handling optional unit and ranges."""
    if not text:
        return None
    
    import re
    text_lower = text.lower()
    
    # Define forbidden units/prefixes based on target extraction
    forbidden_units = []
    forbidden_prefixes = []
    max_val_cutoff = None
    
    if unit:
        if any(u in unit.lower() for u in ['msun', 'lsun', 'solar']):
            # For M/L, forbid Temperature(K), Radius(radii), Surface Gravity, Time
            forbidden_units = ['k', 'kelvin', 'radii', 'r_sun', 'rsun', 'dex', 'g/cm3', 'days', 'mag', 'solar radii', 'cm/s2']
            # Forbid numbers preceded explicitly by other params
            forbidden_prefixes = ['log g', 'logg', 'log(g)', 'gravity', 'surface', 'radius', 'teff', 'temperature', 'metallicity', 'fe/h', '[fe/h]']
            
        if any(u in unit.lower() for u in ['msun', 'solar mass']):
            # Target is Mass -> Forbid Luminosity
            forbidden_prefixes.append('luminosity')
            # Mass usually < 300 Msun for stars
            max_val_cutoff = 300.0
            
        if any(u in unit.lower() for u in ['lsun', 'solar luminosity']):
             # Target is Luminosity -> Forbid Mass
             forbidden_prefixes.append('mass')

        if any(u in unit.lower() for u in ['gyr', 'age']):
             # For Age, forbid Temperature(K), Msun, Lsun
             forbidden_units = ['k', 'kelvin', 'msun', 'lsun', 'r_sun', 'rsun', 'solar radii', 'cm/s2']
             forbidden_prefixes = ['log g', 'logg', 'gravity', 'surface', 'radius', 'teff', 'temperature', 'metallicity', 'fe/h', 'mass', 'luminosity']

    def _is_safe(val: float, start_idx: int, end_idx: int) -> bool:
        # 1. value sanity
        if max_val_cutoff is not None:
            if val > max_val_cutoff:
                return False
        if val > 10000000: # Global sanity
            return False
            
        # 2. forbidden units (following text)
        following_text = text_lower[end_idx:end_idx+20].strip()
        if any(following_text.startswith(bad_u) for bad_u in forbidden_units):
            return False
            
        # 3. forbidden prefixes (preceding text)
        prev_text = text_lower[max(0, start_idx-30):start_idx].strip()
        # Be careful not to match simple words if they are part of a sentence like "mass is determined by surface gravity"
        # We look for the forbidden key appearing *at the end* of the chunk (closest to number)
        # e.g. "... surface gravity of 4.5" -> "surface gravity of" ends with "of"
        # We check if any forbidden prefix is contained in the last few words
        if forbidden_prefixes:
            # Check if likely referring to forbidden param
            # e.g. "log g = ", "log g ", "gravity is ", "mass (M) is"
            
            # 1. Remove parenthesized/bracketed content: "(M)", "(log g)", "[Fe/H]"
            clean_prev = re.sub(r'\(.*?\)|\[.*?\]', '', prev_text)
            
            # 2. Remove connectors at the end
            clean_prev = re.sub(r'\s+(?:is|of|approx|approximately|about|estimated|to|be|=|:)\s*$', '', clean_prev)
            clean_prev = clean_prev.strip()
            
            # 3. Remove trailing punctuation commonly wrapping labels
            clean_prev = re.sub(r'[)\]}>,]+$', '', clean_prev).strip()
            
            if any(clean_prev.endswith(bad_p) for bad_p in forbidden_prefixes):
                return False
                
        return True

    # Unit variations
    unit_pattern = ''
    if unit:
        if unit.lower() in ['lsun', 'l_sun', 'solar luminosity']:
            unit_pattern = r'(?:\s*(?:lsun|l_sun|solar\s*luminosity|l\s*sun|solar))?'
        elif unit.lower() in ['msun', 'm_sun', 'solar mass']:
            unit_pattern = r'(?:\s*(?:msun|m_sun|solar\s*mass|m\s*sun|solar|mass))?'
        elif unit.lower() in ['gyr', 'age']:
             unit_pattern = r'(?:\s*(?:gyr|billion years|billion|years old))?'
    
    # Robust number regex: handles "1.23", "1. 0", "1 .23", "- 5. 2"
    number_re = r'([-+]?\s*\d+(?:\s*\.\s*\d+)?)(?:[eE][-+]?\d+)?'
    
    # Range pattern
    range_patterns = [
        rf'(?:between)\s*{number_re}\s*(?:and)\s*{number_re}{unit_pattern}',
        rf'{number_re}\s*(?:-|to)\s*{number_re}{unit_pattern}',
    ]

    for pattern in range_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                s1 = match.group(1).replace(' ', '')
                s2 = match.group(2).replace(' ', '')
                val1 = float(s1)
                val2 = float(s2)
                avg_val = (val1 + val2) / 2.0
                
                # Check safety on the range
                if _is_safe(avg_val, match.start(), match.end()):
                     return avg_val
            except (ValueError, IndexError):
                pass
    
    # Combine with context for single values
    patterns = [
        rf'(?:is|of|around|about|approximately|estimated\s*to\s*be)\s*{number_re}{unit_pattern}',
        rf'^{number_re}{unit_pattern}',
        rf'{number_re}{unit_pattern}\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                s_val = match.group(1).replace(' ', '')
                val = float(s_val)
                # Find where the number part was matched
                # Group 1 is the number. We need its span relative to original text
                # Re-finding it within the full match is safer or using pattern structure
                # The regex structure is Prefix + Number + Unit.
                # match.start(1) gives start of Number group
                if _is_safe(val, match.start(1), match.end(1)):
                    return val
            except (ValueError, IndexError):
                pass
                
    # Fallback: scan all numbers but apply strict filters
    candidates = []
    for match in re.finditer(number_re, text_lower):
        s_val = match.group(1).replace(' ', '')
        try:
            val = float(s_val)
        except ValueError:
            continue
        
        if _is_safe(val, match.start(), match.end()):
             candidates.append(val)

    if candidates:
        return candidates[0]
            
    return None



def analyze_followup_stellar_type(
    json_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze stellar type classification from followup_qa responses.

    Args:
        json_path: Path to JSON file with generation results
        output_dir: Directory to save plots (if None, uses same dir as json_path)

    Returns:
        Dictionary with analysis statistics
    """
    # Load results
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle different JSON formats
    # Format 1: List of experiments [{experiment_name: ..., samples: ...}, ...]
    # Format 2: Single result with {metadata: {...}, samples: [...]}
    if isinstance(data, list):
        all_results = data
    elif isinstance(data, dict) and 'samples' in data:
        # Single result format - extract experiment name from metadata or path
        exp_name = "default_experiment"
        if 'metadata' in data and isinstance(data['metadata'], dict):
            # Try to extract a meaningful name from checkpoint path
            checkpoint_path = data['metadata'].get('checkpoint_path', '')
            if checkpoint_path:
                # Extract directory name from path like "logs/regular_loss1p0_2025-11-27-16-44/..."
                import re
                match = re.search(r'logs/([^/]+)', checkpoint_path)
                if match:
                    exp_name = match.group(1)
        
        all_results = [{
            'experiment_name': exp_name,
            'samples': data['samples'],
            'metadata': data.get('metadata', {})
        }]
    else:
        raise ValueError(f"Unexpected JSON format. Expected list or dict with 'samples', got {type(data)}")

    print(f"Loaded results for {len(all_results)} experiments\n")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(json_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Extract statistics for each experiment
    experiment_stats = {}
    experiment_examples = {}  # Store examples for visualization

    # Load SNR map for fallback
    snr_map = load_snr_lookup()

    for exp_result in all_results:
        exp_name = exp_result['experiment_name']
        samples = exp_result['samples']

        print(f"\n{'='*80}")
        print(f"Processing {exp_name}")
        print(f"{'='*80}")
        print(f"Total samples: {len(samples)}")

        # Initialize statistics
        stats = {
            'total_stellar_type_questions': 0,
            'luminosity_questions': 0,
            'mass_questions': 0,
            'age_questions': 0,
            'valid_responses': 0,
            'lum_valid_responses': 0,
            'mass_valid_responses': 0,
            'age_valid_responses': 0,
            'invalid_responses': 0,
            'evol_class_correct': 0,
            'spectral_letter_correct': 0,
            'full_subclass_correct': 0,
            'correct_matches': 0,
            'incorrect_matches': 0,
            'extraction_failures': 0,
            'total_subclass_distance': 0, 
            'count_subclass_distance': 0
        }
        
        # Track stellar parameters for scatter plots
        param_data = {
            'true_Teff': [],
            'pred_Teff': [],
            'snr_Teff': [],
            'true_logg': [],
            'pred_logg': [],
            'snr_logg': [],
            'true_FeH': [],
            'pred_FeH': [],
            'snr_FeH': [],
            'true_Lstar': [],
            'pred_Lstar': [],
            'snr_Lstar': [],
            'true_Mstar': [],
            'pred_Mstar': [],
            'snr_Mstar': [],
            'true_Age': [],
            'pred_Age': [],
            'snr_Age': [],
        }

        # Store examples
        examples = {
            'correct': [],
            'incorrect': [],
            'invalid': [],
        }

        # Process each sample
        for idx ,sample in enumerate(samples):
            # print(f"\nProcessing sample {idx}")
            # Handle both old format (followup_qa) and new format (follow_up_answers)
            followup_qa = sample.get('followup_qa', sample.get('follow_up_answers', []))
            
            # Get stellar params to derive expected classification
            stellar_params = sample.get('stellar_params', {})
            
            # Get subclass for fine-grained spectral type comparison
            # Try to get from stellar_data first (new format), then fallback to top-level (old format)
            subclass = None
            stellar_data = sample.get('stellar_data')
            if stellar_data and isinstance(stellar_data, dict):
                subclass = stellar_data.get('subclass')
            if not subclass:
                subclass = sample.get('subclass', None)

            # Get original question and answer for context
            original_question = sample.get('dataset_question', '')
            original_answer = sample.get('model_answer', '')
            
            for idx, qa in enumerate(followup_qa):
                pred_L = None
                # Only look at stellar type questions OR luminosity/mass
                question = qa.get('question', '')
                q_type = qa.get('type', '')
                
                # Check for relevant keywords
                is_relevant = any(k in question.lower() for k in ['stellar', 'classification', 'type', 'luminosity', 'mass'])
                if not is_relevant:
                    continue

                # Check if this is a stellar type question (not just any "type" question)
                is_stellar_type_q = 'type' in q_type.lower()
                # print(idx, " type: ", q_type, is_stellar_type_q)

                if not is_stellar_type_q:
                    # Check for Lstar/Mstar questions
                    if 'luminosity' in question.lower():
                        is_luminosity_q = True
                    else:
                        is_luminosity_q = False
                        
                    if 'mass' in question.lower() and 'stellar mass' not in question.lower(): # Basic check, refine if needed
                         # "stellar mass" might be in "what is the stellar mass?"
                         is_mass_q = True
                    elif 'mass' in question.lower():
                        is_mass_q = True
                    else:
                        is_mass_q = False

                    if 'age' in question.lower() or ('age' in q_type.lower() if q_type else False):
                        is_age_q = True
                    else:
                        is_age_q = False

                    if not (is_luminosity_q or is_mass_q or is_age_q):
                        continue
                else:
                    is_luminosity_q = False
                    is_mass_q = False
                    is_age_q = False

                if is_stellar_type_q:
                    stats['total_stellar_type_questions'] += 1
                if is_luminosity_q:
                    stats['luminosity_questions'] += 1
                if is_mass_q:
                    stats['mass_questions'] += 1
                if is_age_q:
                    stats['age_questions'] += 1

                # Get generated answer - handle both formats
                generated = qa.get('generated_response', qa.get('answer', ''))
                
                # Get expected answer - handle both formats
                expected = qa.get('expected_answer', '')
                
                # Derive expected stellar type from stellar_params for evolutionary class
                expected_evol_class = None
                if stellar_params:
                    teff = stellar_params.get('Teff', 0)
                    logg = stellar_params.get('logg', 0)
                    
                    # Classify based on logg and teff (simple heuristic)
                    if logg < 3.5:  # Low surface gravity
                        if logg < 1.0:
                            expected_evol_class = "supergiant"
                        else:
                            expected_evol_class = "giant"
                    elif logg > 4.5:  # High surface gravity
                        expected_evol_class = "white dwarf"
                    else:  # Main sequence range
                        if 3.5 <= logg <= 4.5:
                            expected_evol_class = "dwarf"  # main sequence
                        else:
                            expected_evol_class = "subgiant"
                
                # Get expected spectral type from subclass (e.g., 'G3' -> 'G')
                expected_spectral = None
                if subclass and isinstance(subclass, str) and len(subclass) > 0:
                    # Extract first letter (spectral type)
                    first_char = subclass[0].upper()
                    if first_char in SPECTRAL_TYPES:
                        expected_spectral = first_char

                # Check if response is valid text
                if not is_valid_text(generated):
                    stats['invalid_responses'] += 1
                    if len(examples['invalid']) < 10:  # Store up to 10 examples
                        examples['invalid'].append({
                            'question': question,
                            'expected': expected or expected_evol_class or expected_spectral or '',
                            'generated': generated[:200],  # Truncate long responses
                            'sample_idx': sample.get('sample_idx', sample.get('sample_index', -1)),
                        })

                stats['valid_responses'] += 1

                # Extract stellar types from generated text
                generated_evol_type = extract_stellar_type_from_text(generated)
                generated_subclass = extract_full_subclass_from_text(generated)
                
                # If subclass is found (e.g. "F6"), derive spectral type ("F") from it
                # This overrides the text-based spectral type extraction if they differ,
                # or fills it in if missing.
                generated_spectral_candidates = set()
                if generated_subclass and generated_subclass[0] in SPECTRAL_TYPES:
                    generated_spectral_candidates.add(generated_subclass[0])
                
                # Extract text-based mentions (now a list)
                text_spectrals = extract_spectral_type_from_text(generated)
                if text_spectrals:
                    generated_spectral_candidates.update(text_spectrals)
                
                # Convert to comma-separated string for compatibility with logging/examples
                generated_spectral = ",".join(sorted(list(generated_spectral_candidates))) if generated_spectral_candidates else None
                
                # Extract stellar parameters from generated text
                extracted_params = extract_stellar_params_from_text(generated)

                # Get SNR (try JSON first, then lookup)
                snr = qa.get('snr')
                if snr is None:
                    # Fallback lookup
                    obsid = sample.get('obsid')
                    if obsid:
                        snr = snr_map.get(str(obsid))
                
                if snr is None:
                    snr = 0.0

                # Collect parameter data for scatter plots
                if stellar_params.get('Teff') is not None and extracted_params['Teff'] is not None:
                    # print("teff: ", extracted_params['Teff'], "true: ", stellar_params['Teff'])
                    if extracted_params['Teff'] > 1000: # remove invalid values
                        param_data['true_Teff'].append(stellar_params['Teff'])
                        param_data['pred_Teff'].append(extracted_params['Teff'])
                        param_data['snr_Teff'].append(snr)
                if stellar_params.get('logg') is not None and extracted_params['logg'] is not None:
                    param_data['true_logg'].append(stellar_params['logg'])
                    param_data['pred_logg'].append(extracted_params['logg'])
                    param_data['snr_logg'].append(snr)
                if stellar_params.get('FeH') is not None and extracted_params['FeH'] is not None:
                    param_data['true_FeH'].append(stellar_params['FeH'])
                    param_data['pred_FeH'].append(extracted_params['FeH'])
                    param_data['snr_FeH'].append(snr)
                
                # Lstar/Mstar extraction
                if is_luminosity_q:
                    pred_L = extract_value_from_text(generated, unit='Lsun')
                    true_L = qa.get('true_value')
                    if true_L is not None and pred_L is not None:
                        if pred_L < 1000:
                            # Linear vs Linear Comparison (as requested)
                            # Linear vs Linear Comparison (as requested)
                            param_data['true_Lstar'].append(float(true_L))
                            param_data['pred_Lstar'].append(pred_L)
                            param_data['snr_Lstar'].append(snr)
                        
                        stats['lum_valid_responses'] += 1
                        
                if is_mass_q:
                    pred_M = extract_value_from_text(generated, unit='Msun')
                    true_M = qa.get('true_value')
                    if true_M is not None and pred_M is not None:
                        param_data['true_Mstar'].append(float(true_M))
                        param_data['pred_Mstar'].append(pred_M)
                        param_data['snr_Mstar'].append(snr)
                        stats['mass_valid_responses'] += 1
                    elif true_M is not None:
                         # Track failed extraction if needed, but for now just count valid
                         pass
                
                if is_age_q:
                    pred_Age = extract_value_from_text(generated, unit='Gyr')
                    true_Age = qa.get('true_value')
                    
                    if true_Age is not None and pred_Age is not None:
                         param_data['true_Age'].append(float(true_Age))
                         param_data['pred_Age'].append(pred_Age)
                         param_data['snr_Age'].append(snr)
                         stats['age_valid_responses'] += 1

                if is_luminosity_q:
                    # Update valid count if we extracted something
                    pass # Handled above now

                # Expected full subclass (normalize to uppercase)
                expected_subclass = subclass.upper() if subclass and isinstance(subclass, str) else None

                if pred_L is not None and pred_L > 1000:
                    print(f"==========={idx}==========")
                    print(f"Generated: {generated}")
                    print(f"Generated evol type: {generated_evol_type}")
                    print(f"Generated spectral: {generated_spectral}")
                    print(f"Generated subclass: {generated_subclass}")
                    print(f"Expected: {expected}")
                    print(f"Expected evol type: {expected_evol_class}")
                    print(f"Expected spectral: {expected_spectral}")
                    print(f"Expected subclass: {expected_subclass}")
                    print("extracted Lsun: ", pred_L)
                    print("true Lsun: ", true_L)

                # Check matches at three levels
                # Level 1 (High): Evolutionary class (dwarf/giant/etc)
                evol_match = (expected_evol_class is not None and
                            generated_evol_type is not None and
                            expected_evol_class == generated_evol_type)

                # Level 2 (Mid): Spectral type letter (O/B/A/F/G/K/M)
                spectral_match = False
                if expected_spectral is not None and generated_spectral is not None:
                    candidates = generated_spectral.split(',')
                    spectral_match = expected_spectral in candidates
                
                # Level 3 (Fine): Full subclass (G3/F0/K2/etc)
                subclass_match = (expected_subclass is not None and
                                generated_subclass is not None and
                                expected_subclass == generated_subclass)

                # Track each level separately
                if evol_match:
                    stats['evol_class_correct'] += 1
                if spectral_match:
                    stats['spectral_letter_correct'] += 1
                if subclass_match:
                    stats['full_subclass_correct'] += 1

                # Overall: correct if ANY level matches
                is_correct = evol_match or spectral_match or subclass_match

                if is_correct:
                    stats['correct_matches'] += 1
                    if len(examples['correct']) < 10:
                        examples['correct'].append({
                            'original_question': original_question,
                            'original_answer': original_answer,
                            'question': question,
                            'subclass': subclass,
                            'expected_spectral': expected_spectral,
                            'expected_evol_class': expected_evol_class,
                            'generated': generated,
                            'sample_idx': sample.get('sample_idx', sample.get('sample_index', -1)),
                            'generated_evol_type': generated_evol_type,
                            'generated_spectral': generated_spectral,
                            'generated_subclass': generated_subclass,
                            'Teff': stellar_params.get('Teff', 'N/A'),
                            'logg': stellar_params.get('logg', 'N/A'),
                            'FeH': stellar_params.get('FeH', 'N/A'),
                        })
                else:
                    if generated_evol_type is None and generated_spectral is None and generated_subclass is None:
                        stats['extraction_failures'] += 1
                    else:
                        stats['incorrect_matches'] += 1

                    if len(examples['incorrect']) < 10:
                        examples['incorrect'].append({
                            'original_question': original_question,
                            'original_answer': original_answer,
                            'question': question,
                            'subclass': subclass,
                            'expected_spectral': expected_spectral,
                            'expected_evol_class': expected_evol_class,
                            'generated': generated,
                            'sample_idx': sample.get('sample_idx', sample.get('sample_index', -1)),
                            'generated_evol_type': generated_evol_type,
                            'generated_spectral': generated_spectral,
                            'generated_subclass': generated_subclass,
                            'Teff': stellar_params.get('Teff', 'N/A'),
                            'logg': stellar_params.get('logg', 'N/A'),
                            'FeH': stellar_params.get('FeH', 'N/A'),
                        })
        

        # Calculate accuracy
        if stats['total_stellar_type_questions'] > 0:
            accuracy = (stats['correct_matches'] / stats['total_stellar_type_questions']) * 100
            valid_rate = (stats['valid_responses'] / stats['total_stellar_type_questions']) * 100
            # Three-level accuracies
            evol_acc = (stats['evol_class_correct'] / stats['total_stellar_type_questions']) * 100
            spectral_acc = (stats['spectral_letter_correct'] / stats['total_stellar_type_questions']) * 100
            subclass_acc = (stats['full_subclass_correct'] / stats['total_stellar_type_questions']) * 100
        else:
            accuracy = 0
            valid_rate = 0
            evol_acc = spectral_acc = subclass_acc = 0

        stats['accuracy'] = accuracy
        stats['valid_rate'] = valid_rate
        stats['evol_class_accuracy'] = evol_acc
        stats['spectral_letter_accuracy'] = spectral_acc
        stats['full_subclass_accuracy'] = subclass_acc

        experiment_stats[exp_name] = stats
        experiment_examples[exp_name] = examples

        # Print summary
        # print(f"\nStellar Type Questions: {stats['total_stellar_type_questions']}")
        # print(f"Valid Responses: {stats['valid_responses']} ({valid_rate:.1f}%)")
        # print(f"Invalid/Gibberish Responses: {stats['invalid_responses']}")
        # print(f"Overall Correct Matches (any level): {stats['correct_matches']}")
        # print(f"Incorrect Matches: {stats['incorrect_matches']}")
        # print(f"Extraction Failures: {stats['extraction_failures']}")
        # print(f"Overall Accuracy (of valid responses): {accuracy:.1f}%")
        # print(f"\nThree-Level Breakdown:")
        # print(f"  High-Level (Evol. Class - dwarf/giant/etc.): {stats['evol_class_correct']}/{stats['total_stellar_type_questions']} ({evol_acc:.1f}%)")
        # print(f"  Mid-Level (Spectral Letter - G/F/K/etc.): {stats['spectral_letter_correct']}/{stats['total_stellar_type_questions']} ({spectral_acc:.1f}%)")
        # print(f"  Mid-Level (Spectral Letter - G/F/K/etc.): {stats['spectral_letter_correct']}/{stats['total_stellar_type_questions']} ({spectral_acc:.1f}%)")
        # print(f"  Fine-Grained (Full Subclass - G3/F0/K2/etc.): {stats['full_subclass_correct']}/{stats['total_stellar_type_questions']} ({subclass_acc:.1f}%)")
        
        # print(f"\nLuminosity Questions: {stats['luminosity_questions']}")
        # print(f"  Valid Extraction: {stats['lum_valid_responses']}")
        # print(f"Mass Questions: {stats['mass_questions']}")
        # print(f"  Valid Extraction: {stats['mass_valid_responses']}")

    # Create plots
    create_followup_plots(experiment_stats, experiment_examples, param_data, output_dir)

    return experiment_stats


def create_followup_plots(
    experiment_stats: Dict[str, Dict],
    experiment_examples: Dict[str, Dict],
    param_data: Dict[str, list],
    output_dir: Path,
) -> None:
    """Create comparison plots and example visualizations."""
    import textwrap
    
    def wrap_text(text, width=80):
        """Wrap text to specified width"""
        if not text or text == 'N/A':
            return text
        return '\n'.join(textwrap.fill(line, width=width) for line in text.split('\n'))

    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 14

    exp_names = list(experiment_stats.keys())

    # Short labels
    short_labels = {}
    for name in exp_names:
        label = name.replace('ablation_s_', '').replace('_', ' ')
        short_labels[name] = label

    # ============================================================================
    # Plot 1: Accuracy and Valid Response Rate
    # ============================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(exp_names))
    width = 0.35

    accuracies = [experiment_stats[exp]['accuracy'] for exp in exp_names]
    valid_rates = [experiment_stats[exp]['valid_rate'] for exp in exp_names]

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (Correct/Total)',
                   color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, valid_rates, width, label='Valid Response Rate',
                   color='coral', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Followup Stellar Type Question Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([short_labels[name] for name in exp_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'followup_stellar_type_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'followup_stellar_type_accuracy.png'}")

    # ============================================================================
    # Plot 2: Response Breakdown (stacked bar)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    correct_counts = [experiment_stats[exp]['correct_matches'] for exp in exp_names]
    incorrect_counts = [experiment_stats[exp]['incorrect_matches'] for exp in exp_names]
    extraction_fail_counts = [experiment_stats[exp]['extraction_failures'] for exp in exp_names]
    invalid_counts = [experiment_stats[exp]['invalid_responses'] for exp in exp_names]

    ax.bar(x, correct_counts, label='Correct', color='green', alpha=0.7, edgecolor='black')
    ax.bar(x, incorrect_counts, bottom=correct_counts, label='Incorrect',
           color='orange', alpha=0.7, edgecolor='black')
    bottom = np.array(correct_counts) + np.array(incorrect_counts)
    ax.bar(x, extraction_fail_counts, bottom=bottom, label='Extraction Failed',
           color='yellow', alpha=0.7, edgecolor='black')
    bottom = bottom + np.array(extraction_fail_counts)
    ax.bar(x, invalid_counts, bottom=bottom, label='Invalid/Gibberish',
           color='red', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Followup Stellar Type Response Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([short_labels[name] for name in exp_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'followup_response_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'followup_response_breakdown.png'}")

    # ============================================================================
    # Plot 3: Example Visualizations (2 Random Examples)
    # ============================================================================
    import random
    
    for exp_name, examples in experiment_examples.items():
        # Combine all examples (correct + incorrect) and randomly sample 4
        all_examples = examples.get('correct', []) + examples.get('incorrect', [])
        if len(all_examples) < 4:
            print(f"Not enough examples for {exp_name}, skipping example visualization")
            continue
        
        random_examples = random.sample(all_examples, min(4, len(all_examples)))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Example Responses: {exp_name}', fontsize=14, fontweight='bold')
        
        for idx, (ax, ex) in enumerate(zip(axes.flatten(), random_examples)):
            ax.axis('off')
            
            # Determine if this example was correct
            is_correct = ex in examples.get('correct', [])
            title_color = 'green' if is_correct else 'orange'
            # status = '✓ Correct' if is_correct else '✗ Incorrect'
            
            ax.set_title(f'Sample {ex["sample_idx"]}', 
                        fontsize=12, fontweight='bold', color=title_color, loc='left')
            
            text_lines = []
            text_lines.append(f"=== Original Question ===")
            text_lines.append(wrap_text(ex.get('original_question', 'N/A'), 90))
            text_lines.append("")
            text_lines.append(f"=== Original Answer ===")
            text_lines.append(wrap_text(ex.get('original_answer', 'N/A'), 90))
            text_lines.append("")
            text_lines.append(f"=== Followup Question ===")
            text_lines.append(wrap_text(ex['question'], 90))
            text_lines.append("")
            text_lines.append(f"=== Followup Answer ===")
            text_lines.append(wrap_text(ex['generated'], 90))
            text_lines.append("")
            text_lines.append(f"=== True Stellar Parameters ===")
            text_lines.append(f"Subclass: {ex.get('subclass', 'N/A')}")
            text_lines.append(f"Teff: {ex.get('Teff', 'N/A'):.0f} K")
            text_lines.append(f"logg: {ex.get('logg', 'N/A'):.2f}")
            text_lines.append(f"[Fe/H]: {ex.get('FeH', 'N/A'):.2f}")
            text_lines.append("")
            # text_lines.append(f"=== Extracted Classifications ===")
            # text_lines.append(f"Full Subclass: {ex.get('generated_subclass', 'N/A')}")
            # text_lines.append(f"Spectral Letter: {ex.get('generated_spectral', 'N/A')}")
            # text_lines.append(f"Evolutionary Type: {ex.get('generated_evol_type', 'N/A')}")
            
            text = "\n".join(text_lines)
            bg_color = 'lightgreen' if is_correct else 'lightyellow'
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.3),
                   wrap=True)
        
        plt.tight_layout()
        safe_name = exp_name.replace('/', '_')
        plt.savefig(output_dir / f'examples_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'examples_{safe_name}.png'}")
    
    # ============================================================================
    # Plot: Stellar Parameter Scatter Plots
    # ============================================================================
    plot_keys = ['true_Teff', 'true_logg', 'true_FeH', 'true_Lstar', 'true_Mstar']
    if param_data and any(len(param_data.get(key, [])) > 0 for key in plot_keys):
        # Determine number of plots
        valid_plots = []
        if len(param_data.get('true_Teff', [])) > 0:
            valid_plots.append(('Teff', 'Effective Temperature (K)', param_data['true_Teff'], param_data['pred_Teff'], param_data['snr_Teff']))
        if len(param_data.get('true_logg', [])) > 0:
            valid_plots.append(('logg', 'Surface Gravity (log g)', param_data['true_logg'], param_data['pred_logg'], param_data['snr_logg']))
        if len(param_data.get('true_FeH', [])) > 0:
            valid_plots.append(('FeH', 'Metallicity ([Fe/H])', param_data['true_FeH'], param_data['pred_FeH'], param_data['snr_FeH']))
        if len(param_data.get('true_Lstar', [])) > 0:
            valid_plots.append(('Lstar', 'Luminosity (Lsun)', param_data['true_Lstar'], param_data['pred_Lstar'], param_data['snr_Lstar']))
        if len(param_data.get('true_Mstar', [])) > 0:
            valid_plots.append(('Mstar', 'Mass (Msun)', param_data['true_Mstar'], param_data['pred_Mstar'], param_data['snr_Mstar']))
        if len(param_data.get('true_Age', [])) > 0:
            valid_plots.append(('Age', 'Age (Gyr)', param_data['true_Age'], param_data['pred_Age'], param_data['snr_Age']))
         
        n_plots = len(valid_plots)
        cols = min(n_plots, 3)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        fig.suptitle('Stellar Parameters: Predicted vs True', fontsize=14, fontweight='bold')
        
        # Flatten axes for easy iteration if multiple
        if n_plots > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
            
        for idx, (param_name, param_label, true_vals, pred_vals, snr_vals) in enumerate(valid_plots):
            ax = axes_flat[idx]
            # print(f"Param: {param_name}")
            # print(f"True: {true_vals}")
            # print(f"Pred: {pred_vals}")
            
            if len(true_vals) > 0 and len(pred_vals) > 0:
                # Scatter plot with SNR coloring
                sc = ax.scatter(true_vals, pred_vals, c=snr_vals, cmap='viridis', alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
                # Add colorbar
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label('SNR')
                
                # Diagonal line (perfect prediction)
                min_val = min(min(true_vals), min(pred_vals))
                max_val = max(max(true_vals), max(pred_vals))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
                
                # Calculate R² and RMSE
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                if len(true_vals) > 1:
                    r2 = r2_score(true_vals, pred_vals)
                    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
                    mae = mean_absolute_error(true_vals, pred_vals)
                    title_str = f'{param_name}\nR²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}'
                else:
                    title_str = f'{param_name}'
                
                ax.set_xlabel(f'True {param_label}', fontsize=11)
                ax.set_ylabel(f'Predicted {param_label}', fontsize=11)
                ax.set_title(title_str, fontsize=12, fontweight='bold')
                
                # Set Log-Log scale for Luminosity and Mass
                if param_name in ['Lstar', 'Mstar']:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    
                ax.legend(loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3, which="both")
                
        # Hide unused subplots
        for idx in range(n_plots, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'stellar_params_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'stellar_params_scatter.png'}")


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze followup stellar type classification"
    )
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='Path to followup_comparison.json file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save plots (default: same as json_path)'
    )

    args = parser.parse_args()

    print("="*80)
    print("ANALYZING FOLLOWUP STELLAR TYPE CLASSIFICATION")
    print("="*80)
    print(f"Input: {args.json_path}")
    print(f"Output: {args.output_dir or Path(args.json_path).parent}")
    print()

    stats = analyze_followup_stellar_type(
        json_path=args.json_path,
        output_dir=args.output_dir,
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
