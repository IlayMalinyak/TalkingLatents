import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import json
import math
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, Any, List, Type
import os
from pathlib import Path
from astropy.io import fits
import re
from typing import Optional, Tuple, Dict, Any, List, Type, Union

import os
os.system('pip install tiktoken fairscale fire blobfile')
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from llama3.llama.tokenizer import Tokenizer
from data.transforms import RandomMasking
from data.feature_normalizer import FeatureNormalizer
from src.follow_up_templates import create_follow_up_specs


class StellarQuestionsDataset(Dataset):
    """
    PyTorch Dataset for stellar descriptions and optional spectral features
    Now includes tokenization for LLaMA

    Args:
        json_file (str): Path to the JSON file with stellar data
        features_array (Optional[np.ndarray]): Optional array of spectral features
        split (str): One of 'train', 'val', 'test'
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set (remaining after train/val)
        random_state (int): Random seed for reproducible splits
        filter_valid_descriptions (bool): Whether to filter out samples with no description
        cache_dir (Optional[str]): Directory to cache split indices for consistency
        tokenizer_path (Optional[str]): Path to SentencePiece tokenizer model
        max_length (int): Maximum sequence length for tokenization
        enable_followup (bool): Whether to append follow-up questions
        followup_json_file (Optional[str]): Path to second JSON for description-based follow-ups
        followup_mode (str): "stellar_type", "description", or "mixed" (50/50 default)
    """
    
    def __init__(self, 
                 json_file: str,
                 features_array: Optional[np.ndarray] = None,
                 split: str = 'train',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42,
                 spectral_transforms: Optional[Any] = None,
                 filter_valid_descriptions: bool = True,
                 cache_dir: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 tokenizer: Optional[Any] = None,
                 tokenizer_backend: str = 'llama',
                 max_length: int = 512,
                 num_spectral_features: int = 1,
                 normalize_features: bool = True,
                 feature_stats: Optional[Dict[str, np.ndarray]] = None,
                 feature_norm_epsilon: float = 1e-6,
                 enable_followup: bool = False,
                 followup_prob: float = 0.0,
                 max_followup_turns: int = 1,
                 followup_seed: int = 42,
                 # Optional second JSON for follow-up Q&A
                 followup_json_file: Optional[str] = None,
                 followup_mode: str = "mixed",
                 multimodal_df: Optional[pd.DataFrame] = None,
                 index_df: Optional[pd.DataFrame] = None):
        
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.json_file = json_file
        self.features_array = features_array
        self.num_spectral_features = num_spectral_features
        self.split = split
        self.random_state = random_state
        self.filter_valid_descriptions = filter_valid_descriptions
        self.tokenizer_path = tokenizer_path
        self.tokenizer_backend = tokenizer_backend
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.transforms = spectral_transforms
        self.mask_transform = RandomMasking()  # Example masking
        self.normalize_features = normalize_features and (self.features_array is not None)
        self.feature_norm_epsilon = feature_norm_epsilon
        self.feature_normalizer = FeatureNormalizer(
            enabled=self.normalize_features,
            epsilon=self.feature_norm_epsilon,
        )
        self.enable_followup = enable_followup
        self.followup_prob = followup_prob
        self.max_followup_turns = max_followup_turns
        self.followup_json_file = followup_json_file
        self.followup_mode = followup_mode
        seed_offset = followup_seed + hash((split, random_state))
        self.followup_rng = random.Random(seed_offset)

        # Load follow-up JSON if provided
        self.followup_data = None
        if self.followup_json_file and self.enable_followup:
            self._load_followup_json()

        self.multimodal_df = multimodal_df
        if self.multimodal_df is not None:
            # Ensure index for faster lookup if possible, but user might pass raw dataframe
            # We'll assume it's indexable by obsid or we find the row using a column
            print(f"Loaded multimodal dataframe with {len(self.multimodal_df)} rows")

        self.index_df = index_df
        self.obsid_to_feature_idx = {}
        if self.index_df is not None:
             print(f"Loaded index dataframe for feature alignment with {len(self.index_df)} rows")
             # Build lookup map: obsid -> row_idx
             # Handle potential column name variations if needed, but assuming 'obsid' based on user request
             if 'obsid' in self.index_df.columns:
                 # Create mapping for both string and int versions to be robust
                 for idx, row in self.index_df.iterrows():
                     obsid_val = row['obsid']
                     self.obsid_to_feature_idx[str(obsid_val)] = idx
                     try:
                         self.obsid_to_feature_idx[int(obsid_val)] = idx
                     except (ValueError, TypeError):
                         pass
             else:
                 print("Warning: index_df provided but 'obsid' column not found.")

        # Print follow-up mode info
        if self.enable_followup:
            if self.followup_data is not None:
                print(f"Follow-up mode: {self.followup_mode} (with {len(self.followup_data)} description samples loaded)")
            else:
                print(f"Follow-up mode: stellar_type only (no followup_json_file provided)")
        
        self.numeric_bounds = {
            'Teff': (3000.0, 7500.0),
            'logg': (0.0, 5.0),
            'FeH': (-3.0, 0.5),
        }
        self.numeric_key_alternatives = {
            'Teff': ['Teff', 'teff_k', 'teff', 'effective_temperature'],
            'logg': ['logg', 'log_g'],
            'FeH': ['FeH', 'feh', '[Fe/H]', 'metallicity'],
        }

        # Load tokenizer if not provided
        if self.tokenizer is None:
            self._load_tokenizer()
        
        # Load and process data
        self._load_data()
        self._create_splits(train_ratio, val_ratio, test_ratio, cache_dir)
        self._initialize_feature_normalizer(feature_stats)
        
    def _load_followup_json(self):
        """Load the follow-up JSON file for Q&A pairs."""
        try:
            with open(self.followup_json_file, 'r') as f:
                raw_followup = json.load(f)
            
            # Convert list to dict keyed by obsid
            self.followup_data = {}
            count = 0
            for item in raw_followup:
                # Try to find obsid in top level or inside stellar_data
                obsid = item.get("obsid")
                if obsid is None and "stellar_data" in item:
                    obsid = item["stellar_data"].get("obsid")
                
                if obsid is not None:
                    # Store by integer obsid if possible
                    try:
                        obsid = int(obsid)
                    except (ValueError, TypeError):
                        pass
                    self.followup_data[obsid] = item
                    count += 1
                    
            print(f"Loaded {count} follow-up samples from {self.followup_json_file} (indexed by obsid)")
        except Exception as e:
            print(f"Warning: Could not load follow-up JSON {self.followup_json_file}: {e}")
            self.followup_data = None

    def _get_followup_from_description(self, obsid: int) -> Optional[Tuple[str, str]]:
        """Extract question and answer from the follow-up JSON file by obsid."""
        if self.followup_data is None:
            return None
        
        # Ensure obsid is int/consistent
        try:
            obsid = int(obsid)
        except (ValueError, TypeError):
            print(f"Warning: Invalid obsid {obsid} (not an integer)")
            pass

        followup_sample = self.followup_data.get(obsid)
        if followup_sample is None:
            print(f"Warning: No follow-up sample found for obsid {obsid}")
            return None

        description = followup_sample.get("description", "")

        if not description:
            print(f"Warning: No description found for obsid {obsid}")
            return None

        # Parse the description to get question and answer
        parsed = self.parse_description_text(description)
        question = parsed.get("question", "")
        answer = parsed.get("answer", "")

        if not question or not answer:
            print(f"Warning: No question or answer found for obsid {obsid}")
            return None

        return question, answer

    def _load_tokenizer(self):
        """Load SentencePiece tokenizer if available"""
        if self.tokenizer_path and os.path.exists(self.tokenizer_path):
            try:
                self.tokenizer = Tokenizer(model_path=self.tokenizer_path)
                print(f"Loaded tokenizer from {self.tokenizer_path}")
            except ImportError:
                print("SentencePiece not available. Install with: pip install sentencepiece")
                self.tokenizer = None
            except Exception as e:
                print(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
        else:
            print(f"Tokenizer path not found: {self.tokenizer_path}")
            self.tokenizer = None
    
    def _tokenize_text(self, text: str, bos=True) -> torch.Tensor:
        """Tokenize text using tokenizer or fallback"""
        if self.tokenizer is not None:
            # Use real tokenizer
            token_ids = self.tokenizer.encode(text, bos=bos, eos=False)
            num_tokens = len(token_ids)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                # Pad with pad token (usually 0)
                pad_token_id = self.tokenizer.pad_id if hasattr(self.tokenizer, 'pad_id') else 0
                token_ids = token_ids + [pad_token_id] * (self.max_length - len(token_ids))
            return torch.tensor(token_ids, dtype=torch.long), num_tokens
        else:
            print("Tokenizer not available, using fallback tokenization.")
            # Fallback: create deterministic tokens based on text
            # Simple hash-based tokenization for consistent results
            words = text.lower().split()
            token_ids = []
            num_tokens = 0
            
            for word in words:
                # Create consistent token ID from word hash
                word_hash = hash(word) % 10000  # Limit vocab size
                token_ids.append(abs(word_hash) + 1)  # Avoid 0 (pad token)
                num_tokens += 1
                if len(token_ids) >= self.max_length:
                    break
            
            # Pad to max_length
            while len(token_ids) < self.max_length:
                token_ids.append(0)  # Pad token
                
            return torch.tensor(token_ids[:self.max_length], dtype=torch.long), num_tokens
        
    def _tokenize_text_no_pad(self, text: str, bos=True) -> Tuple[List[int], int]:
        """Tokenize text without padding - return raw token list"""
        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(text, bos=bos, eos=False)
            return token_ids, len(token_ids)
        else:
            # Fallback: create deterministic tokens based on text
            words = text.lower().split()
            token_ids = []
            
            for word in words:
                word_hash = hash(word) % 10000
                token_ids.append(abs(word_hash) + 1)

            return token_ids, len(token_ids)

    def _extend_with_tokens(self,
                            full_tokens: List[int],
                            target_tokens: List[int],
                            tokens: List[int],
                            mask_targets: bool) -> None:
        if not tokens:
            return
        available = self.max_length - len(full_tokens)
        if available <= 0:
            return
        chunk = tokens[:available]
        full_tokens.extend(chunk)
        if mask_targets:
            target_tokens.extend([-100] * len(chunk))
        else:
            target_tokens.extend(chunk)

    def _extract_physical_params(self, stellar_data: Dict[str, Any], obsid: Optional[int] = None) -> Dict[str, Optional[float]]:
        """Return raw Teff/logg/FeH values when available. Also merges from multimodal_df if preset."""
        params: Dict[str, Optional[float]] = {}
        
        # 1. Basic params from internal JSON
        if isinstance(stellar_data, dict):
             for param in ['Teff', 'logg', 'FeH']:
                value = None
                for key in self.numeric_key_alternatives.get(param, [param]):
                    raw_val = stellar_data.get(key)
                    if raw_val is not None:
                        try:
                            value = float(raw_val)
                        except (TypeError, ValueError):
                            value = None
                        break
                params[param] = value
        
        # 2. Add multimodal params if available
        if self.multimodal_df is not None and obsid is not None:
            try:
                row = None
                row = self.multimodal_df[self.multimodal_df['obsid'] == obsid]
                if row is not None:
                    # Extract required fields: 'binarity_class_hard', 'final_age', 'age_ref', 'Age'
                    for col in ['binarity_class_hard', 'final_age', 'age_ref', 'Age']:
                        if col in row:
                            val = row[col]
                            # Handle Series vs scalar
                            if hasattr(val, 'item'):
                                val = val.item()
                            params[col] = val
            except Exception as e:
                # Be robust
                pass

        return params

    def _append_followup_turns(self,
                               full_tokens: List[int],
                               target_tokens: List[int],
                               stellar_params: Dict[str, Optional[float]],
                               sample_idx: int = -1,
                               obsid: Optional[int] = None) -> List[Tuple[str, str]]:
        """Append followup turns and return list of (question, answer) text pairs."""
        if not self.enable_followup or self.followup_prob <= 0.0:
            return []
        if self.followup_rng.random() > self.followup_prob:
            return []

        followups = []

        if self.multimodal_df is not None:
             # Multimodal Followup Logic
             choice = self.followup_rng.choice(['binarity', 'age', 'stellar_type'])
             
             if choice == 'binarity':
                 q_text = "Is this star a binary?"
                 bin_class = stellar_params.get('binarity_class_hard')
                 if bin_class == 1:
                     a_text = "Yes"
                 elif bin_class == 2:
                     a_text = "No"
                 else:
                     # nan or other -> probably no
                     a_text = "Probably no"
                 followups.append({'question': q_text, 'answer': a_text})

             elif choice == 'age':
                 q_text = "What is the age of this star in Gyrs?"
                 final_age = stellar_params.get('final_age')
                 age_val = None
                 
                 # Logic: use final_age if not nan
                 if final_age is not None and not (isinstance(final_age, float) and math.isnan(final_age)):
                     age_val = final_age
                 else:
                     # fallback to Age and set age_ref
                     age = stellar_params.get('Age')
                     if age is not None and not (isinstance(age, float) and math.isnan(age)):
                         age_val = age
                         stellar_params['age_ref'] = 'isochrone'
                 
                 if age_val is not None:
                     a_text = f"{float(age_val):.2f}"
                 else:
                     a_text = "Unknown"
                 
                 followups.append({'question': q_text, 'answer': a_text})

             else: # stellar_type
                  stellar_type_followups = create_follow_up_specs(
                    stellar_params,
                    self.followup_rng,
                    max_pairs=1, 
                    include_answers=True,
                )
                  if stellar_type_followups:
                      followups.append(stellar_type_followups[0])

        else:
            # ORIGINAL LOGIC
            # FIRST followup: Generate stellar type question from templates
            stellar_type_followups = create_follow_up_specs(
                stellar_params,
                self.followup_rng,
                max_pairs=1,  # Get just one stellar type question
                include_answers=True,
            )
            if stellar_type_followups:
                followups.append(stellar_type_followups[0])
        # SECOND followup: Get description from JSON file (if available and max_followup_turns >= 2)
        if self.max_followup_turns >= 2 and self.followup_data is not None and obsid is not None:
            # Note: sample_idx here is actually used as obsid in the updated logic
            qa_pair = self._get_followup_from_description(obsid)
            if qa_pair is not None:
                question, answer = qa_pair
                followups.append({'question': question, 'answer': answer})

        # Tokenize and append both followup questions
        text_pairs = []
        for spec in followups:
            question_text = f"\nFollow-up question: {spec['question']}\nAnswer:"
            q_tokens, _ = self._tokenize_text_no_pad(question_text, bos=False)
            answer_text = (spec.get('answer') or "").strip()
            if not answer_text:
                answer_text = "It would remain broadly consistent apart from the requested adjustment."
            a_tokens, _ = self._tokenize_text_no_pad(answer_text, bos=False)
            self._extend_with_tokens(full_tokens, target_tokens, q_tokens, mask_targets=True)
            self._extend_with_tokens(full_tokens, target_tokens, a_tokens, mask_targets=False)
            text_pairs.append((spec['question'], answer_text))
            if len(full_tokens) >= self.max_length:
                break

        return text_pairs
    
    def _load_data(self):
        """Load data from JSON file"""
        print(f"Loading data from {self.json_file}...")
        
        with open(self.json_file, 'r') as f:
            self.raw_data = json.load(f)
            
        print(f"Loaded {len(self.raw_data)} samples from JSON")
        
        # Polyfill description from qa_pairs if missing
        for sample in self.raw_data:
            if not sample.get('description') and sample.get('qa_pairs'):
                try:
                    # Pick the first QA pair - or prefer reasoning/classification?
                    # Let's just pick the first one for now to ensure we have data
                    pair = sample['qa_pairs'][0]
                    # Create the JSON structure expected by parse_description_text
                    desc_obj = {
                        "Question": pair.get('question', ''),
                        "Description": pair.get('answer', '')
                    }
                    sample['description'] = json.dumps(desc_obj)
                except (IndexError, AttributeError, TypeError):
                    pass

        # Filter samples with valid descriptions if requested
        if self.filter_valid_descriptions:
            valid_samples = []
            for sample in self.raw_data:
                desc = sample.get('description')
                if desc and desc.strip() and desc.lower() not in ['none', 'null', '']:
                    valid_samples.append(sample)
            
            print(f"Filtered to {len(valid_samples)} samples with valid descriptions")
            self.raw_data = valid_samples
        
        # Extract dataframe indices
        self.df_indices = []
        for sample in self.raw_data:
            df_idx = sample.get('index')
            if df_idx is not None:
                self.df_indices.append(df_idx)
            else:
                print(f"Warning: Sample missing dataframe index: {sample.get('obsid', 'unknown')}")
                
            
    def parse_description_text(self, description_text: str) -> Dict[str, str]:
        """
        Parse the description text to extract question and answer.
        
        Args:
            description_text (str): The description field containing JSON string
            
        Returns:
            dict: Dictionary with 'question' and 'answer' keys
        """
        if not description_text or not description_text.strip():
            return {'question': '', 'answer': ''}
        
        # Clean up common escape issues before parsing
        cleaned_text = description_text
        
        # Fix common problematic escapes
        escape_fixes = [
            (r'\[', '['),      # \[ -> [
            (r'\]', ']'),      # \] -> ]
            (r'\(', '('),      # \( -> (
            (r'\)', ')'),      # \) -> )
            (r'\ ', ' '),      # \ -> space
            (r'\/', '/'),      # \/ -> /
        ]
        
        for old, new in escape_fixes:
            cleaned_text = cleaned_text.replace(old, new)
        
        try:
            # First try to parse the cleaned string as JSON
            desc_json = json.loads(cleaned_text)
            
            # Extract question and description/answer
            question = desc_json.get('Question', '').strip()
            answer = desc_json.get('Description', '').strip()
            
            return {
                'question': question,
                'answer': answer
            }
            
        except json.JSONDecodeError as e:
            # If that fails, try to find JSON object boundaries
            try:
                # Look for the first '{' and try to find the matching '}'
                start_idx = cleaned_text.find('{')
                if start_idx == -1:
                    raise ValueError("No JSON object found")
                
                # Find the matching closing brace by counting braces
                brace_count = 0
                end_idx = -1
                
                for i, char in enumerate(cleaned_text[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx == -1:
                    raise ValueError("No matching closing brace found")
                
                # Extract just the JSON part
                json_part = cleaned_text[start_idx:end_idx]
                desc_json = json.loads(json_part)
                
                # Extract question and description/answer
                question = desc_json.get('Question', '').strip()
                answer = desc_json.get('Description', '').strip()
                
                return {
                    'question': question,
                    'answer': answer
                }
                
            except (json.JSONDecodeError, ValueError) as e2:
                # Final fallback: try to extract using regex patterns
                try:
                    
                    # Extract Question and Description using regex
                    question_match = re.search(r'"Question"\s*:\s*"([^"]*)"', cleaned_text)
                    desc_match = re.search(r'"Description"\s*:\s*"([^"]*)"', cleaned_text, re.DOTALL)
                    
                    question = question_match.group(1) if question_match else ''
                    answer = desc_match.group(1) if desc_match else ''
                    
                    if question or answer:
                        return {
                            'question': question.strip(),
                            'answer': answer.strip()
                        }
                    else:
                        raise ValueError("No patterns matched")
                        
                except Exception as e3:
                    print(f"Error parsing description JSON (attempt 3): {e3}")
                    print(f"Attempt 2 error: {e2}")
                    print(f"Original error: {e}")
                    print(f"Problematic text (first 200 chars): {cleaned_text[:200]}")
                    
                    # Return the original text as answer if all parsing fails
                    return {
                        'question': '',
                        'answer': description_text.strip()
                    }
    
    def _create_splits(self, train_ratio: float, val_ratio: float, test_ratio: float, cache_dir: Optional[str] = None):
        """Create train/val/test splits with caching for consistency"""

        n_samples = len(self.raw_data)
        indices = np.arange(n_samples)
        
        # Create cache key based on file and parameters
        cache_key = f"{Path(self.json_file).stem}_{n_samples}_{train_ratio}_{val_ratio}_{test_ratio}_{self.random_state}"
        cache_file = None
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"splits_{cache_key}.npz")
        
        # Try to load cached splits
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached splits from {cache_file}")
            cached = np.load(cache_file)
            train_indices = cached['train_indices']
            val_indices = cached['val_indices'] 
            test_indices = cached['test_indices']
        else:
            print("Creating new train/val/test splits...")
            
            # First split: separate test set (Fixed first 1000 samples as per user request)
            # This avoids the test_size=0.0 error and provides the requested deterministic split
            test_indices = indices[:1000]
            temp_indices = indices[1000:]
            
            # Second split: separate train and val from remaining data
            adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
            train_indices, val_indices = train_test_split(
                temp_indices,
                test_size=adjusted_val_ratio,
                random_state=self.random_state,
                shuffle=True
            )
            
            # Cache splits if directory provided
            if cache_file:
                print(f"Caching splits to {cache_file}")
                np.savez(cache_file, 
                        train_indices=train_indices,
                        val_indices=val_indices, 
                        test_indices=test_indices)
        
        # Select indices for current split
        if self.split == 'train':
            self.split_indices = train_indices
        elif self.split == 'val':
            self.split_indices = val_indices
        else:  # test
            self.split_indices = test_indices

        print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        print(f"Current split ({self.split}): {len(self.split_indices)} samples")

    def _initialize_feature_normalizer(self, provided_stats: Optional[Dict[str, np.ndarray]]) -> None:
        if self.feature_normalizer is None or self.features_array is None:
            return
        indices = self._collect_feature_indices()
        self.feature_normalizer.initialize(self.features_array, indices, provided_stats)

    def _collect_feature_indices(self) -> List[int]:
        indices: List[int] = []
        if self.features_array is None:
            return indices
            
        for raw_idx in self.split_indices:
            sample = self.raw_data[raw_idx]
            
            # Logic to find feature index
            df_idx = None
            if self.index_df is not None:
                 obsid = sample.get('obsid')
                 if obsid is not None:
                     df_idx = self.obsid_to_feature_idx.get(obsid)
                     if df_idx is None:
                         # Try int/str conversion just in case key format differs
                         try:
                             df_idx = self.obsid_to_feature_idx.get(int(obsid))
                         except (ValueError, TypeError):
                             pass
                         if df_idx is None:
                             df_idx = self.obsid_to_feature_idx.get(str(obsid))

            # Fallback to internal index if index_df not used or lookup failed
            if df_idx is None:
                df_idx = sample.get('index')

            if df_idx is not None and 0 <= df_idx < len(self.features_array):
                indices.append(df_idx)
        return indices

    def get_feature_normalization_stats(self, copy: bool = True) -> Optional[Dict[str, np.ndarray]]:
        if self.feature_normalizer is None:
            return None
        return self.feature_normalizer.get_stats(copy=copy)

    def denormalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.feature_normalizer is None:
            return features
        return self.feature_normalizer.inverse(features)

    def _apply_feature_normalization(self, features: np.ndarray) -> np.ndarray:
        if self.feature_normalizer is None:
            # print("not normalizing features")
            return np.asarray(features, dtype=np.float32)
        # print("normalizing features")
        return self.feature_normalizer.transform(features)
        
    def read_lamost_spectra(self, filename):
        try:
            with fits.open(filename) as hdulist:
                binaryext = hdulist[1].data
                header = hdulist[0].header
            spectra = torch.tensor(binaryext['FLUX'].astype(np.float32))
            wv = binaryext['WAVELENGTH'].astype(np.float32)
            rv = header['HELIO_RV']
            meta = {'RV': rv, 'wavelength': wv}
        except FileNotFoundError:
            print(f"File not found: {filename}")
            spectra = torch.zeros(1, 4096, dtype=torch.float32)
            wv = np.linspace(3690, 9100, 4096, dtype=np.float32)
            meta = {'RV': 0.0, 'wavelength': wv}
        if self.transforms:
            spectra, _, meta = self.transforms(spectra, None, meta)
        spectra_masked, mask, _ = self.mask_transform(spectra, None, meta)
        pad = torch.zeros(1, 4096 - spectra.shape[-1])
        spectra = torch.cat([spectra, pad], dim=-1)
        spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
        return spectra, spectra_masked, meta
    
    def read_apogee_spectra(self, filename):
        try:
            with fits.open(filename) as hdul:
                spectra = torch.tensor(hdul[1].data.astype(np.float32).squeeze()[None])
            meta = {}
            header = hdul[1].header
            # Create pixel array (1-indexed for FITS convention)
            pixels = np.arange(1, spectra.shape[-1] + 1)
            
            # Calculate log10(wavelength):
            # log_wave = CRVAL1 + CDELT1 * (pixel - CRPIX1)
            log_wavelength = header['CRVAL1'] + header['CDELT1'] * (pixels - header['CRPIX1'])
            
            # Convert to linear wavelength in Angstroms
            wv = 10**log_wavelength
            meta = {'wavelength': wv}
        except FileNotFoundError:
            print(f"File not found: {filename}")
            spectra = torch.zeros(1, 8576, dtype=torch.float32)
            wv = np.linspace(15100, 17000, 8576, dtype=np.float32)
            meta = {'wavelength': wv}
        if self.transforms:
            spectra, _, meta = self.transforms(spectra, None, meta)
        spectra_masked, mask, _ = self.mask_transform(spectra, None, meta)
        pad = torch.zeros(1, 8576 - spectra.shape[-1])
        spectra = torch.cat([spectra, pad], dim=-1)
        spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
        return spectra, spectra_masked, meta
    
    def get_raw_spectra(self, obsid: int, id_type='obsid') -> Optional[np.ndarray]:
        if id_type == 'obsid':
                obsdir = str(obsid)[:4]
                spectra_filename = os.path.join(f'/home/ilay.kamai/work/lamost/data', f'{obsdir}/{obsid}.fits')
                spectra, spectra_masked, meta = self.read_lamost_spectra(spectra_filename)
                meta['obsid'] = obsid
        elif id_type == 'APOGEE_ID':
            spectra_filename = f"/home/ilay.kamai/work/apogee/data/aspcapStar-dr17-{obsid}.fits"
            spectra, spectra_masked, meta = self.read_apogee_spectra(spectra_filename)
            meta['apogee_id'] = obsid
        else:
            raise ValueError(f"Unknown obsid format: {id_type}")
        return spectra, spectra_masked, meta
    
    def __len__(self):
        return len(self.split_indices)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with space reserved for features"""
        sample_idx = self.split_indices[idx]
        sample = self.raw_data[sample_idx]
        
        # Parse description
        description_text = sample.get('description', '')
        parsed_desc = self.parse_description_text(description_text)
        
        # Get raw tokens (no padding)
        question_tokens, num_tok_q = self._tokenize_text_no_pad(parsed_desc['question'], bos=True)
        answer_tokens, num_tok_a = self._tokenize_text_no_pad(parsed_desc['answer'], bos=False)
        
        # Calculate available space after reserving feature space
        available_length = self.max_length - self.num_spectral_features
        
        # Combine sequences BEFORE padding (without feature space)
        combined_tokens = question_tokens + answer_tokens
        total_tokens = len(combined_tokens)
        
        # Handle truncation if needed
        if total_tokens > available_length:
            combined_tokens = combined_tokens[:available_length]
            # Adjust counts if truncated
            if num_tok_q > available_length:
                num_tok_q = available_length
                num_tok_a = 0
            elif num_tok_q + num_tok_a > available_length:
                num_tok_a = available_length - num_tok_q
            total_tokens = available_length
        
        # Create the full sequence with feature space AT THE BEGINNING
        # Structure: [FEATURE_SPACE] + [question_tokens] + [answer_tokens] + follow-ups + [PADDING]
        feature_start_idx = 0
        full_sequence: List[int] = [-100] * self.num_spectral_features
        target_sequence: List[int] = [-100] * self.num_spectral_features

        # Add question tokens after features
        question_start_idx = len(full_sequence)
        base_question_tokens = question_tokens[:num_tok_q]
        full_sequence.extend(base_question_tokens)
        target_sequence.extend([-100] * len(base_question_tokens))

        # Add answer tokens
        answer_start_idx = len(full_sequence)
        base_answer_tokens = answer_tokens[:num_tok_a]
        full_sequence.extend(base_answer_tokens)
        target_sequence.extend(base_answer_tokens)
        base_answer_length = len(base_answer_tokens)

        # Optional follow-up turns conditioned on stellar parameters
        stellar_data = sample.get('stellar_data', {})
        obsid = sample.get('obsid', None)
        try:
             obsid_int = int(obsid) if obsid is not None else None
        except:
             obsid_int = None
        stellar_params = self._extract_physical_params(stellar_data, obsid=obsid_int)
        followup_text_pairs = []  # Track followup Q&A pairs as text
        if self.enable_followup:
            followup_text_pairs = self._append_followup_turns(full_sequence,
             target_sequence, stellar_params, sample_idx, obsid=obsid_int)

        # Pad remaining space with -100 placeholders
        remaining_space = self.max_length - len(full_sequence)
        if remaining_space > 0:
            full_sequence.extend([-100] * remaining_space)
            target_sequence.extend([-100] * remaining_space)
        elif remaining_space < 0:
            full_sequence = full_sequence[:self.max_length]
            target_sequence = target_sequence[:self.max_length]

        # Convert to tensor
        input_ids = torch.tensor(full_sequence, dtype=torch.long)
        target_ids = torch.tensor(target_sequence, dtype=torch.long)
        
        # Get other data
        # Feature lookup logic
        df_index = None
        if self.index_df is not None:
             # Use the provided index_df mapping
             # obsid extracted earlier around line 800
             if obsid is not None:
                 df_index = self.obsid_to_feature_idx.get(obsid)
                 if df_index is None:
                      # Try alternate types
                      try:
                          df_index = self.obsid_to_feature_idx.get(int(obsid))
                      except (ValueError, TypeError):
                          pass
                      if df_index is None:
                          df_index = self.obsid_to_feature_idx.get(str(obsid))
        
        # Fallback to legacy 'index' field if no index_df or lookup failed
        if df_index is None:
            df_index = sample.get('index')
        
        if self.features_array is not None and df_index is not None:
            norm_features = self._apply_feature_normalization(self.features_array[df_index])
            features = torch.from_numpy(norm_features)
            masked_spectra = features
            spectra = features
        else:
            spectra, masked_spectra, _ = self.get_raw_spectra(sample['obsid'])
            features = masked_spectra
        
        stellar_data = sample.get('stellar_data', {})
        obsid = sample.get('obsid', None)
        
        numeric_tensor = self._extract_numeric_tensor(stellar_data)

        return {
            'input_ids': input_ids,                    # [-100,-100,Q1,Q2,A1,A2,-100,-100,...]
            'target_ids': target_ids,                  # [-100,-100,-100,-100,A1,A2,-100,-100,...]
            'input_length': num_tok_q,                 # Length of question portion
            'feature_start_idx': feature_start_idx,    # Where features are inserted (0)
            'feature_length': self.num_spectral_features,        # Number of feature tokens
            'question_start_idx': question_start_idx,  # Where question begins
            'answer_start_idx': answer_start_idx,      # Where answer begins
            'target_length': base_answer_length,                # Length of base answer portion
            'input_text': parsed_desc['question'],
            'target_text': parsed_desc['answer'],
            'followup_turns': followup_text_pairs,     # List of (question, answer) tuples for followup turns
            'features': features,
            'spectra': spectra,
            'masked_spectra': masked_spectra,
            'stellar_data': stellar_data,
            'obsid': obsid,
            'df_index': df_index,
            'sample_index': sample_idx,
            'y_numeric': numeric_tensor,
        }
    
    def _extract_numeric_tensor(self, stellar_data: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Extract normalized Teff/logg/FeH vector if available."""
        if not isinstance(stellar_data, dict):
            return None

        values = []
        for param in ('Teff', 'logg', 'FeH'):
            raw_val = None
            for key in self.numeric_key_alternatives.get(param, [param]):
                if key in stellar_data and stellar_data[key] is not None:
                    raw_val = stellar_data[key]
                    break
            if raw_val is None:
                return None
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                return None
            if math.isnan(val):
                return None
            low, high = self.numeric_bounds[param]
            norm = (val - low) / (high - low)
            norm = max(0.0, min(1.0, norm))
            values.append(norm)

        if len(values) != 3:
            return None
        return torch.tensor(values, dtype=torch.float32)

    def get_split_info(self) -> Dict[str, int]:
        """Get information about all splits"""
        # This requires recreating splits temporarily
        indices = np.arange(len(self.raw_data))
        temp_indices, test_indices = train_test_split(
            indices, test_size=0.15, random_state=self.random_state, shuffle=True
        )
        train_indices, val_indices = train_test_split(
            temp_indices, test_size=0.15/(0.7+0.15), random_state=self.random_state, shuffle=True
        )
        
        return {
            'total': len(self.raw_data),
            'train': len(train_indices),
            'val': len(val_indices),
            'test': len(test_indices)
        }


def create_stellar_dataloaders(json_file: str,
                             features_array: Optional[np.ndarray] = None,
                             batch_size: int = 32,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             random_state: int = 42,
                             num_workers: int = 0,
                             cache_dir: Optional[str] = None,
                             world_size: int = 1,
                             device: Optional[str] = None,
                             dataset_cls: Type["StellarQuestionsDataset"] = StellarQuestionsDataset,
                             multimodal_df: Optional[pd.DataFrame] = None,
                             index_df: Optional[pd.DataFrame] = None,
                             **dataset_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, test dataloaders
    """

    dataset_cls = dataset_cls or StellarQuestionsDataset
    shared_feature_stats = dataset_kwargs.pop('feature_stats', None)

    # Create datasets for each split
    train_dataset = dataset_cls(
        json_file=json_file,
        features_array=features_array,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        feature_stats=shared_feature_stats,
        multimodal_df=multimodal_df,
        index_df=index_df,
        **dataset_kwargs
    )

    feature_stats = train_dataset.get_feature_normalization_stats(copy=True)
    if feature_stats is None:
        feature_stats = shared_feature_stats

    val_dataset = dataset_cls(
        json_file=json_file,
        features_array=features_array,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        feature_stats=feature_stats,
        multimodal_df=multimodal_df,
        index_df=index_df,
        **dataset_kwargs
    )

    test_dataset = dataset_cls(
        json_file=json_file,
        features_array=features_array,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        feature_stats=feature_stats,
        multimodal_df=multimodal_df,
        index_df=index_df,
        **dataset_kwargs
    )

    # Common loader kwargs
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    # Handle distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            shuffle=True, 
            seed=random_state,
            drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=world_size, 
            shuffle=False, 
            seed=random_state,
            drop_last=False
        )
        test_sampler = DistributedSampler(
            test_dataset, 
            num_replicas=world_size, 
            shuffle=False, 
            seed=random_state,
            drop_last=False
        )

        train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle tokenized descriptions and optional features
    """
    # Stack the sequences
    input_ids = torch.stack([item['input_ids'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    
    # Feature insertion positions
    feature_start_indices = torch.tensor([item['feature_start_idx'] for item in batch], dtype=torch.long)
    feature_lengths = torch.tensor([item['feature_length'] for item in batch], dtype=torch.long)
    answer_start_indices = torch.tensor([item['answer_start_idx'] for item in batch], dtype=torch.long)
    
    # Other info
    input_lengths = torch.tensor([item['input_length'] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)

    question_start_indices = torch.tensor([item['question_start_idx'] for item in batch], dtype=torch.long)
    

    input_texts = [item['input_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    followup_turns = [item.get('followup_turns', []) for item in batch]  # Collect followup turns
    obsids = [item['obsid'] for item in batch]
    df_indices = [item['df_index'] for item in batch]
    stellar_data = [item['stellar_data'] for item in batch]
    spectra = [item['spectra'] for item in batch]
    masked_spectra = [item['masked_spectra'] for item in batch]
    y_numeric_list = [item.get('y_numeric') for item in batch]

    # Handle features - check if any sample has features
    features_list = [item['features'] for item in batch]
    if any(f is not None for f in features_list):
        # Stack features, replacing None with zeros
        feature_dim = None
        for f in features_list:
            if f is not None:
                feature_dim = f.shape[0] if len(f.shape) == 1 else f.shape
                break
        
        if feature_dim is not None:
            processed_features = []
            for f in features_list:
                if f is not None:
                    processed_features.append(f)
                else:
                    processed_features.append(torch.zeros(feature_dim, dtype=torch.float32))
            features_tensor = torch.stack(processed_features)
        else:
            features_tensor = None
    else:
        features_tensor = None

    y_numeric, y_numeric_present = _stack_numeric(y_numeric_list)
    
    return {
        'input_ids':  input_ids,                    # [batch, seq_len]
        'target_ids': target_ids,                    # [batch, seq_len]
        'feature_start_indices': feature_start_indices,  # Where to insert features per sample
        'question_start_indices': question_start_indices,  # Where questions start per sample
        'feature_lengths': feature_lengths,          # Number of features per sample  
        'answer_start_indices': answer_start_indices, # Where answers start per sample
        'input_lengths': input_lengths,              # Question lengths
        'target_lengths': target_lengths,            # Answer lengths
        'input_texts': input_texts,
        'target_texts': target_texts,
        'followup_turns': followup_turns,            # List of followup Q&A pairs per sample
        'features': features_tensor,
        'spectra': torch.stack(spectra),
        'masked_spectra': torch.stack(masked_spectra),
        'obsids': obsids,
        'df_indices': df_indices,
        'stellar_data': stellar_data,
        'y_numeric': y_numeric,
        'y_numeric_present': y_numeric_present
    }


def _stack_numeric(values: List[Optional[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack optional numeric tensors with a presence mask."""
    if not values:
        return torch.zeros((0, 3), dtype=torch.float32), torch.zeros(0, dtype=torch.bool)

    valid = [v for v in values if v is not None]
    if not valid:
        zeros = torch.zeros((len(values), 3), dtype=torch.float32)
        mask = torch.zeros(len(values), dtype=torch.bool)
        return zeros, mask

    shape = valid[0].shape
    batch = torch.zeros((len(values),) + shape, dtype=torch.float32)
    mask = torch.zeros(len(values), dtype=torch.bool)
    for idx, tensor in enumerate(values):
        if tensor is None:
            continue
        batch[idx] = tensor.to(dtype=torch.float32)
        mask[idx] = True
    return batch, mask


# Example usage and testing
if __name__ == "__main__":

    TOKENIZER_PATH = "/home/ilay.kamai/work/.llama/Llama3.2-1B/tokenizer.model"
    tokenizer = Tokenizer(model_path=TOKENIZER_PATH)

    json_path = "/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json"
    spectral_features = np.load('/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/features.npy')
    # Example usage
    print("Example usage:")
    # Case 3: Create all dataloaders at once
    print("\n3. Create train/val/test dataloaders:")
    train_loader, val_loader, test_loader = create_stellar_dataloaders(
        json_file=json_path,
        features_array=spectral_features,  # Optional
        batch_size=32,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        tokenizer_path=TOKENIZER_PATH,
        num_spectral_features=8,
        cache_dir='cache/'  # Cache splits for consistency
    )
    
    for i, data in enumerate(train_loader):
        print(f"Batch input tokens shape: {data['input_ids'].shape}")
        print(f"Batch target tokens shape: {data['target_ids'].shape}")
        print("f start indices:", data['feature_start_indices'], " feature lengths:", data['feature_lengths'])
        print("Answer start indices:", data['answer_start_indices'], " target lengths:", data['target_lengths'])
        print(data['input_ids'][0][:100])
        first_tokens = data['input_ids'][0][:100].tolist()
        first_targets = data['target_ids'][0][:100].tolist()
        printable_tokens = [tok for tok in first_tokens if tok >= 0]
        printable_targets = [tok for tok in first_targets if tok >= 0]
        print("decoded: ", tokenizer.decode(printable_tokens) if printable_tokens else "")
        print("decoded target: ", tokenizer.decode(printable_targets) if printable_targets else "")

        print(data['input_texts'][0][:100])
        print(data['target_ids'][0][:100])
        print(data['target_texts'][0][:100])
        print("tot lengths: ", data['input_lengths'] + data['target_lengths'])

        if i == 10:
            break
