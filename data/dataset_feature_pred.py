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
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple, Dict, Any, List, Type
import os
from pathlib import Path
import re


import os
os.system('pip install tiktoken fairscale fire blobfile')
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)

from llama3.llama.tokenizer import Tokenizer
from data.feature_normalizer import FeatureNormalizer
from src.follow_up_templates import create_follow_up_specs


class StellarFeaturePredictionDataset(Dataset):
    """
    PyTorch Dataset for cross-star feature prediction from stellar parameters.

    This dataset implements a cross-star prediction task: given the features and parameters
    of one star (star 1), along with the parameters of a different star (star 2), the model
    predicts star 2's latent features and physical description.

    Question format:
    "this is a star <star_data> with Teff {Teff1} K, logg {logg1}, and FeH {FeH1}.
     describe a star with Teff {Teff2} K, logg {logg2}, and FeH {FeH2}"

    Where <star_data> is replaced with star 1's features during training (placeholder of -100 tokens).

    Answer format: "{physical_description2}"

    Note: The target features (star 2) are not included in the text answer to keep sequences short.
    They are available separately in the returned dictionary under the 'features_star2' key.

    Args:
        json_file (str): Path to the JSON file with stellar data
        features_array (np.ndarray): Array of spectral features (required for this dataset)
        split (str): One of 'train', 'val', 'test'
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set (remaining after train/val)
        random_state (int): Random seed for reproducible splits
        filter_valid_params (bool): Whether to filter samples with invalid parameters
        cache_dir (Optional[str]): Directory to cache split indices for consistency
        tokenizer_path (Optional[str]): Path to SentencePiece tokenizer model
        max_length (int): Maximum sequence length for tokenization
        feature_precision (int): Number of decimal places for feature values (default: 4)
        feature_separator (str): Separator between feature values (default: ' ')
        num_spectral_features (int): Number of tokens to reserve for spectral features (default: 4)
        condition_on_star1 (bool): If True, condition on star1 parameters (default mode).
                                   If False, only ask to describe star2 without star1 context.
        random_pairing (bool): If True, pair stars randomly. If False, pair with most similar stars (default: False).
        enable_followup (bool): Whether to append follow-up questions
        followup_prob (float): Probability of adding follow-up questions to a sample
        max_followup_turns (int): Maximum number of follow-up turns to add
        followup_seed (int): Random seed for follow-up question generation
        followup_json_file (Optional[str]): Path to second JSON for description-based follow-ups
        followup_mode (str): "stellar_type", "description", or "mixed" (50/50 default)
    """

    def __init__(self,
                 json_file: str,
                 features_array: np.ndarray,
                 split: str = 'train',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42,
                 filter_valid_params: bool = True,
                 cache_dir: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 tokenizer: Optional[Any] = None,
                 max_length: int = 1024,
                 feature_precision: int = 4,
                 feature_separator: str = ' ',
                 num_spectral_features: int = 4,
                 normalize_features: bool = True,
                 feature_stats: Optional[Dict[str, np.ndarray]] = None,
                 feature_norm_epsilon: float = 1e-6,
                 condition_on_star1: bool = True,
                 random_pairing: bool = True,
                 enable_followup: bool = False,
                 followup_prob: float = 0.0,
                 max_followup_turns: int = 1,
                 followup_seed: int = 42,
                 # Optional second JSON for follow-up Q&A
                 followup_json_file: Optional[str] = None,
                 followup_mode: str = "mixed"):

        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        assert features_array is not None, "features_array is required for feature prediction dataset"

        self.json_file = json_file
        self.features_array = features_array
        self.split = split
        self.random_state = random_state
        self.filter_valid_params = filter_valid_params
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.feature_precision = feature_precision
        self.feature_separator = feature_separator
        self.num_spectral_features = num_spectral_features
        self.normalize_features = normalize_features
        self.feature_norm_epsilon = feature_norm_epsilon
        self.condition_on_star1 = condition_on_star1
        self.random_pairing = random_pairing
        self.feature_normalizer = FeatureNormalizer(
            enabled=self.normalize_features and (self.features_array is not None),
            epsilon=self.feature_norm_epsilon,
        )
        self.enable_followup = enable_followup
        self.followup_prob = followup_prob
        self.max_followup_turns = max_followup_turns
        self.followup_json_file = followup_json_file
        self.followup_mode = followup_mode
        seed_offset = followup_seed + hash((split, random_state))
        self._followup_rng = random.Random(seed_offset)

        # Load follow-up JSON if provided
        self.followup_data = None
        if self.followup_json_file and self.enable_followup:
            self._load_followup_json()

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
        
        # Pre-computed normalized parameters and nearest neighbor index
        self._normalized_params_cache = None
        self._nn_index = None
        self._feature_cache = {}  # Cache normalized features to avoid repeated computation
        self._precomputed_pairs = None  # Pre-computed star pairs for O(1) access
        self._shuffled_indices = None  # Shuffled indices for random pairing
        self.numeric_key_alternatives = {
            'Teff': ['Teff', 'teff_k', 'teff', 'effective_temperature'],
            'logg': ['logg', 'log_g'],
            'FeH': ['FeH', 'feh', '[Fe/H]', 'metallicity'],
        }

        # Load tokenizer if available
        if self.tokenizer is None:
            self._load_tokenizer()

        # Load and process data
        self._load_data()
        self._create_splits(train_ratio, val_ratio, test_ratio, cache_dir)
        self._initialize_feature_normalizer(feature_stats)
        if self.random_pairing:
            self._initialize_shuffled_indices()
        else:
            self._precompute_star_pairs(cache_dir)

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

    def _load_followup_json(self):
        """Load the follow-up JSON file for Q&A pairs."""
        try:
            with open(self.followup_json_file, 'r') as f:
                self.followup_data = json.load(f)
            print(f"Loaded {len(self.followup_data)} follow-up samples from {self.followup_json_file}")
        except Exception as e:
            print(f"Warning: Could not load follow-up JSON {self.followup_json_file}: {e}")
            self.followup_data = None

    def _get_followup_from_description(self, sample_idx: int) -> Optional[Tuple[str, str]]:
        """Extract question and answer from the follow-up JSON file."""
        if self.followup_data is None or sample_idx >= len(self.followup_data):
            return None

        followup_sample = self.followup_data[sample_idx]
        description = followup_sample.get("description", "")

        if not description:
            return None

        # Parse the description to get question and answer
        parsed = self.parse_description_text(description)
        question = parsed.get("question", "")
        answer = parsed.get("answer", "")

        if not question or not answer:
            return None

        return question, answer

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

    def _append_followup_turns(self,
                               full_tokens: List[int],
                               target_tokens: List[int],
                               params: Dict[str, Optional[float]],
                               sample_idx: int = -1) -> None:
        if not self.enable_followup or self.followup_prob <= 0.0:
            return
        if self._followup_rng.random() > self.followup_prob:
            return

        # Decide which mode to use based on followup_mode
        use_description = False
        if self.followup_mode == "description" and self.followup_data is not None:
            use_description = True
        elif self.followup_mode == "mixed" and self.followup_data is not None:
            # 50% probability for description, 50% for stellar type
            use_description = self._followup_rng.random() < 0.5

        followups = []

        # Try to use description from follow-up JSON file
        if use_description and sample_idx >= 0:
            qa_pair = self._get_followup_from_description(sample_idx)
            if qa_pair is not None:
                question, answer = qa_pair
                # Format as a followup spec for consistent processing
                followups.append({'question': question, 'answer': answer})

        # Generate stellar type questions from templates (if not using description or as fallback)
        if not followups:
            followups = create_follow_up_specs(
                params,
                self._followup_rng,
                max_pairs=self.max_followup_turns,
                include_answers=True,
            )

        for spec in followups:
            question_text = f"\nFollow-up question: {spec['question']}\nAnswer:"
            q_tokens, _ = self._tokenize_text_no_pad(question_text, bos=False)
            answer_text = (spec.get('answer') or "").strip()
            if not answer_text:
                answer_text = "It would retain broadly similar observable signs with subtle shifts."
            a_tokens, _ = self._tokenize_text_no_pad(answer_text, bos=False)
            self._extend_with_tokens(full_tokens, target_tokens, q_tokens, mask_targets=True)
            self._extend_with_tokens(full_tokens, target_tokens, a_tokens, mask_targets=False)
            if len(full_tokens) >= self.max_length:
                break

    def _load_data(self):
        """Load data from JSON file and filter for valid stellar parameters"""
        print(f"Loading data from {self.json_file}...")

        with open(self.json_file, 'r') as f:
            self.raw_data = json.load(f)

        print(f"Loaded {len(self.raw_data)} samples from JSON")

        # Filter samples with valid stellar parameters and features
        if self.filter_valid_params:
            valid_samples = []
            for sample in self.raw_data:
                stellar_data = sample.get('stellar_data', {})
                df_index = sample.get('index')

                # Check if all required parameters exist
                has_params = True
                for param in ['Teff', 'logg', 'FeH']:
                    raw_val = None
                    for key in self.numeric_key_alternatives.get(param, [param]):
                        if key in stellar_data and stellar_data[key] is not None:
                            raw_val = stellar_data[key]
                            break
                    if raw_val is None:
                        has_params = False
                        break
                    try:
                        val = float(raw_val)
                        if math.isnan(val):
                            has_params = False
                            break
                    except (TypeError, ValueError):
                        has_params = False
                        break

                # Check if features are available
                has_features = (df_index is not None and
                               df_index < len(self.features_array))

                if has_params and has_features:
                    valid_samples.append(sample)

            print(f"Filtered to {len(valid_samples)} samples with valid parameters and features")
            self.raw_data = valid_samples

        # Extract dataframe indices
        self.df_indices = []
        for sample in self.raw_data:
            df_idx = sample.get('index')
            if df_idx is not None:
                self.df_indices.append(df_idx)
            else:
                print(f"Warning: Sample missing dataframe index: {sample.get('obsid', 'unknown')}")

    def _extract_stellar_params(self, stellar_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract Teff, logg, FeH from stellar_data"""
        params = {}
        for param in ['Teff', 'logg', 'FeH']:
            raw_val = None
            for key in self.numeric_key_alternatives.get(param, [param]):
                if key in stellar_data and stellar_data[key] is not None:
                    raw_val = stellar_data[key]
                    break
            if raw_val is not None:
                try:
                    params[param] = float(raw_val)
                except (TypeError, ValueError):
                    params[param] = None
            else:
                params[param] = None
        return params

    def _format_features_as_text(self, features: np.ndarray) -> str:
        """Convert feature array to formatted text string"""
        # Flatten if needed
        if features.ndim > 1:
            features = features.flatten()

        # Format each value with specified precision
        formatted_values = [f"{val:.{self.feature_precision}f}" for val in features]

        # Join with separator
        return self.feature_separator.join(formatted_values)

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
        cache_key = f"featurepred_{Path(self.json_file).stem}_{n_samples}_{train_ratio}_{val_ratio}_{test_ratio}_{self.random_state}"
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

            # First split: separate test set
            temp_indices, test_indices = train_test_split(
                indices,
                test_size=test_ratio,
                random_state=self.random_state,
                shuffle=True
            )

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
        feature_indices = self._collect_feature_indices()
        self.feature_normalizer.initialize(self.features_array, feature_indices, provided_stats)

    def _collect_feature_indices(self) -> List[int]:
        indices: List[int] = []
        if self.features_array is None:
            return indices
        for raw_idx in self.split_indices:
            sample = self.raw_data[raw_idx]
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

    def _apply_feature_normalization(self, features: np.ndarray, cache_key: int = None) -> np.ndarray:
        """Apply feature normalization with optional caching"""
        if self.feature_normalizer is None:
            return np.asarray(features, dtype=np.float32)
        
        # Use cache if key provided
        if cache_key is not None and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        result = self.feature_normalizer.transform(features)
        
        # Cache result if key provided and cache not too large
        if cache_key is not None and len(self._feature_cache) < 10000:
            self._feature_cache[cache_key] = result
            
        return result

    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize stellar parameters to [0, 1] range"""
        normalized = []
        for param in ['Teff', 'logg', 'FeH']:
            val = params[param]
            low, high = self.numeric_bounds[param]
            norm = (val - low) / (high - low)
            norm = max(0.0, min(1.0, norm))
            normalized.append(norm)
        return np.array(normalized)

    def _precompute_star_pairs(self, cache_dir: Optional[str] = None):
        """
        Pre-compute all star pairs for O(1) runtime access.
        Each sample gets paired based on random_pairing flag:
        - If random_pairing=True: pairs with random other samples
        - If random_pairing=False: pairs with most similar stellar parameters
        """
        n_samples = len(self.split_indices)
        pairing_mode = "random" if self.random_pairing else "similarity"
        print(f"Pre-computing {pairing_mode} star pairs for {n_samples} samples...")
        
        # Create cache key for pairs (include pairing mode)
        cache_key = f"pairs_{Path(self.json_file).stem}_{n_samples}_{self.split}_{self.random_state}_{pairing_mode}"
        cache_file = None
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
        
        # Try to load cached pairs
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached star pairs from {cache_file}")
            cached = np.load(cache_file)
            self._precomputed_pairs = {
                'star1_indices': cached['star1_indices'],
                'star2_indices': cached['star2_indices']
            }
            print(f"✓ Loaded {len(self._precomputed_pairs['star1_indices'])} pre-computed pairs")
            return
        
        # Set random seed for reproducible pairings
        np.random.seed(self.random_state)
        
        star1_indices = []
        star2_indices = []
        
        if self.random_pairing:
            # Random pairing mode
            for idx in range(n_samples):
                star1_indices.append(idx)
                
                # Choose random sample (excluding self)
                available = [i for i in range(n_samples) if i != idx]
                if available:
                    star2_indices.append(np.random.choice(available))
                else:
                    # Edge case: only one sample
                    star2_indices.append(idx)
        else:
            # Similarity-based pairing mode
            # Extract and normalize all parameters
            all_params = []
            valid_sample_indices = []
            
            for idx in range(n_samples):
                sample_idx = self.split_indices[idx]
                sample = self.raw_data[sample_idx]
                stellar_data = sample.get('stellar_data', {})
                params = self._extract_stellar_params(stellar_data)
                
                # Check if parameters are valid
                if not any(v is None for v in params.values()):
                    normalized = self._normalize_params(params)
                    all_params.append(normalized)
                    valid_sample_indices.append(idx)
            
            if len(all_params) < 2:
                print("Warning: Not enough valid samples for similarity pairing, falling back to random")
                # Fallback to random pairing
                for idx in range(n_samples):
                    star1_indices.append(idx)
                    available = [i for i in range(n_samples) if i != idx]
                    star2_indices.append(np.random.choice(available) if available else idx)
            else:
                # Build nearest neighbor index for all valid samples
                all_params = np.array(all_params)
                nn_index = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='euclidean')
                nn_index.fit(all_params)
                
                # Pre-compute pairs for all samples
                for idx in range(n_samples):
                    star1_indices.append(idx)
                    
                    # Find closest match (excluding self)
                    if idx in valid_sample_indices:
                        param_idx = valid_sample_indices.index(idx)
                        query_params = all_params[param_idx:param_idx+1]
                        
                        # Get nearest neighbors (first one is self, so take second)
                        distances, indices = nn_index.kneighbors(query_params, n_neighbors=min(6, len(all_params)))
                        
                        # Find first neighbor that's not self
                        closest_idx = None
                        for nn_idx in indices[0]:
                            candidate_idx = valid_sample_indices[nn_idx]
                            if candidate_idx != idx:
                                closest_idx = candidate_idx
                                break
                        
                        if closest_idx is not None:
                            star2_indices.append(closest_idx)
                        else:
                            # Fallback to random sample
                            available = [i for i in range(n_samples) if i != idx]
                            star2_indices.append(np.random.choice(available) if available else idx)
                    else:
                        # For invalid samples, pair with random valid sample
                        if valid_sample_indices:
                            star2_indices.append(np.random.choice(valid_sample_indices))
                        else:
                            star2_indices.append((idx + 1) % n_samples)
        
        # Store pairs
        self._precomputed_pairs = {
            'star1_indices': np.array(star1_indices),
            'star2_indices': np.array(star2_indices)
        }
        
        # Cache pairs if directory provided
        if cache_file:
            print(f"Caching star pairs to {cache_file}")
            np.savez(cache_file,
                    star1_indices=self._precomputed_pairs['star1_indices'],
                    star2_indices=self._precomputed_pairs['star2_indices'])
        
        print(f"✓ Pre-computed {len(star1_indices)} {pairing_mode} star pairs")

    def _initialize_shuffled_indices(self):
        """Initialize shuffled indices for random pairing"""
        n_samples = len(self.split_indices)
        self._shuffled_indices = np.arange(n_samples)
        np.random.seed(self.random_state)
        np.random.shuffle(self._shuffled_indices)
        print(f"✓ Initialized shuffled indices for {n_samples} samples")
    
    def shuffle_pairs(self, epoch: int = None):
        """
        Shuffle the pairing indices for random pairing mode.
        Should be called at the start of each epoch to avoid overfitting.
        
        Args:
            epoch (int, optional): Epoch number for seeding (ensures reproducibility)
        """
        if not self.random_pairing:
            return  # No-op for similarity-based pairing
        
        if self._shuffled_indices is None:
            self._initialize_shuffled_indices()
            return
        
        # Use epoch-based seeding for reproducibility while maintaining randomness
        if epoch is not None:
            np.random.seed(self.random_state + epoch)
        
        np.random.shuffle(self._shuffled_indices)
        print(f"✓ Shuffled pairing indices for epoch {epoch if epoch is not None else 'unknown'}")

    def _get_star_pair(self, idx: int) -> Tuple[int, int]:
        """
        Get star pair for given index - O(1) operation.
        
        Returns:
            Tuple[int, int]: (star1_idx, star2_idx) in split_indices
        """
        if self.random_pairing:
            # print("features dataset: using random pairs!")
            # Use shuffled indices for random pairing
            if self._shuffled_indices is None:
                self._initialize_shuffled_indices()
            
            star1_idx = idx
            # Use modulo to handle cases where we cycle through the dataset
            star2_idx = self._shuffled_indices[idx % len(self._shuffled_indices)]
            
            # Ensure we don't pair a star with itself
            if star2_idx == star1_idx:
                star2_idx = (star2_idx + 1) % len(self.split_indices)
            
            return star1_idx, star2_idx
        else:
            # print("features dataset: using similary based pairs!")
            # Use pre-computed similarity-based pairs
            if self._precomputed_pairs is None:
                # Fallback to simple pairing
                star2_idx = (idx + 1) % len(self.split_indices)
                return idx, star2_idx
            
            star1_idx = self._precomputed_pairs['star1_indices'][idx]
            star2_idx = self._precomputed_pairs['star2_indices'][idx]
            return star1_idx, star2_idx

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample for cross-star feature prediction"""
        # Get star pair - O(1) operation!
        star1_idx, star2_idx = self._get_star_pair(idx)
        
        # Star 1: Source star (with features as input context)
        sample_idx1 = self.split_indices[star1_idx]
        sample1 = self.raw_data[sample_idx1]
        stellar_data1 = sample1.get('stellar_data', {})
        params1 = self._extract_stellar_params(stellar_data1)
        df_index1 = sample1.get('index')
        features1 = torch.from_numpy(self._apply_feature_normalization(self.features_array[df_index1], cache_key=df_index1))

        # Star 2: Target star (pre-computed closest match)
        sample_idx2 = self.split_indices[star2_idx]
        sample2 = self.raw_data[sample_idx2]
        stellar_data2 = sample2.get('stellar_data', {})
        params2 = self._extract_stellar_params(stellar_data2)
        df_index2 = sample2.get('index')
        features2 = torch.from_numpy(self._apply_feature_normalization(self.features_array[df_index2], cache_key=df_index2))

        # Parse description for star 2 (target)
        description_text2 = sample2.get('description', '')
        parsed_desc2 = self.parse_description_text(description_text2)
        physical_description2 = parsed_desc2['answer']

        # Create question based on conditioning mode
        if self.condition_on_star1:
            # Conditioned mode: include star1's parameters for context
            question = (f"this is a star with Teff {params1['Teff']:.2f} K, "
                       f"logg {params1['logg']:.2f}, and FeH {params1['FeH']:.2f}. "
                       f"describe a star with Teff {params2['Teff']:.2f} K, "
                       f"logg {params2['logg']:.2f}, and FeH {params2['FeH']:.2f}")
        else:
            # Non-conditioned mode: only ask to describe star2 without star1 context
            question = (f"describe a star with Teff {params2['Teff']:.2f} K, "
                       f"logg {params2['logg']:.2f}, and FeH {params2['FeH']:.2f}")

        # Create answer: only the physical description of star 2
        # The features are available separately in 'features_star2'
        answer = physical_description2

        # Tokenize question and answer
        question_tokens, num_tok_q = self._tokenize_text_no_pad(question, bos=True)
        answer_tokens, num_tok_a = self._tokenize_text_no_pad(answer, bos=False)

        # Calculate available space after reserving feature space at the beginning
        available_length = self.max_length - self.num_spectral_features

        # Combine sequences
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
        # Structure: [FEATURE_SPACE (-100)] + [question_tokens] + [answer_tokens] + optional follow-ups + [PADDING]
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

        # Optionally add synthesized follow-up QA turns
        if self.enable_followup:
            self._append_followup_turns(full_sequence, target_sequence, params2, sample_idx2)

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

        # Extract normalized parameter tensors
        numeric_tensor1 = self._extract_numeric_tensor(stellar_data1)
        numeric_tensor2 = self._extract_numeric_tensor(stellar_data2)

        return {
            'input_ids': input_ids,                    # [-100,-100,Q1,Q2,...,A1,A2,...,PAD,PAD,...]
            'target_ids': target_ids,                  # [-100,-100,-100,-100,...,A1,A2,...,-100,-100,...]
            'input_length': num_tok_q,                 # Length of question portion
            'feature_start_indices': feature_start_idx,    # Where features are inserted (0)
            'feature_length': self.num_spectral_features,  # Number of feature tokens
            'question_start_indices': question_start_idx,  # Where question begins
            'answer_start_indices': answer_start_idx,      # Where answer begins
            'target_length': base_answer_length,                # Length of base answer portion
            'input_text': question,                    # Question text
            'target_text': answer,                     # Answer text
            'features_star1': features1,               # Star 1 features (input context)
            'features_star2': features2,               # Star 2 features (prediction target)
            'masked_spectra': features1,               # Spectral features for star 1 (required by model)
            'stellar_params_star1': params1,           # Star 1 params (Teff, logg, FeH)
            'stellar_params_star2': params2,           # Star 2 params (Teff, logg, FeH)
            'stellar_data_star1': stellar_data1,       # Full stellar data for star 1
            'stellar_data_star2': stellar_data2,       # Full stellar data for star 2
            'stellar_data': stellar_data2,             # Alias for compatibility (points to target star)
            'obsid_star1': sample1.get('obsid', None),
            'obsid_star2': sample2.get('obsid', None),
            'df_index_star1': df_index1,
            'df_index_star2': df_index2,
            'sample_index_star1': sample_idx1,
            'sample_index_star2': sample_idx2,
            'y_numeric_star1': numeric_tensor1,        # Normalized [Teff, logg, FeH] for star 1
            'y_numeric_star2': numeric_tensor2,        # Normalized [Teff, logg, FeH] for star 2
            'condition_on_star1': self.condition_on_star1,  # Whether conditioning is enabled
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


def create_feature_prediction_dataloaders(
    json_file: str,
    features_array: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    world_size: int = 1,
    device: Optional[str] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for feature prediction task

    Args:
        json_file: Path to JSON file with stellar data
        features_array: Numpy array of spectral features (required)
        batch_size: Batch size per GPU
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        num_workers: Number of data loading workers
        cache_dir: Directory to cache split indices
        world_size: Number of GPUs for distributed training
        device: Device string for data loading
        **dataset_kwargs: Additional arguments for dataset class
                          (including condition_on_star1: bool to control conditioning mode
                           and random_pairing: bool to control pairing strategy)

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, test dataloaders
        
    Note:
        For datasets with random_pairing=True, call dataset.shuffle_pairs(epoch) 
        at the start of each training epoch to avoid overfitting.
    """

    shared_feature_stats = dataset_kwargs.pop('feature_stats', None)

    # Create datasets for each split
    train_dataset = StellarFeaturePredictionDataset(
        json_file=json_file,
        features_array=features_array,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        feature_stats=shared_feature_stats,
        **dataset_kwargs
    )

    feature_stats = train_dataset.get_feature_normalization_stats(copy=True)
    if feature_stats is None:
        feature_stats = shared_feature_stats

    val_dataset = StellarFeaturePredictionDataset(
        json_file=json_file,
        features_array=features_array,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        feature_stats=feature_stats,
        **dataset_kwargs
    )

    test_dataset = StellarFeaturePredictionDataset(
        json_file=json_file,
        features_array=features_array,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        feature_stats=feature_stats,
        **dataset_kwargs
    )

    # Create samplers for distributed training
    if world_size > 1 and dist.is_initialized():
        rank = dist.get_rank()
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train if train_sampler is None else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """
    Quick test to verify the dataset works correctly.
    Run: python data/dataset_feature_pred.py
    """
    import sys

    print("="*80)
    print("Testing StellarFeaturePredictionDataset")
    print("="*80)

    # Default paths (adjust if needed)
    json_file = "/data/TalkingLatents/data/dataset/stellar_descriptions_questions.json"
    features_file = "/data/TalkingLatents/logs/2025-07-29/features.npy"
    tokenizer_path = "/data/.llama/Llama3.1-8B/tokenizer.model"

    # Check if files exist
    if not os.path.exists(json_file):
        print(f"ERROR: JSON file not found: {json_file}")
        print("Please adjust the path in the script or provide it as argument.")
        sys.exit(1)

    if not os.path.exists(features_file):
        print(f"ERROR: Features file not found: {features_file}")
        print("Please adjust the path in the script or provide it as argument.")
        sys.exit(1)

    print(f"\n✓ Found JSON file: {json_file}")
    print(f"✓ Found features file: {features_file}")

    # Load features
    print(f"\nLoading features...")
    features_array = np.load(features_file)
    print(f"✓ Features shape: {features_array.shape}")

    # Create a small dataset (conditioned mode)
    print(f"\nCreating conditioned dataset...")
    dataset = StellarFeaturePredictionDataset(
        json_file=json_file,
        features_array=features_array,
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        tokenizer_path=tokenizer_path,
        max_length=1024,
        feature_precision=4,
        cache_dir='./cache_test',
        condition_on_star1=True  # Default conditioned mode
    )

    print(f"✓ Conditioned dataset created with {len(dataset)} samples")
    
    # Create non-conditioned dataset for comparison
    print(f"\nCreating non-conditioned dataset...")
    dataset_unconditioned = StellarFeaturePredictionDataset(
        json_file=json_file,
        features_array=features_array,
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        tokenizer_path=tokenizer_path,
        max_length=1024,
        feature_precision=4,
        cache_dir='./cache_test',
        condition_on_star1=False  # Non-conditioned mode
    )
    
    # Create random pairing dataset for comparison
    print(f"\nCreating random pairing dataset...")
    dataset_random = StellarFeaturePredictionDataset(
        json_file=json_file,
        features_array=features_array,
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        tokenizer_path=tokenizer_path,
        max_length=1024,
        feature_precision=4,
        cache_dir='./cache_test',
        condition_on_star1=True,  
        random_pairing=True  # Random pairing mode
    )

    print(f"✓ Non-conditioned dataset created with {len(dataset_unconditioned)} samples")
    print(f"✓ Random pairing dataset created with {len(dataset_random)} samples")

    # Test shuffling for random dataset
    print(f"\nTesting shuffle functionality...")
    print("Initial star2 indices:", [dataset_random._get_star_pair(i)[1] for i in range(5)])
    dataset_random.shuffle_pairs(epoch=1)
    print("After shuffle (epoch 1):", [dataset_random._get_star_pair(i)[1] for i in range(5)])
    dataset_random.shuffle_pairs(epoch=2) 
    print("After shuffle (epoch 2):", [dataset_random._get_star_pair(i)[1] for i in range(5)])

    # Get samples from both datasets
    print("\n" + "="*80)
    print("Sample Output Comparison:")
    print("="*80)
    sample_conditioned = dataset[0]
    sample_unconditioned = dataset_unconditioned[0]
    sample_random = dataset_random[0]

    print(f"\nStar Parameters (same for both modes):")
    print(f"  Star 1 - Teff: {sample_conditioned['stellar_params_star1']['Teff']:.2f}, logg: {sample_conditioned['stellar_params_star1']['logg']:.2f}, FeH: {sample_conditioned['stellar_params_star1']['FeH']:.2f}")
    print(f"  Star 2 - Teff: {sample_conditioned['stellar_params_star2']['Teff']:.2f}, logg: {sample_conditioned['stellar_params_star2']['logg']:.2f}, FeH: {sample_conditioned['stellar_params_star2']['FeH']:.2f}")

    print(f"\nCONDITIONED MODE Question ({sample_conditioned['input_length']} tokens):")
    print(f"  {sample_conditioned['input_text']}")

    print(f"\nNON-CONDITIONED MODE Question ({sample_unconditioned['input_length']} tokens):")
    print(f"  {sample_unconditioned['input_text']}")

    print(f"\nAnswer preview (same for both modes, {sample_conditioned['target_length']} tokens, showing first 300 chars):")
    answer_preview = sample_conditioned['target_text'][:300] + "..." if len(sample_conditioned['target_text']) > 300 else sample_conditioned['target_text']
    print(f"  {answer_preview}")

    print(f"\nDataset configuration:")
    print(f"  Similarity-based (conditioned): condition_on_star1={sample_conditioned['condition_on_star1']}, random_pairing=False")
    print(f"  Non-conditioned: condition_on_star1={sample_unconditioned['condition_on_star1']}, random_pairing=False")  
    print(f"  Random pairing: condition_on_star1={sample_random['condition_on_star1']}, random_pairing=True")

    print(f"\nTensor Shapes (same for both modes):")
    print(f"  input_ids: {sample_conditioned['input_ids'].shape}")
    print(f"  target_ids: {sample_conditioned['target_ids'].shape}")
    print(f"  features_star1: {sample_conditioned['features_star1'].shape}")
    print(f"  features_star2: {sample_conditioned['features_star2'].shape}")
    print(f"  y_numeric_star1: {sample_conditioned['y_numeric_star1'].shape if sample_conditioned['y_numeric_star1'] is not None else None}")
    print(f"  y_numeric_star2: {sample_conditioned['y_numeric_star2'].shape if sample_conditioned['y_numeric_star2'] is not None else None}")

    print(f"\nSequence Structure (conditioned mode):")
    print(f"  Feature space: positions {sample_conditioned['feature_start_indices']} to {sample_conditioned['feature_start_indices'] + sample_conditioned['feature_length']} (length: {sample_conditioned['feature_length']})")
    print(f"  Question: positions {sample_conditioned['question_start_indices']} to {sample_conditioned['answer_start_indices']} (length: {sample_conditioned['input_length']})")
    print(f"  Answer: positions {sample_conditioned['answer_start_indices']} to {sample_conditioned['answer_start_indices'] + sample_conditioned['target_length']} (length: {sample_conditioned['target_length']})")

    # Test dataloader
    print("\n" + "="*80)
    print("Testing DataLoader:")
    print("="*80)

    train_loader, val_loader, test_loader = create_feature_prediction_dataloaders(
        json_file=json_file,
        features_array=features_array,
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        num_workers=0,
        cache_dir='./cache_test',
        tokenizer_path=tokenizer_path,
        max_length=1024
    )

    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  target_ids: {batch['target_ids'].shape}")
    print(f"  features_star1: {batch['features_star1'].shape}")
    print(f"  features_star2: {batch['features_star2'].shape}")
    print(f" example input ids: ", batch['input_ids'][0][:20])
    print(f" question: ", batch['input_text'][0])
    print(f" answer: ", batch['target_text'][0])

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)

    # Cleanup
    import shutil
    if os.path.exists('./cache_test'):
        shutil.rmtree('./cache_test')
        print("\n✓ Cleaned up test cache")
