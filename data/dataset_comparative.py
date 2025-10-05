import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, Any, List
import os
from pathlib import Path
import re
from astropy.io import fits


import os
os.system('pip install tiktoken fairscale fire blobfile')
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from llama3.llama.tokenizer import Tokenizer
from data.transforms import RandomMasking



class StellarComparativeDataset(Dataset):
    """
    PyTorch Dataset for stellar comparative multiple choice questions
    
    Handles questions with STAR_A and STAR_B placeholders that get replaced 
    with observational data during training.
    
    Args:
        json_file (str): Path to the JSON file with comparative question data
        split (str): One of 'train', 'val', 'test'
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set  
        test_ratio (float): Proportion for test set (remaining after train/val)
        random_state (int): Random seed for reproducible splits
        cache_dir (Optional[str]): Directory to cache split indices for consistency
        tokenizer_path (Optional[str]): Path to SentencePiece tokenizer model
        max_length (int): Maximum sequence length for tokenization
        num_stellar_features (int): Number of tokens to reserve for stellar data
        include_error_stats (bool): Whether to include parameter error statistics
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
                 cache_dir: Optional[str] = None,
                 allow_new_splits: bool = True,
                 tokenizer_path: Optional[str] = None,
                 max_length: int = 512,
                 num_stellar_features: int = 64,
                 include_error_stats: bool = True):
        
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.json_file = json_file
        self.features_array = features_array
        self.split = split
        self.random_state = random_state
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.num_stellar_features = num_stellar_features  # Features for each star (2 stars total)
        self.transforms = spectral_transforms
        self.mask_transform = RandomMasking()
        self.include_error_stats = include_error_stats
        self.tokenizer = None
        
        # Load tokenizer if available
        self._load_tokenizer()
        
        # Load and process data
        self._load_data()
        self.allow_new_splits = allow_new_splits
        self._create_splits(train_ratio, val_ratio, test_ratio, cache_dir)
        
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
            # Fallback: create deterministic tokens based on text
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
        # else:
        #     # Fallback: create deterministic tokens based on text
        #     words = text.lower().split()
        #     token_ids = []
            
        #     for word in words:
        #         word_hash = hash(word) % 10000
        #         token_ids.append(abs(word_hash) + 1)
                
        #     return token_ids, len(token_ids)
    
    def _load_data(self):
        """Load data from JSON file"""
        print(f"Loading comparative questions from {self.json_file}...")
        
        with open(self.json_file, 'r') as f:
            data = json.load(f)
            
        # Handle both old format (list) and new format (dict with questions key)
        if isinstance(data, list):
            self.raw_data = data
            self.error_statistics = None
            self.metadata = None
        else:
            self.raw_data = data.get('questions', [])
            self.error_statistics = data.get('error_statistics', {})
            self.metadata = data.get('metadata', {})
            
        print(f"Loaded {len(self.raw_data)} comparative questions from JSON")
        
        if self.error_statistics:
            print(f"Dataset error statistics available:")
            if self.error_statistics.get('teff_rmse_k'):
                print(f"  Teff RMSE: {self.error_statistics['teff_rmse_k']:.1f} K")
            if self.error_statistics.get('logg_rmse'):
                print(f"  log g RMSE: {self.error_statistics['logg_rmse']:.3f}")
                
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
            cached = np.load(cache_file, allow_pickle=True)
            train_indices = cached['train_indices']
            val_indices = cached['val_indices'] 
            test_indices = cached['test_indices']
        else:
            if cache_file and not self.allow_new_splits:
                raise FileNotFoundError(
                    f"Cached split file not found at {cache_file}. "
                    "Re-run with allow_new_splits=True to generate splits."
                )
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
        
    def format_stellar_data(self, obs_data: Dict[str, Any]) -> str:
        """Format observational data into a readable string"""
        if not obs_data:
            return "No observational data available"
            
        lines = []
        for key, value in obs_data.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    if abs(value) < 0.01 and value != 0:
                        lines.append(f"{key}: {value:.2e}")
                    else:
                        lines.append(f"{key}: {value:.3f}")
                else:
                    lines.append(f"{key}: {value}")
        
        return "; ".join(lines) if lines else "No valid observational data"
    
    def create_multiple_choice_question(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create multiple choice question using STAR_A and STAR_B as normal text
        """
        question_template = sample.get('question', '')
        options = sample.get('options', [])
        correct_label = sample.get('expected_answer_label', 'None').replace('A', 'STAR_A').replace('B', 'STAR_B')
        correct_explanation = sample.get('model_rationale', 'None')
        full_answer = f'{correct_label}. {correct_explanation}'
        
        # Get observational data for later use
        obs_data_a = sample.get('observational_data_a', {})
        obs_data_b = sample.get('observational_data_b', {})
        
        # Keep STAR_A and STAR_B as normal text - no special tokens
        question_text = question_template
        
        # Format options
        options_text = []
        option_labels = []
        for option in options:
            label = option.get('label', 'A').replace('A', 'STAR_A').replace('B', 'STAR_B')
            text = option.get('text', '')
            options_text.append(f"{label}. {text}")
            option_labels.append(label)
        
        # Find correct answer index
        correct_index = 0
        for i, label in enumerate(option_labels):
            if label == correct_label:
                correct_index = i
                break
        
        # Create full question text
        # full_question = question_text + "\n" + "\n".join(options_text)
        full_question = question_text
        
        return {
            'question': question_text,
            'options': options_text,
            'option_labels': option_labels, 
            'correct_label': correct_label,
            'correct_index': correct_index,
            'full_question': full_question,
            'full_answer': full_answer,
            'star_a_data': obs_data_a,
            'star_b_data': obs_data_b,
        }
        
    def __len__(self):
        return len(self.split_indices)
        
    def find_star_positions_and_insert_features(self, tokens: List[int], full_text: str) -> Tuple[List[int], Dict[str, List[int]]]:
        """
        Find "STAR_A" and "STAR_B" tokens and insert K feature slots after them
        """
        # Get token IDs for "STAR_A" and "STAR_B" 
        star_a_token_ids = self.tokenizer.encode(' STAR_A', bos=False, eos=False)[-1]
        star_b_token_ids = self.tokenizer.encode(' STAR_B', bos=False, eos=False)[-1]
        
        # Convert tokens to numpy array if it isn't already, and ensure correct dtype
        tokens_array = np.array(tokens, dtype=np.int64)
        
        # Ensure token IDs are the same dtype
        star_a_token_ids = np.int64(star_a_token_ids)
        star_b_token_ids = np.int64(star_b_token_ids)
        
        # Find positions of STAR_A and STAR_B in token sequence
        star_a_positions = np.where(tokens_array == star_a_token_ids)[0]
        star_b_positions = np.where(tokens_array == star_b_token_ids)[0]
        
        # Create feature tokens (-100 values)
        feature_tokens = np.full(self.num_stellar_features, -100, dtype=tokens_array.dtype)
        
        # Determine insertion positions
        star_a_insert_pos = star_a_positions[0] + 1 if len(star_a_positions) > 0 else 0
        star_b_insert_pos = star_b_positions[0] + 1 if len(star_b_positions) > 0 else 0
        
        # Sort insertion positions in descending order to avoid index shifting issues
        insertion_positions = []
        feature_info = {}
        
        if len(star_a_positions) > 0:
            insertion_positions.append((star_a_insert_pos, 'STAR_A'))
            feature_info['star_a_indices'] = list(range(star_a_insert_pos, star_a_insert_pos + self.num_stellar_features))
        else:
            insertion_positions.append((0, 'STAR_A'))
            feature_info['star_a_indices'] = list(range(0, self.num_stellar_features))
        
        if len(star_b_positions) > 0:
            insertion_positions.append((star_b_insert_pos, 'STAR_B'))
            feature_info['star_b_indices'] = list(range(star_b_insert_pos, star_b_insert_pos + self.num_stellar_features))
        else:
            insertion_positions.append((0, 'STAR_B'))
            feature_info['star_b_indices'] = list(range(0, self.num_stellar_features))
        
        # Sort by position in descending order to insert from right to left
        insertion_positions.sort(key=lambda x: x[0], reverse=True)
        
        # Insert feature tokens
        extended_tokens = tokens_array.copy()
        
        # Handle the case where both need to be inserted at the beginning
        if star_a_insert_pos == 0 and star_b_insert_pos == 0:
            # Insert both at the beginning: STAR_A first, then STAR_B
            all_feature_tokens = np.concatenate([feature_tokens, feature_tokens])
            extended_tokens = np.insert(extended_tokens, 0, all_feature_tokens)
            feature_info['star_a_indices'] = list(range(0, self.num_stellar_features))
            feature_info['star_b_indices'] = list(range(self.num_stellar_features, 2 * self.num_stellar_features))
        else:
            # Insert from right to left to avoid index shifting
            if star_b_insert_pos > star_a_insert_pos:
                # Insert STAR_B first (rightmost)
                extended_tokens = np.insert(extended_tokens, star_b_insert_pos, feature_tokens)
                extended_tokens = np.insert(extended_tokens, star_a_insert_pos, feature_tokens)
                # Update feature_info accounting for the insertion
                feature_info['star_a_indices'] = list(range(star_a_insert_pos, star_a_insert_pos + self.num_stellar_features))
                feature_info['star_b_indices'] = list(range(star_b_insert_pos + self.num_stellar_features, 
                                                            star_b_insert_pos + 2 * self.num_stellar_features))
            else:
                # Insert STAR_A first (rightmost), or they're at the same position
                extended_tokens = np.insert(extended_tokens, star_a_insert_pos, feature_tokens)
                if star_b_insert_pos != star_a_insert_pos:
                    extended_tokens = np.insert(extended_tokens, star_b_insert_pos, feature_tokens)
                    feature_info['star_b_indices'] = list(range(star_b_insert_pos, star_b_insert_pos + self.num_stellar_features))
                    feature_info['star_a_indices'] = list(range(star_a_insert_pos + self.num_stellar_features, 
                                                            star_a_insert_pos + 2 * self.num_stellar_features))
                else:
                    # Same position - insert STAR_B after STAR_A
                    extended_tokens = np.insert(extended_tokens, star_a_insert_pos + self.num_stellar_features, feature_tokens)
                    feature_info['star_a_indices'] = list(range(star_a_insert_pos, star_a_insert_pos + self.num_stellar_features))
                    feature_info['star_b_indices'] = list(range(star_a_insert_pos + self.num_stellar_features, 
                                                            star_a_insert_pos + 2 * self.num_stellar_features))
        
        return extended_tokens.tolist(), feature_info
    
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
                spectra_filename = os.path.join(f'/data/lamost/data', f'{obsdir}/{obsid}.fits')
                spectra, spectra_masked, meta = self.read_lamost_spectra(spectra_filename)
                meta['obsid'] = obsid
        elif id_type == 'APOGEE_ID':
            spectra_filename = f"/data/apogee/data/aspcapStar-dr17-{obsid}.fits"
            spectra, spectra_masked, meta = self.read_apogee_spectra(spectra_filename)
            meta['apogee_id'] = obsid
        else:
            raise ValueError(f"Unknown obsid format: {id_type}")
        return spectra, spectra_masked, meta

    def create_features(self, index, obsid):
        spectra, masked_spectra, _ = self.get_raw_spectra(obsid)
        
        if self.features_array is not None and index is not None:
            features = torch.tensor(self.features_array[index].astype(np.float32))
            masked_spectra = features
        else:
            features = masked_spectra
        
        return spectra, masked_spectra, features
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with multiple choice question and feature slots inserted after STAR_A/STAR_B"""
        sample_idx = self.split_indices[idx]
        sample = self.raw_data[sample_idx]
        
        # Create multiple choice question (no special tokens)
        mcq_data = self.create_multiple_choice_question(sample)
        
        # Tokenize the full question and correct answer (like in dataset_interpert.py)
        question_tokens, num_tok_q = self._tokenize_text_no_pad(mcq_data['full_question'], bos=True)
        answer_tokens, num_tok_a = self._tokenize_text_no_pad(mcq_data['full_answer'], bos=False)
        
        # Combine question and answer tokens before inserting features
        combined_tokens = question_tokens + answer_tokens
        
        # Find STAR_A and STAR_B positions and insert K feature slots after them
        expanded_tokens, feature_indices = self.find_star_positions_and_insert_features(combined_tokens, mcq_data['full_question'] + ' ' + mcq_data['correct_label'])
        
        # Handle truncation if needed
        available_length = self.max_length
        if len(expanded_tokens) > available_length:
            expanded_tokens = expanded_tokens[:available_length]
            
            # Update feature indices after truncation
            feature_indices['star_a_indices'] = [i for i in feature_indices['star_a_indices'] if i < available_length]
            feature_indices['star_b_indices'] = [i for i in feature_indices['star_b_indices'] if i < available_length]
            
            # Adjust token counts if truncated
            if num_tok_q > available_length:
                num_tok_q = available_length
                num_tok_a = 0
            elif num_tok_q + num_tok_a > available_length:
                num_tok_a = available_length - num_tok_q
        
        # Calculate answer start position in expanded tokens
        # The answer starts after the question tokens plus any feature tokens inserted in the question
        answer_start_in_expanded = num_tok_q + len(feature_indices.get('star_a_indices', [])) + len(feature_indices.get('star_b_indices', []))
        
        # Create the full sequence
        full_sequence = expanded_tokens[:]
        
        # Pad remaining space with 0 (pad token)
        remaining_space = self.max_length - len(full_sequence)
        full_sequence.extend([0] * remaining_space)
        
        # Convert to tensor
        input_ids = torch.tensor(full_sequence, dtype=torch.long)
        
        # The target is the correct option index
        target_index = torch.tensor(mcq_data['correct_index'], dtype=torch.long)
        
        # Extract stellar parameter data  
        star_a_params = sample.get('star_a', {})
        star_b_params = sample.get('star_b', {})
        
        # Create feature tensors from observational data
        def obs_data_to_tensor(obs_data: Dict[str, Any]) -> torch.Tensor:
            """Convert observational data to fixed-size tensor"""
            values = []
            for key in sorted(obs_data.keys()):
                val = obs_data[key]
                if isinstance(val, (int, float)):
                    values.append(float(val))
                else:
                    values.append(0.0)  # Default for non-numeric
            
            # Pad or truncate to exact size needed
            target_size = self.num_stellar_features
            if len(values) > target_size:
                values = values[:target_size]
            else:
                values.extend([0.0] * (target_size - len(values)))
                
            return torch.tensor(values, dtype=torch.float32)
        
        star_a_features = obs_data_to_tensor(mcq_data['star_a_data'])
        star_b_features = obs_data_to_tensor(mcq_data['star_b_data'])
        
        # Create target_ids: mask question with -100, keep only answer (like in dataset_interpert.py)
        target_ids = input_ids.clone()
        target_ids[:answer_start_in_expanded] = -100  # Mask question and features
        # Answer tokens and padding remain as they are in input_ids

        # Get other data
        df_indices = sample.get('indices')
        obsids = sample['obsids']
        spectra_a, masked_spectra_a, features_a = self.create_features(df_indices['a'], obsids['a'])
        spectra_b, masked_spectra_b, features_b = self.create_features(df_indices['b'], obsids['b'])

        
        # For compatibility with the training code, we keep target_ids but it will be mostly masked
        # The actual learning will rely on the target_index for multiple choice classification
        
        return {
            'input_ids': input_ids,                           # Sequence with question + answer and -100 slots inserted after STAR_A/STAR_B
            'target_ids': target_ids,                         # Target sequence with question masked (-100) and only answer for training
            'target_index': target_index,                     # Correct multiple choice index  
            'input_length': num_tok_q,                        # Length of question portion
            'target_length': num_tok_a,                       # Length of answer portion
            'answer_start_idx': answer_start_in_expanded,     # Where answer begins in the sequence
            'star_a_feature_indices': torch.tensor(feature_indices['star_a_indices'], dtype=torch.long),  # Exact positions for Star A features
            'star_b_feature_indices': torch.tensor(feature_indices['star_b_indices'], dtype=torch.long),  # Exact positions for Star B features
            'num_stellar_features': self.num_stellar_features,  # K features per star
            'sequence_length': len(expanded_tokens),          # Length before padding
            'question_text': mcq_data['question'],
            'input_text': mcq_data['full_question'],          # For compatibility with dataset_interpert.py
            'target_text': mcq_data['full_answer'],         # The answer text for compatibility
            'options': mcq_data['options'],
            'option_labels': mcq_data['option_labels'],
            'correct_label': mcq_data['correct_label'],
            'full_question_text': mcq_data['full_question'],
            'star_a_features': star_a_features,               # [K] tensor with Star A features
            'star_b_features': star_b_features,               # [K] tensor with Star B features
            'star_a_params': star_a_params,                   # Full stellar parameter data
            'star_b_params': star_b_params,                   # Full stellar parameter data  
            'comparison_type': sample.get('comparison_type', 'unknown'),
            'features_a': features_a,
            'spectra_a': spectra_a,
            'masked_spectra_a': masked_spectra_a,
            'features_b': features_b,
            'spectra_b': spectra_b,
            'masked_spectra_b': masked_spectra_b,
            'pair_id': sample.get('pair_id', ''),
            'obsid': sample.get('obsids', {}),
            'sample_index': sample_idx
        }
    
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


def create_comparative_dataloaders(json_file: str,
                                 batch_size: int = 32,
                                 features_array: Optional[np.ndarray] = None,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 random_state: int = 42,
                                 num_workers: int = 0,
                                 cache_dir: Optional[str] = None,
                                 allow_new_splits: bool = True,
                                 **dataset_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for comparative questions
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, test dataloaders
    """
    print("crete_dataset: features arryu: {features+arra}")
    # Create datasets for each split
    train_dataset = StellarComparativeDataset(
        json_file=json_file,
        features_array = features_array,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        allow_new_splits=allow_new_splits,
        **dataset_kwargs
    )
    
    val_dataset = StellarComparativeDataset(
        json_file=json_file,
        features_array = features_array,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        allow_new_splits=allow_new_splits,
        **dataset_kwargs
    )
    
    test_dataset = StellarComparativeDataset(
        json_file=json_file,
        features_array = features_array,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        allow_new_splits=allow_new_splits,
        **dataset_kwargs
    )

    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if dist.is_initialized() else None
    
    # Create dataloaders
    train_kwargs = dict(
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_comparative_fn,
        drop_last=False,
    )
    if num_workers > 0:
        train_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    
    val_kwargs = dict(
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_comparative_fn,
        drop_last=False,
    )
    if num_workers > 0:
        val_kwargs.update(persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, **val_kwargs)
    
    test_kwargs = dict(
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_comparative_fn,
        drop_last=False,
    )
    if num_workers > 0:
        test_kwargs.update(persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, **test_kwargs)
    
    return train_loader, val_loader, test_loader


def collate_comparative_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for comparative questions with feature placeholders
    """
    # Stack the sequences and targets
    input_ids = torch.stack([item['input_ids'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    target_indices = torch.stack([item['target_index'] for item in batch])
    
    # Stack feature indices - these should all be the same size (K features per star)
    star_a_feature_indices = torch.stack([item['star_a_feature_indices'] for item in batch])
    star_b_feature_indices = torch.stack([item['star_b_feature_indices'] for item in batch])
    
    # Other batch data
    sequence_lengths = torch.tensor([item['sequence_length'] for item in batch], dtype=torch.long)
    input_lengths = torch.tensor([item['input_length'] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)
    answer_start_indices = torch.tensor([item['answer_start_idx'] for item in batch], dtype=torch.long)
    num_stellar_features = batch[0]['num_stellar_features']  # Same for all samples
    
    # Stack stellar features
    star_a_features = torch.stack([item['star_a_features'] for item in batch])
    star_b_features = torch.stack([item['star_b_features'] for item in batch])
    
    # Collect text and metadata
    question_texts = [item['question_text'] for item in batch]
    input_texts = [item['input_text'] for item in batch]  # For compatibility
    target_texts = [item['target_text'] for item in batch]  # For compatibility
    full_question_texts = [item['full_question_text'] for item in batch]
    options = [item['options'] for item in batch]
    option_labels = [item['option_labels'] for item in batch]
    correct_labels = [item['correct_label'] for item in batch]
    comparison_types = [item['comparison_type'] for item in batch]
    pair_ids = [item['pair_id'] for item in batch]
    obsids = [item['obsid'] for item in batch]
    star_a_params = [item['star_a_params'] for item in batch]
    star_b_params = [item['star_b_params'] for item in batch]
    spectra_a = [item['spectra_a'] for item in batch]
    masked_spectra_a = [item['masked_spectra_a'] for item in batch]
    features_a = [item['features_a'] for item in batch]
    spectra_b = [item['spectra_b'] for item in batch]
    masked_spectra_b = [item['masked_spectra_b'] for item in batch]
    features_b = [item['features_b'] for item in batch]
    
    return {
        'input_ids': input_ids,                              # [batch, seq_len] with question + answer and feature placeholders
        'target_ids': target_ids,                            # [batch, seq_len] - question masked, answer for training
        'target_indices': target_indices,                    # [batch] - correct option indices
        'input_lengths': input_lengths,                      # [batch] - question lengths
        'target_lengths': target_lengths,                    # [batch] - answer lengths  
        'answer_start_indices': answer_start_indices,        # [batch] - where answers start
        'star_a_feature_indices': star_a_feature_indices,    # [batch, K] - exact indices for Star A features
        'star_b_feature_indices': star_b_feature_indices,    # [batch, K] - exact indices for Star B features
        'num_stellar_features': num_stellar_features,        # K - number of features per star
        'sequence_lengths': sequence_lengths,                # [batch] - length before padding
        'question_texts': question_texts,
        'input_texts': input_texts,                          # For compatibility with dataset_interpert.py
        'target_texts': target_texts,                        # For compatibility with dataset_interpert.py
        'full_question_texts': full_question_texts,
        'options': options,
        'option_labels': option_labels,
        'correct_labels': correct_labels,
        'comparison_types': comparison_types,
        'pair_ids': pair_ids,
        'features_a': torch.stack(features_a),
        'spectra_a': torch.stack(spectra_a),
        'masked_spectra_a': torch.stack(masked_spectra_a),
        'features_b': torch.stack(features_b),
        'spectra_b': torch.stack(spectra_b),
        'masked_spectra_b': torch.stack(masked_spectra_b),
        'obsid': obsids,
        'star_a_params': star_a_params,
        'star_b_params': star_b_params
    }


# Example usage and testing
if __name__ == "__main__":
    
    TOKENIZER_PATH = "/data/.llama/Llama3.2-1B/tokenizer.model"
    
    # Example JSON file path (replace with your actual path)
    json_path = '/data/TalkingLatents/data/dataset/comparative_dataset.json'
    features_path = '/data/TalkingLatents/logs/2025-07-29/features.npy'
    features_arr = np.load(features_path)
    
    print("Example usage:")
    print("\nCreate train/val/test dataloaders:")
    
    train_loader, val_loader, test_loader = create_comparative_dataloaders(
        json_file=json_path,
        features_array=features_arr,
        batch_size=4,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        tokenizer_path=TOKENIZER_PATH,
        num_stellar_features=4,  # Features per star
        cache_dir='cache/'  # Cache splits for consistency
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Test a few batches
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Target indices shape: {batch['target_indices'].shape}")
        print(f"  Star A feature indices shape: {batch['star_a_feature_indices'].shape}")
        print(f"  Star B feature indices shape: {batch['star_b_feature_indices'].shape}")
        print(f"  Star A features shape: {batch['masked_spectra_a'].shape}")
        print(f"  Star B features shape: {batch['masked_spectra_b'].shape}")
        print(f"  Num stellar features: {batch['num_stellar_features']}")
        print(f"  Comparison types: {batch['comparison_types']}")
        print(f"  Correct labels: {batch['correct_labels']}")
        print(f"  Sample question: {batch['question_texts'][0][:100]}...")
        print(f"  Input IDs sample: {batch['input_ids'][0][:50]}...")  # First 50 tokens
        print(f"  Star A indices: {batch['star_a_feature_indices'][0]}")
        print(f"  Star B indices: {batch['star_b_feature_indices'][0]}")
        print(f" spectral input shape: {batch['masked_spectra_a'].shape}")
        
        if i >= 2:  # Just test a few batches
            break
  
