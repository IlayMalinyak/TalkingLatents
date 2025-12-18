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


import os
# Reusing the setup from dataset_interpert.py
# os.system('pip install tiktoken fairscale fire blobfile') 
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from llama3.llama.tokenizer import Tokenizer
from data.transforms import RandomMasking
from data.feature_normalizer import FeatureNormalizer
# Reuse existing dataset class as base or just copy necessary parts
# Since we need to modify core methods like __getitem__ and _create_splits significantly, 
# it is cleaner to copy the class and modify it rather than subclassing and overriding almost everything 
# that depends on internal state like self.raw_data structure.
from data.dataset_interpert import StellarQuestionsDataset, collate_fn, _stack_numeric


class StellarDiverseDataset(StellarQuestionsDataset):
    """
    StellarDiverseDataset for hybrid QA structure.
    
    Structure of JSON:
    [
      {
        "index": 3,
        "obsid": 154001006,
        "qa_pairs": [
          {"type": "retrieval", "question": "...", "answer": "..."},
          {"type": "classification", "question": "...", "answer": "..."},
          {"type": "reasoning", "question": "...", "answer": "..."}
        ],
        ...
      },
      ...
    ]
    
    Logic:
    1. First 1000 samples -> TEST set (High Quality).
    2. __getitem__: Primary QA is always "retrieval".
    3. Follow-up: Randomly chosen from "classification" or "reasoning".
    """
    
    def _load_data(self):
        """Load data from JSON file with new structure"""
        print(f"Loading data from {self.json_file}...")
        
        with open(self.json_file, 'r') as f:
            self.raw_data = json.load(f)
            
        print(f"Loaded {len(self.raw_data)} samples from JSON")
        
        # Filter samples to ensure they have qa_pairs
        valid_samples = []
        for sample in self.raw_data:
            if 'qa_pairs' in sample and isinstance(sample['qa_pairs'], list) and len(sample['qa_pairs']) > 0:
                 valid_samples.append(sample)
            else:
                # Fallback check if it still matches the old structure just in case, but request says new structure
                pass
        
        if len(valid_samples) < len(self.raw_data):
            print(f"Filtered {len(self.raw_data) - len(valid_samples)} invalid samples (missing qa_pairs).")
        
        self.raw_data = valid_samples
        print(f"Working with {len(self.raw_data)} valid samples.")

        # Extract dataframe indices
        self.df_indices = []
        for sample in self.raw_data:
            df_idx = sample.get('index')
            if df_idx is not None:
                self.df_indices.append(df_idx)
            else:
                pass 
                # print(f"Warning: Sample missing dataframe index: {sample.get('obsid', 'unknown')}")

    def _create_splits(self, train_ratio: float, val_ratio: float, test_ratio: float, cache_dir: Optional[str] = None):
        """
        Create train/val/test splits.
        Constraint: First 1000 samples are ALWAYS test set.
        The rest are split according to ratios (excluding the forced test set).
        """
        n_samples = len(self.raw_data)
        indices = np.arange(n_samples)
        
        # Cache key
        cache_key = f"diverse_{Path(self.json_file).stem}_{n_samples}_{train_ratio}_{val_ratio}_{test_ratio}_{self.random_state}"
        cache_file = None
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"splits_{cache_key}.npz")
        
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached splits from {cache_file}")
            cached = np.load(cache_file)
            train_indices = cached['train_indices']
            val_indices = cached['val_indices'] 
            test_indices = cached['test_indices']
        else:
            print("Creating new train/val/test splits with first 1000 samples reserved for TEST...")
            
            # Forced test set
            forced_test_indices = indices[:1000]
            remaining_indices = indices[1000:]
            
            # Recalculate ratios for the remaining data
            # The user asked for "first 1000 samples are high quality and should be kept for test set"
            # It implies these ARE the test set (or the core of it).
            # We will split the *remaining* data into Train and Val (and maybe more Test if ratio demands, but usually we just want train/val there)
            
            # Let's normalize train/val ratios to sum to 1 for the remaining data
            # If test_ratio was 0.1, we might effectively ignore it for the remaining part if we consider the 1000 as "the" test set,
            # OR we can treat the 1000 as a separate holdout and still split the rest.
            # However, typically "kept for test set" implies we simply take them out.
            # Let's split remaining into Train/Val.
            
            total_ratio = train_ratio + val_ratio
            if total_ratio <= 0:
                print("Warning: Train+Val ratio <= 0. Using defaults.")
                eff_train_ratio = 0.8
            else:
                eff_train_ratio = train_ratio / total_ratio
                
            train_idx_rem, val_idx_rem = train_test_split(
                remaining_indices,
                train_size=eff_train_ratio,
                random_state=self.random_state,
                shuffle=True
            )
            
            train_indices = train_idx_rem
            val_indices = val_idx_rem
            test_indices = forced_test_indices
            
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

    def _get_qa_pair_by_type(self, qa_pairs: List[Dict], q_type: str) -> Optional[Dict]:
        """Helper to find first QA pair of a specific type."""
        for qa in qa_pairs:
            if qa.get('type') == q_type:
                return qa
        return None

    def _append_followup_turns(self,
                               full_tokens: List[int],
                               target_tokens: List[int],
                               stellar_params: Dict[str, Optional[float]],
                               sample_idx: int = -1) -> Tuple[List[Tuple[str, str]], List[int], List[int]]:
        """
        Append follow-up turns.
        Returns:
            - List of (question, answer) tuples
            - List of input ids for the standalone follow-up sequences
            - List of label ids for the standalone follow-up sequences
        """
        if not self.enable_followup or self.followup_prob <= 0.0:
            return [], [], []
        if self.followup_rng.random() > self.followup_prob:
            return [], [], []

        # Get the sample data again
        sample = self.raw_data[sample_idx]
        qa_pairs = sample.get('qa_pairs', [])
        
        # Candidates for follow-up
        candidates = []
        for qa in qa_pairs:
            t = qa.get('type')
            if t in ['classification', 'reasoning']:
                 candidates.append(qa)
        
        if not candidates:
            return [], [], []

        # Randomly select one
        chosen_qa = self.followup_rng.choice(candidates)
        
        question = chosen_qa.get('question', '')
        answer = chosen_qa.get('answer', '')
        
        # Tokenize and append to main sequence
        text_pairs = []
        standalone_input_ids = []
        standalone_labels = []
        
        question_text = f"\nFollow-up question: {question}\nAnswer:"
        q_tokens, _ = self._tokenize_text_no_pad(question_text, bos=False)
        
        answer_text = (answer or "").strip()
        if not answer_text:
             answer_text = "Information not available."

        a_tokens, _ = self._tokenize_text_no_pad(answer_text, bos=False)
        
        # 1. Append to main sequence (accumulated history)
        self._extend_with_tokens(full_tokens, target_tokens, q_tokens, mask_targets=True)
        self._extend_with_tokens(full_tokens, target_tokens, a_tokens, mask_targets=False)
        text_pairs.append((question, answer_text))
        
        # 2. Create standalone sequence for Cycle CE
        # Input: [BOS] [Question] [Answer]
        # Label: [-100] [-100...Q] [Answer]
        
        # Usually checking tokenizer.bos_id would be better, but assuming 128000 for llama3 or passed implicitly
        # For simplicity, we just aggregate the tokens we got.
        # If the model expects specific formatting for standalone, ensure it here.
        
        # Prepare standalone tokens
        # We might want a BOS token if not present?
        # _tokenize_text_no_pad returns raw tokens.
        
        # Let's simple concat: Q tokens + A tokens
        standalone_input_ids.extend(q_tokens)
        standalone_labels.extend([-100] * len(q_tokens))
        
        standalone_input_ids.extend(a_tokens)
        standalone_labels.extend(a_tokens)
        
        return text_pairs, standalone_input_ids, standalone_labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with Retrieval QA as primary."""
        sample_idx = self.split_indices[idx]
        sample = self.raw_data[sample_idx]
        
        qa_pairs = sample.get('qa_pairs', [])
        
        # Primary: Retrieval (same as before)
        retrieval_qa = self._get_qa_pair_by_type(qa_pairs, 'retrieval')
        if retrieval_qa:
            question_text = retrieval_qa.get('question', '')
            answer_text = retrieval_qa.get('answer', '')
        else:
            if qa_pairs:
                question_text = qa_pairs[0].get('question', '')
                answer_text = qa_pairs[0].get('answer', '')
            else:
                question_text = ""
                answer_text = ""
        
        # Get raw tokens (no padding)
        question_tokens, num_tok_q = self._tokenize_text_no_pad(question_text, bos=True)
        answer_tokens, num_tok_a = self._tokenize_text_no_pad(answer_text, bos=False)
        
        # Calculate available space 
        available_length = self.max_length - self.num_spectral_features
        
        # Combine sequences BEFORE padding
        combined_tokens = question_tokens + answer_tokens
        total_tokens = len(combined_tokens)
        
        # Handle truncation
        if total_tokens > available_length:
            combined_tokens = combined_tokens[:available_length]
            if num_tok_q > available_length:
                num_tok_q = available_length
                num_tok_a = 0
            elif num_tok_q + num_tok_a > available_length:
                num_tok_a = available_length - num_tok_q
            total_tokens = available_length
        
        # Create full sequence
        feature_start_idx = 0
        full_sequence: List[int] = [-100] * self.num_spectral_features
        target_sequence: List[int] = [-100] * self.num_spectral_features

        # Add question
        question_start_idx = len(full_sequence)
        base_question_tokens = question_tokens[:num_tok_q]
        full_sequence.extend(base_question_tokens)
        target_sequence.extend([-100] * len(base_question_tokens))

        # Add answer
        answer_start_idx = len(full_sequence)
        base_answer_tokens = answer_tokens[:num_tok_a]
        full_sequence.extend(base_answer_tokens)
        target_sequence.extend(base_answer_tokens)
        base_answer_length = len(base_answer_tokens)

        # Follow-up
        stellar_data = sample.get('stellar_data', {})
        stellar_params = self._extract_physical_params(stellar_data)
        followup_text_pairs = []
        followup_input_ids = []
        followup_labels = []
        
        if self.enable_followup:
            followup_text_pairs, fp_input, fp_label = self._append_followup_turns(
                full_sequence, target_sequence, stellar_params, sample_idx
            )
            # Pad standalone followup sequences to a fixed length (e.g. 256 or max_length)
            # Or just let collate_fn pad them? 
            # Ideally collate_fn handles padding, but here we can return tensors.
            # Let's pad to a reasonable max length for followups or clip.
            
            if fp_input:
                # Convert to tensor and pad roughly or keep as list for collate?
                # Using max_length might be too much if max_length is large.
                # Let's use self.max_length for consistency but we might want shorter.
                
                # Check limits
                if len(fp_input) > self.max_length:
                    fp_input = fp_input[:self.max_length]
                    fp_label = fp_label[:self.max_length]
                
                # Pad
                pad_len = self.max_length - len(fp_input)
                fp_input_padded = fp_input + [0] * pad_len # Assuming 0 is pad
                fp_label_padded = fp_label + [-100] * pad_len
                
                followup_input_ids = torch.tensor(fp_input_padded, dtype=torch.long)
                followup_labels = torch.tensor(fp_label_padded, dtype=torch.long)
            else:
                 # Should we return empty tensors?
                 # If we return None, collate needs to handle it.
                 pass
        
        # Pad main sequence
        remaining_space = self.max_length - len(full_sequence)
        if remaining_space > 0:
            full_sequence.extend([-100] * remaining_space)
            target_sequence.extend([-100] * remaining_space)
        elif remaining_space < 0:
            full_sequence = full_sequence[:self.max_length]
            target_sequence = target_sequence[:self.max_length]

        input_ids = torch.tensor(full_sequence, dtype=torch.long)
        target_ids = torch.tensor(target_sequence, dtype=torch.long)
        
        # Features
        df_index = sample.get('index')
        if self.features_array is not None and df_index is not None:
             norm_features = self._apply_feature_normalization(self.features_array[df_index])
             features = torch.from_numpy(norm_features)
             masked_spectra = features
             spectra = features
        else:
             spectra, masked_spectra, _ = self.get_raw_spectra(sample['obsid'])
             features = masked_spectra

        obsid = sample.get('obsid', None)
        numeric_tensor = self._extract_numeric_tensor(stellar_data)

        res = {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'input_length': num_tok_q,
            'feature_start_idx': feature_start_idx,
            'feature_length': self.num_spectral_features,
            'question_start_idx': question_start_idx,
            'answer_start_idx': answer_start_idx,
            'target_length': base_answer_length,
            'input_text': question_text,
            'target_text': answer_text,
            'followup_turns': followup_text_pairs,
            'features': features,
            'spectra': spectra,
            'masked_spectra': masked_spectra,
            'stellar_data': stellar_data,
            'obsid': obsid,
            'df_index': df_index,
            'sample_index': sample_idx,
            'y_numeric': numeric_tensor,
        }
        
        if len(followup_input_ids) > 0:
            res['followup_input_ids'] = followup_input_ids
            res['followup_labels'] = followup_labels
            
        return res

from data.dataset_interpert import collate_fn as base_collate_fn

def diverse_collate_fn(batch):
    batch_dict = base_collate_fn(batch)
    
    # Handle followup tensors
    # Collect valid (non-empty) tensors
    f_input_list = [item.get('followup_input_ids') for item in batch if item.get('followup_input_ids') is not None]
    f_label_list = [item.get('followup_labels') for item in batch if item.get('followup_labels') is not None]
    
    if f_input_list:
        # Stack
        batch_dict['followup_input_ids'] = torch.stack(f_input_list)
        batch_dict['followup_labels'] = torch.stack(f_label_list)
    
    return batch_dict

def create_diverse_dataloaders(json_file: str,
                             features_array: Optional[np.ndarray] = None,
                             batch_size: int = 32,
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.2, 
                             test_ratio: float = 0.0,
                             random_state: int = 42,
                             num_workers: int = 0,
                             cache_dir: Optional[str] = None,
                             world_size: int = 1,
                             device: Optional[str] = None,
                             **dataset_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Re-implementing logic to use diverse_collate_fn
    
    # 1. Create Datasets
    # We instantiate the dataset class directly. 
    # Note: StellarDiverseDataset does internally split logic based on its instance
    # But usually we create 3 instances or 1 and split?
    # The StellarQuestionsDataset logic is: __init__ calls _create_splits.
    # So we need to create 3 instances, one for each split.
    
    # We first load feature stats if we want consistent normalization?
    # The base implementation does this:
    # 1. Create temp dataset to get stats (if needed)
    # 2. Create train/val/test datasets
    # 3. Create DataLoaders
    
    # Ideally, we utilize the same flow.
    
    feature_stats = None
    if features_array is not None and dataset_kwargs.get('normalize_features', True):
        # Create a temporary train dataset to compute statistics
        temp_ds = StellarDiverseDataset(
            json_file=json_file,
            features_array=features_array,
            split='train',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            cache_dir=cache_dir,
            **dataset_kwargs
        )
        feature_stats = temp_ds.get_feature_normalization_stats()
        del temp_ds
    
    # Create datasets
    train_dataset = StellarDiverseDataset(
        json_file=json_file,
        features_array=features_array,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        feature_stats=feature_stats,
        **dataset_kwargs
    )
    
    val_dataset = StellarDiverseDataset(
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
    
    test_dataset = StellarDiverseDataset(
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

    # Loader args
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=diverse_collate_fn, # Use our custom collate
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    # Distributed Samplers
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            shuffle=True, 
            seed=random_state,
            drop_last=True
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
        
        # Train loader gets drop_last=True via sampler (if dist) or arg
        # Note: When sampler is provided to DataLoader, drop_last arg in DataLoader matches sampler or is ignored?
        # Actually DataLoader has its own drop_last. When sampler is used, DataLoader drop_last is often ignored/mutually exclusive?
        # No, for DistributedSampler, the sampler handles indices.
        # But DataLoader still batches them.
        # If DistributedSampler.drop_last=True, it ensures total samples % world_size == 0 (by dropping).
        # But we ALSO need DataLoader to drop the last batch if it is incomplete per-GPU.
        
        # Re-defining loader_kwargs_train specifically
        loader_kwargs_train = loader_kwargs.copy()
        loader_kwargs_train['drop_last'] = True

        train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs_train)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, **loader_kwargs)
    else:
        loader_kwargs_train = loader_kwargs.copy()
        loader_kwargs_train['drop_last'] = True
        
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs_train)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        
    return train_loader, val_loader, test_loader
