import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import json
import os
import math
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from astropy.io import fits
import sys

# Import utils
from llama3.llama.tokenizer import Tokenizer
from data.transforms import RandomMasking
from data.feature_normalizer import FeatureNormalizer
from src.follow_up_templates import _infer_stellar_type, create_follow_up_specs
from data.dataset_interpert import StellarQuestionsDataset, collate_fn as base_collate_fn

class StellarHybridDataset(StellarQuestionsDataset):
    """
    Hybrid Dataset:
    - Turn 1 (Main Input):
        Q: Dynamic Stellar Type Question (from templates).
        A: Inferred Stellar Type (e.g. "G-type dwarf").
        Input: [Features] [Q1] [A1]
    
    - Turn 2 (Follow-up / Cycle Target):
        Q: "Describe the star." or dynamic (from JSON).
        A: Original Description from JSON.
        Target: [Q2] [A2]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with Hybrid Structure."""
        sample_idx = self.split_indices[idx]
        sample = self.raw_data[sample_idx]
        
        # 1. Parse Original Description (Used for Follow-up A2)
        description_text = sample.get('description', '')
        parsed_desc = self.parse_description_text(description_text)
        
        # 2. Extract Stellar Params & Infer Type (Used for Main A1)
        stellar_data = sample.get('stellar_data', {})
        # Use parent helper
        stellar_params = self._extract_physical_params(stellar_data)
        
        # A1: Inferred Type
        stellar_type_str = _infer_stellar_type(stellar_params)
        
        # Q1: Dynamic Question from Templates
        # self.followup_rng is seeded in __init__
        # We want a simple stellar type question.
        q1_text = "What is the stellar type of this object?" # Default
        specs = create_follow_up_specs(stellar_params, self.followup_rng, max_pairs=1, include_answers=False)
        for spec in specs:
            if spec.get('type') == 'star_type':
                q1_text = spec.get('question', q1_text)
                break
        
        # Q2: Question from JSON (or Default)
        q2_text = parsed_desc.get('question', '').strip()
        if not q2_text:
            q2_text = "Describe the star."
            
        # A2: Original Answer
        a2_text = parsed_desc.get('answer', '').strip()
        if not a2_text:
            a2_text = "No description available."

        # --- TOKENIZATION ---
        
        # Turn 1 (Main)
        q1_tokens, len_q1 = self._tokenize_text_no_pad(q1_text, bos=True)
        a1_tokens, len_a1 = self._tokenize_text_no_pad(stellar_type_str, bos=False)
        
        # Turn 2 (Follow-up)
        q2_tokens, len_q2 = self._tokenize_text_no_pad(q2_text, bos=False)
        a2_tokens, len_a2 = self._tokenize_text_no_pad(a2_text, bos=False)
        
        # --- BUILD MAIN SEQUENCE (Q1 + A1) ---
        # Note: In Late Fusion, spectral features are processed separately, so we DO NOT need placeholders in input_ids.
        
        feature_start_idx = 0 # Legacy logic, keeping variable for compatibility if needed, but it effectively starts at 0 now.
        
        # Get padding token ID
        pad_token_id = self.tokenizer.pad_id if hasattr(self.tokenizer, 'pad_id') else 0
        
        full_sequence: List[int] = []
        target_sequence: List[int] = []
        
        # Add Q1
        question_start_idx = len(full_sequence)
        full_sequence.extend(q1_tokens)
        target_sequence.extend([-100] * len(q1_tokens)) # Mask Q1 in target
        
        # Add A1
        answer_start_idx = len(full_sequence)
        full_sequence.extend(a1_tokens)
        target_sequence.extend(a1_tokens) # Predict A1
        
        # Pad Main Sequence
        remaining_space = self.max_length - len(full_sequence)
        if remaining_space > 0:
            full_sequence.extend([pad_token_id] * remaining_space)
            target_sequence.extend([-100] * remaining_space)
        elif remaining_space < 0:
            full_sequence = full_sequence[:self.max_length]
            target_sequence = target_sequence[:self.max_length]
            
        input_ids = torch.tensor(full_sequence, dtype=torch.long)
        target_ids = torch.tensor(target_sequence, dtype=torch.long)
        
        # --- BUILD FOLLOW-UP SEQUENCE (Q2 + A2) ---
        # [Q2] [A2] 
        fu_sequence = q2_tokens + a2_tokens
        fu_labels = [-100] * len(q2_tokens) + a2_tokens
        
        # Pad Follow-up
        fu_max_len = self.max_length
        rem_fu = fu_max_len - len(fu_sequence)
        if rem_fu > 0:
            fu_sequence.extend([pad_token_id] * rem_fu) 
            fu_labels.extend([-100] * rem_fu) 
        elif rem_fu < 0:
            fu_sequence = fu_sequence[:fu_max_len]
            fu_labels = fu_labels[:fu_max_len]
            
        followup_input_ids = torch.tensor(fu_sequence, dtype=torch.long)
        followup_labels = torch.tensor(fu_labels, dtype=torch.long)
        
        # --- FEATURES ---
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

        # Return Dictionary matching dataset_diverse keys
        res = {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'input_length': len_q1,
            'feature_start_idx': feature_start_idx,
            'feature_length': self.num_spectral_features,
            'question_start_idx': question_start_idx,
            'answer_start_idx': answer_start_idx,
            'target_length': len_a1,
            'input_text': q1_text,
            'target_text': stellar_type_str,
            'followup_input_ids': followup_input_ids,
            'followup_labels': followup_labels,
            'followup_turns': [(q2_text, a2_text)],
            'features': features,
            'spectra': spectra,
            'masked_spectra': masked_spectra,
            'stellar_data': stellar_data,
            'obsid': obsid,
            'df_index': df_index,
            'sample_index': sample_idx,
            'y_numeric': numeric_tensor,
        }
        return res

def hybrid_collate_fn(batch):
    """Collate function that stacks followup tensors."""
    batch_dict = base_collate_fn(batch)
    
    f_input_list = [item.get('followup_input_ids') for item in batch if item.get('followup_input_ids') is not None]
    f_label_list = [item.get('followup_labels') for item in batch if item.get('followup_labels') is not None]
    
    if f_input_list:
        batch_dict['followup_input_ids'] = torch.stack(f_input_list)
        batch_dict['followup_labels'] = torch.stack(f_label_list)
        
    return batch_dict

def create_hybrid_dataloaders(json_file: str,
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
    
    ds_train = StellarHybridDataset(
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
    
    ds_val = StellarHybridDataset(
        json_file=json_file,
        features_array=features_array,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        **dataset_kwargs
    )
    
    ds_test = StellarHybridDataset(
        json_file=json_file,
        features_array=features_array,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        **dataset_kwargs
    )
    
    sampler_train = DistributedSampler(ds_train, shuffle=True) if world_size > 1 else None
    sampler_val = DistributedSampler(ds_val, shuffle=False) if world_size > 1 else None
    sampler_test = DistributedSampler(ds_test, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler_train, 
                              shuffle=(sampler_train is None), num_workers=num_workers,
                              collate_fn=hybrid_collate_fn, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, sampler=sampler_val, 
                            shuffle=False, num_workers=num_workers,
                            collate_fn=hybrid_collate_fn, drop_last=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, sampler=sampler_test, 
                             shuffle=False, num_workers=num_workers,
                             collate_fn=hybrid_collate_fn, drop_last=True)
                             
    return train_loader, val_loader, test_loader
