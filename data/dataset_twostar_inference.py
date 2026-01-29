import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Optional, Tuple, Dict, Any, List
import os
from data.dataset_interpert import StellarQuestionsDataset

class StellarTwoStarInferenceDataset(StellarQuestionsDataset):
    """
    Dataset for generating 2-star comparison inference samples.
    Randomly pairs stars and asks comparison questions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We need a robust list of indices to sample from
        # self.split_indices is populated by _create_splits in base class

    def _create_splits(self, train_ratio: float, val_ratio: float, test_ratio: float, cache_dir: Optional[str] = None):
        """Override to simply use all data for the requested split (usually test)."""
        n_samples = len(self.raw_data)
        indices = np.arange(n_samples)
        # We assign all indices to the current split to ensure we can sample from everything
        self.split_indices = indices
        print(f"2-Star Inference Split ({self.split}): Using all {len(self.split_indices)} available samples.")
    
    def _get_comparison_question(self, 
                               params_a: Dict[str, Optional[float]], 
                               params_b: Dict[str, Optional[float]],
                               desc_a: str = "",
                               desc_b: str = "") -> Tuple[str, str, str]:
        """
        Generate a comparison question and answer based on stellar parameters.
        Returns: (attribute_name, question_text, answer_text)
        """
        comparisons = [
            ('Teff', 'hotter', 'temperature'),
            ('Teff', 'cooler', 'temperature'),
            ('logg', 'higher logg', 'surface gravity'),
            ('logg', 'lower logg', 'surface gravity'),
            ('FeH', 'more metal-rich', 'metallicity'),
            ('FeH', 'more metal-poor', 'metallicity'),
        ]
        
        # Filter valid comparisons (where both stars have the param)
        valid = []
        for param, adj, human_name in comparisons:
            if (params_a.get(param) is not None and params_b.get(param) is not None):
                 valid.append((param, adj, human_name))
                 
        if not valid:
            return "unknown", "unknown", f"Description A: {desc_a}\nDescription B: {desc_b}"

        param, adj_comp, human_name = random.choice(valid)
        
        val_a = params_a[param]
        val_b = params_b[param]
        
        # Determine winner
        if 'hotter' in adj_comp or 'higher' in adj_comp or 'rich' in adj_comp:
            winner = 'A' if val_a > val_b else 'B'
        else: # cooler, lower, poor
            winner = 'A' if val_a < val_b else 'B'
            
        # We need to return the adjective for the question construction in __getitem__
        # because the exact phrasing "tell which is [adj]" depends on it.
        # So we'll return the adjective (e.g. "hotter", "more metal-rich") 
        
        answer = f"Star {winner} is {adj_comp}. \nDescription A: {desc_a}\nDescription B: {desc_b}"
        
        return param, adj_comp, answer

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1. Select two diverse samples
        idx_a = self.split_indices[idx]
        idx_b = self.split_indices[random.randint(0, len(self.split_indices) - 1)]
        
        # Ensure they are different
        while idx_b == idx_a and len(self.split_indices) > 1:
            idx_b = self.split_indices[random.randint(0, len(self.split_indices) - 1)]
            
        sample_a = self.raw_data[idx_a]
        sample_b = self.raw_data[idx_b]
        
        # 2. Extract params
        # 2. Extract params
        params_a = self._extract_physical_params(sample_a.get('stellar_data', {}))
        params_b = self._extract_physical_params(sample_b.get('stellar_data', {}))

        # Extract descriptions from JSON (parsed if string, or used directly if appropriate)
        def get_desc(samp):
            d = samp.get('description', '')
            parsed = self.parse_description_text(d)
            return parsed.get('answer', d) # Prefer parsed 'answer' (the description), fall back to raw

        desc_a = get_desc(sample_a)
        desc_b = get_desc(sample_b)
        
        # 3. Generate comparison question components
        # We change _get_comparison_question to return the adjective instead of full question
        comp_param, comp_adj, answer_text = self._get_comparison_question(params_a, params_b, desc_a, desc_b)
        
        # 4. Tokenize components
        q1_text = "Describe this star."
        q2_text = f"Next, describe this star and tell which is {comp_adj}."
        
        # We construct the sequence: [FeatA] [Q1] [FeatB] [Q2] [Answer]
        
        tokens_q1, _ = self._tokenize_text_no_pad(q1_text, bos=True)
        # For subsequent parts, usually no BOS if it flows, but "Next..." might start a sentence. 
        # Llama tokenizer might handle spaces. Let's assume bos=False for continuation.
        tokens_q2, _ = self._tokenize_text_no_pad(q2_text, bos=False)
        tokens_a, _ = self._tokenize_text_no_pad(answer_text, bos=False)
        
        # 5. Construct Sequence
        full_tokens = []
        
        # Feature A slots
        start_idx_a = len(full_tokens)
        full_tokens.extend([-100] * self.num_spectral_features)
        
        # Q1
        full_tokens.extend(tokens_q1)
        
        # Feature B slots
        start_idx_b = len(full_tokens)
        full_tokens.extend([-100] * self.num_spectral_features)
        
        # Q2
        full_tokens.extend(tokens_q2)
        
        # Answer
        answer_start_idx = len(full_tokens)
        full_tokens.extend(tokens_a)
        
        # 6. Pad
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        else:
            pad_len = self.max_length - len(full_tokens)
            full_tokens.extend([-100] * pad_len)
            
        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        
        # 7. Get features
        def get_feat(samp):
            if self.features_array is not None and samp.get('index') is not None:
                ft = self.features_array[samp['index']]
                if self.normalize_features:
                    ft = self._apply_feature_normalization(ft)
                return torch.from_numpy(ft)
            else:
                _, masked, _ = self.get_raw_spectra(samp['obsid'])
                return masked

        feat_a = get_feat(sample_a).float()
        feat_b = get_feat(sample_b).float()
        
        # 8. Indices
        indices_a = torch.arange(start_idx_a, start_idx_a + self.num_spectral_features, dtype=torch.long)
        indices_b = torch.arange(start_idx_b, start_idx_b + self.num_spectral_features, dtype=torch.long)
        
        # 9. Metadata
        metadata = {
            'star_a_obsid': sample_a.get('obsid'),
            'star_b_obsid': sample_b.get('obsid'),
            'star_a_params': params_a,
            'star_b_params': params_b,
            'comparison_param': comp_param,
            'question': f"{q1_text} {q2_text}",
            'expected_answer': answer_text,
            'raw': {
                'star_a': sample_a,
                'star_b': sample_b
            }
        }
        
        return {
            'input_ids': input_ids,
            'star_a_features': feat_a,
            'star_b_features': feat_b,
            'star_a_feature_indices': indices_a,
            'star_b_feature_indices': indices_b,
            'mode': 'two_star',
            'metadata': metadata,
            'target_ids': input_ids.clone(), # dummy
            'input_length': len(tokens_q1) + len(tokens_q2), # Approx length
            'feature_length': self.num_spectral_features * 2,
            'answer_start_idx': answer_start_idx
        }

def collate_twostar_fn(batch):
    # Simple collation
    input_ids = torch.stack([b['input_ids'] for b in batch])
    star_a_feat = torch.stack([b['star_a_features'] for b in batch])
    star_b_feat = torch.stack([b['star_b_features'] for b in batch])
    star_a_idx = torch.stack([b['star_a_feature_indices'] for b in batch])
    star_b_idx = torch.stack([b['star_b_feature_indices'] for b in batch])
    answer_start_indices = torch.tensor([b['answer_start_idx'] for b in batch], dtype=torch.long)
    
    metadata = [b['metadata'] for b in batch]
    mode = [b['mode'] for b in batch]
    
    return {
        'input_ids': input_ids,
        'star_a_features': star_a_feat,
        'star_b_features': star_b_feat,
        'star_a_feature_indices': star_a_idx,
        'star_b_feature_indices': star_b_idx,
        'masked_spectra_a': star_a_feat, # Alias for model compatibility
        'masked_spectra_b': star_b_feat,
        'mode': mode,
        'metadata': metadata,
        'answer_start_indices': answer_start_indices
    }

def create_twostar_inference_dataloader(json_file: str,
                                      features_array: Optional[np.ndarray] = None,
                                      batch_size: int = 4,
                                      tokenizer_path: Optional[str] = None,
                                      tokenizer_backend: str = 'llama',
                                      tokenizer: Optional[Any] = None,
                                      num_spectral_features: int = 4,
                                      **kwargs):
    
    dataset = StellarTwoStarInferenceDataset(
        json_file=json_file,
        features_array=features_array,
        split='test', # Use test split logic usually
        # Force test_ratio=1.0 effectively by using indices
        train_ratio=0.0, val_ratio=0.0, test_ratio=1.0,
        tokenizer_path=tokenizer_path,
        tokenizer_backend=tokenizer_backend,
        tokenizer=tokenizer,
        num_spectral_features=num_spectral_features,
        **kwargs
    )
    
    # Return tuple matching typical expectation if unpacked
    return None, None, DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_twostar_fn)
