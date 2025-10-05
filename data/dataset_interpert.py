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
from astropy.io import fits
import re


import os
os.system('pip install tiktoken fairscale fire blobfile')
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from llama3.llama.tokenizer import Tokenizer
from data.transforms import RandomMasking


class StellarQuestionsDataset(Dataset):
    _GLOBAL_NEIGHBOR_CACHE: Dict[Tuple[int, int, str], Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    _GLOBAL_PHYS_STATS: Dict[Tuple[int, Tuple[str, ...]], Tuple[np.ndarray, np.ndarray]] = {}
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
                 max_length: int = 512,
                 num_spectral_features: int = 1,
                 num_neighbor_samples: int = 0,
                 neighbor_cache_path: Optional[str] = None,
                 neighbor_metric: str = 'euclidean',
                 physics_keys: Tuple[str, ...] = ('Teff', 'logg', 'FeH'),
                 normalize_physics: bool = True):
        
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.json_file = json_file
        self.features_array = features_array
        self.num_spectral_features = num_spectral_features
        self.split = split
        self.random_state = random_state
        self.filter_valid_descriptions = filter_valid_descriptions
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.tokenizer = None
        self.transforms = spectral_transforms
        self.mask_transform = RandomMasking()  # Example masking
        self.num_neighbor_samples = max(0, int(num_neighbor_samples))
        self.neighbor_cache_path = neighbor_cache_path
        self.neighbor_metric = neighbor_metric
        self.physics_keys = tuple(physics_keys)
        self.normalize_physics = normalize_physics
        self.physics_mean = None
        self.physics_std = None
        self.physics_mean_tensor = None
        self.physics_std_tensor = None
        self.df_index_to_phys: Dict[int, np.ndarray] = {}
        self.df_index_to_obsid: Dict[int, int] = {}
        self._neighbor_indices: Optional[np.ndarray] = None
        self._neighbor_distances: Optional[np.ndarray] = None
        self._features_key: Optional[int] = None
        if isinstance(self.features_array, np.ndarray):
            try:
                self._features_key = int(self.features_array.__array_interface__['data'][0])
            except Exception:
                self._features_key = id(self.features_array)

        # Load tokenizer if available
        self._load_tokenizer()

        # Load and process data
        self._load_data()
        self._prepare_physics_and_neighbors()
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
    
    def _load_data(self):
        """Load data from JSON file"""
        print(f"Loading data from {self.json_file}...")
        
        with open(self.json_file, 'r') as f:
            self.raw_data = json.load(f)
            
        print(f"Loaded {len(self.raw_data)} samples from JSON")
        
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


    def _prepare_physics_and_neighbors(self) -> None:
        """Compute physics statistics and build (or load) neighbor graph."""
        if not self.physics_keys:
            return

        phys_vectors = []
        for sample in self.raw_data:
            df_idx = sample.get('index')
            stellar_data = sample.get('stellar_data', {}) or {}
            if df_idx is None:
                continue
            self.df_index_to_obsid[df_idx] = int(sample.get('obsid', -1))
            values = []
            valid = True
            for key in self.physics_keys:
                val = stellar_data.get(key)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    valid = False
                    break
                values.append(float(val))
            if valid:
                vec = np.asarray(values, dtype=np.float32)
                self.df_index_to_phys[df_idx] = vec
                phys_vectors.append(vec)

        if phys_vectors:
            stats_key = None
            if self._features_key is not None:
                stats_key = (self._features_key, self.physics_keys)
                cached_stats = self._GLOBAL_PHYS_STATS.get(stats_key)
                if cached_stats is not None:
                    self.physics_mean, self.physics_std = cached_stats
            if self.physics_mean is None:
                if self.normalize_physics:
                    stacked = np.stack(phys_vectors, axis=0)
                    mean = stacked.mean(axis=0)
                    std = stacked.std(axis=0) + 1e-6
                else:
                    mean = np.zeros(len(self.physics_keys), dtype=np.float32)
                    std = np.ones(len(self.physics_keys), dtype=np.float32)
                self.physics_mean = mean.astype(np.float32)
                self.physics_std = std.astype(np.float32)
                if stats_key is not None:
                    self._GLOBAL_PHYS_STATS[stats_key] = (self.physics_mean, self.physics_std)

        if self.physics_mean is not None:
            self.physics_mean_tensor = torch.from_numpy(self.physics_mean.copy())
            self.physics_std_tensor = torch.from_numpy(self.physics_std.copy())

        if self.num_neighbor_samples <= 0 or self.features_array is None:
            return

        cache_key = None
        if self._features_key is not None:
            cache_key = (self._features_key, self.num_neighbor_samples, self.neighbor_metric)
            cached = self._GLOBAL_NEIGHBOR_CACHE.get(cache_key)
            if cached is not None:
                self._neighbor_indices, self._neighbor_distances = cached
                return

        indices, distances = self._build_neighbor_graph()
        if indices is not None:
            self._neighbor_indices = indices
            self._neighbor_distances = distances
            if cache_key is not None:
                self._GLOBAL_NEIGHBOR_CACHE[cache_key] = (indices, distances)

    def _build_neighbor_graph(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Construct k-NN graph over feature embeddings."""
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(
                n_neighbors=min(self.num_neighbor_samples + 1, len(self.features_array)),
                metric=self.neighbor_metric,
                algorithm='auto'
            )
            nn.fit(self.features_array)
            distances, indices = nn.kneighbors(self.features_array)
        except Exception as e:
            print(f"Warning: unable to build neighbor graph via sklearn ({e}). Disabling neighbor supervision.")
            return None, None

        distances = distances.astype(np.float32)
        indices = indices.astype(np.int64)

        if self.neighbor_cache_path:
            try:
                Path(self.neighbor_cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(self.neighbor_cache_path,
                                    indices=indices,
                                    distances=distances)
            except Exception as e:
                print(f"Warning: failed to save neighbor cache to {self.neighbor_cache_path}: {e}")

        return indices, distances

            
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
        # Structure: [FEATURE_SPACE] + [question_tokens] + [answer_tokens] + [PADDING]
        full_sequence = []
        
        # Reserve space for features at the beginning (fill with -100, will be replaced during training)
        feature_start_idx = 0
        full_sequence.extend([-100] * self.num_spectral_features)
        
        # Add question tokens after features
        question_start_idx = len(full_sequence)
        full_sequence.extend(question_tokens[:num_tok_q])
        
        # Add answer tokens  
        answer_start_idx = len(full_sequence)
        full_sequence.extend(answer_tokens[:num_tok_a])
        
        # Pad remaining space with -100
        remaining_space = self.max_length - len(full_sequence)
        full_sequence.extend([-100] * remaining_space)
        
        # Convert to tensor
        input_ids = torch.tensor(full_sequence, dtype=torch.long)
        
        # Create targets: mask features AND question with -100
        target_ids = input_ids.clone()
        target_ids[:answer_start_idx] = -100  # Mask features + question
        # Answer tokens and padding (-100) remain as they are
        
        # Get other data
        df_index = sample.get('index')
        spectra, masked_spectra, _ = self.get_raw_spectra(sample['obsid'])
        
        if self.features_array is not None and df_index is not None:
            features = torch.tensor(self.features_array[df_index].astype(np.float32))
            masked_spectra = features
        else:
            features = masked_spectra
        
        stellar_data = sample.get('stellar_data', {})
        obsid = sample.get('obsid', None)

        # Physics targets (raw + normalized)
        phys_dim = len(self.physics_keys)
        if phys_dim > 0 and df_index is not None and df_index in self.df_index_to_phys:
            physics_target = torch.from_numpy(self.df_index_to_phys[df_index].copy())
            if self.physics_mean_tensor is not None and self.physics_std_tensor is not None:
                physics_target_norm = (physics_target - self.physics_mean_tensor) / self.physics_std_tensor
            else:
                physics_target_norm = physics_target.clone()
            physics_mask = torch.tensor(1.0, dtype=torch.float32)
        else:
            physics_target = torch.zeros(phys_dim, dtype=torch.float32)
            physics_target_norm = physics_target.clone()
            physics_mask = torch.tensor(0.0, dtype=torch.float32)

        # Neighbor context (latents + physics)
        neighbor_latents = torch.zeros((self.num_neighbor_samples, self.features_array.shape[1] if isinstance(self.features_array, np.ndarray) else 1), dtype=torch.float32)
        neighbor_physics = torch.zeros((self.num_neighbor_samples, phys_dim), dtype=torch.float32)
        neighbor_obsids = torch.full((self.num_neighbor_samples,), -1, dtype=torch.long)
        neighbor_mask = torch.zeros((self.num_neighbor_samples,), dtype=torch.float32)
        neighbor_distances = torch.full((self.num_neighbor_samples,), float('inf'), dtype=torch.float32)

        if (
            self.num_neighbor_samples > 0
            and self._neighbor_indices is not None
            and df_index is not None
            and 0 <= df_index < len(self._neighbor_indices)
        ):
            cand_indices = self._neighbor_indices[df_index]
            cand_distances = self._neighbor_distances[df_index] if self._neighbor_distances is not None else np.zeros_like(cand_indices, dtype=np.float32)
            fill_ptr = 0
            for neigh_idx, neigh_dist in zip(cand_indices, cand_distances):
                if neigh_idx == df_index:
                    continue
                if not (0 <= neigh_idx < len(self.features_array)):
                    continue
                if fill_ptr >= self.num_neighbor_samples:
                    break
                neighbor_latents[fill_ptr] = torch.tensor(self.features_array[neigh_idx], dtype=torch.float32)
                if phys_dim > 0 and neigh_idx in self.df_index_to_phys:
                    neighbor_physics[fill_ptr] = torch.from_numpy(self.df_index_to_phys[neigh_idx].astype(np.float32))
                neighbor_obsids[fill_ptr] = int(self.df_index_to_obsid.get(neigh_idx, -1))
                neighbor_mask[fill_ptr] = 1.0
                neighbor_distances[fill_ptr] = float(neigh_dist)
                fill_ptr += 1

        # Index of positive neighbor (default to first valid entry)
        if neighbor_mask.sum() > 0:
            positive_idx = int(torch.argmax(neighbor_mask).item())
        else:
            positive_idx = 0

        return {
            'input_ids': input_ids,                    # [-100,-100,Q1,Q2,A1,A2,-100,-100,...]
            'target_ids': target_ids,                  # [-100,-100,-100,-100,A1,A2,-100,-100,...]
            'input_length': num_tok_q,                 # Length of question portion
            'feature_start_idx': feature_start_idx,    # Where features are inserted (0)
            'feature_length': self.num_spectral_features,        # Number of feature tokens
            'question_start_idx': question_start_idx,  # Where question begins
            'answer_start_idx': answer_start_idx,      # Where answer begins
            'target_length': num_tok_a,                # Length of answer portion
            'input_text': parsed_desc['question'],
            'target_text': parsed_desc['answer'],
            'features': features,
            'spectra': spectra,
            'masked_spectra': masked_spectra,
            'stellar_data': stellar_data,
            'obsid': obsid,
            'df_index': df_index,
            'sample_index': sample_idx,
            'physics_target': physics_target,
            'physics_target_norm': physics_target_norm,
            'physics_mask': physics_mask,
            'neighbor_latents': neighbor_latents,
            'neighbor_physics': neighbor_physics,
            'neighbor_obsids': neighbor_obsids,
            'neighbor_mask': neighbor_mask,
            'neighbor_distances': neighbor_distances,
            'neighbor_target_idx': torch.tensor(positive_idx, dtype=torch.long)
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


def create_stellar_dataloaders(json_file: str,
                             features_array: Optional[np.ndarray] = None,
                             batch_size: int = 32,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             random_state: int = 42,
                             num_workers: int = 0,
                             cache_dir: Optional[str] = None,
                             **dataset_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, test dataloaders
    """
    
    # Create datasets for each split
    train_dataset = StellarQuestionsDataset(
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
    
    val_dataset = StellarQuestionsDataset(
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
    
    test_dataset = StellarQuestionsDataset(
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

    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if dist.is_initialized() else None
    
    # Create dataloaders
    # Build DataLoader kwargs and only set prefetch/persistent when using workers
    train_kwargs = dict(
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
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
        collate_fn=collate_fn,
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
        collate_fn=collate_fn,
        drop_last=False,
    )
    if num_workers > 0:
        test_kwargs.update(persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, **test_kwargs)
    
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
    obsids = [item['obsid'] for item in batch]
    df_indices = [item['df_index'] for item in batch]
    stellar_data = [item['stellar_data'] for item in batch]
    spectra = [item['spectra'] for item in batch]
    masked_spectra = [item['masked_spectra'] for item in batch]
    
    
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
    
    physics_target = torch.stack([item['physics_target'] for item in batch]) if batch[0]['physics_target'].numel() > 0 else torch.empty(len(batch), 0)
    physics_target_norm = torch.stack([item['physics_target_norm'] for item in batch]) if batch[0]['physics_target_norm'].numel() > 0 else torch.empty(len(batch), 0)
    physics_mask = torch.stack([item['physics_mask'] for item in batch])

    neighbor_latents = torch.stack([item['neighbor_latents'] for item in batch]) if batch[0]['neighbor_latents'].numel() > 0 else torch.empty(len(batch), 0, 0)
    neighbor_physics = torch.stack([item['neighbor_physics'] for item in batch]) if batch[0]['neighbor_physics'].numel() > 0 else torch.empty(len(batch), 0, 0)
    neighbor_obsids = torch.stack([item['neighbor_obsids'] for item in batch]) if batch[0]['neighbor_obsids'].numel() > 0 else torch.empty(len(batch), 0, dtype=torch.long)
    neighbor_mask = torch.stack([item['neighbor_mask'] for item in batch]) if batch[0]['neighbor_mask'].numel() > 0 else torch.empty(len(batch), 0)
    neighbor_distances = torch.stack([item['neighbor_distances'] for item in batch]) if batch[0]['neighbor_distances'].numel() > 0 else torch.empty(len(batch), 0)
    neighbor_targets = torch.stack([item['neighbor_target_idx'] for item in batch])

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
        'features': features_tensor,
        'spectra': torch.stack(spectra),
        'masked_spectra': torch.stack(masked_spectra),
        'obsids': obsids,
        'df_indices': df_indices,
        'stellar_data': stellar_data,
        'physics_target': physics_target,
        'physics_target_norm': physics_target_norm,
        'physics_mask': physics_mask,
        'neighbor_latents': neighbor_latents,
        'neighbor_physics': neighbor_physics,
        'neighbor_obsids': neighbor_obsids,
        'neighbor_mask': neighbor_mask,
        'neighbor_distances': neighbor_distances,
        'neighbor_target_idx': neighbor_targets
    }


# Example usage and testing
if __name__ == "__main__":

    TOKENIZER_PATH = "/data/.llama/Llama3.2-1B/tokenizer.model"
    tokenizer = Tokenizer(model_path=TOKENIZER_PATH)

    json_path = '/data/TalkingLatents/data/dataset/stellar_descriptions_questions.json'
    spectral_features = np.load('/data/TalkingLatents/logs/2025-07-29/features.npy')
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
        num_spectral_features=32,
        cache_dir='cache/'  # Cache splits for consistency
    )
    
    for i, data in enumerate(train_loader):
        print(f"Batch input tokens shape: {data['input_ids'].shape}")
        print(f"Batch target tokens shape: {data['target_ids'].shape}")
        print("f start indices:", data['feature_start_indices'], " feature lengths:", data['feature_lengths'])
        print("Answer start indices:", data['answer_start_indices'], " target lengths:", data['target_lengths'])
        print(data['input_ids'][0][:100])
        print(data['target_ids'][0][:100])
        print("tot lengths: ", data['input_lengths'] + data['target_lengths'])

        if i == 10:
            break
