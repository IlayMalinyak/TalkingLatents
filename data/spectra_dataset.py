import os
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import re
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import List, Sequence, Tuple, Optional
import copy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import traceback
import pickle
import hashlib
import h5py
import json

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from data.transforms import *

def _to_1d_float_tensor(x):
    # Accept numpy/torch/list; return 1D torch.float32 tensor (no empty tensors)
    t = torch.as_tensor(x)
    if t.numel() == 0:
        raise SkipSample("empty spectra")
    t = t.flatten().to(torch.float32)
    return t

def _pad_or_trim_1d(t, L):
    t = _to_1d_float_tensor(t)
    n = t.shape[0]
    if n == L:
        return t
    if n > L:
        return t[:L]
    # pad with zeros
    pad = torch.zeros(L - n, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=0)

class SkipSample(Exception):
    pass

class SpectraDataset(Dataset):
    """
    dataset for spectra data
    Args:
        data_dir: path to the data directory
        transforms: transformations to apply to the data
        df: dataframe containing the data paths
        max_len: maximum length of the spectra
        use_cache: whether to use a cache file
        id: column name for the observation id
    """
    def __init__(self, data_dir,
                     transforms=None,
                     df=None,
                    max_len=3909,
                    use_cache=True,
                    store_wv=True,
                    id='combined_obsid',
                    target_norm='solar',
                    labels=['Teff', 'logg', 'FeH'],
                    ):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.target_norm = target_norm
        self.df = df
        self.id = id
        self.labels = labels
        self.store_wv = store_wv
        if df is None:
            cache_file = os.path.join(self.data_dir, '.path_cache.txt')
            
            if use_cache and os.path.exists(cache_file):
                print("Loading cached file paths...")
                with open(cache_file, 'r') as f:
                    self.path_list = np.array([line.strip() for line in f])
            else:
                print("Creating files list...")
                self.path_list = self._file_listing()
                if use_cache:
                    with open(cache_file, 'w') as f:
                        f.write('\n'.join(self.path_list))
        else:
            self.path_list = None
        self.max_len = max_len
        self.mask_transform = RandomMasking(replace_prob=1)
    
    def _file_listing(self):
        
        def process_chunk(file_names):
            return [self.data_dir / name for name in file_names]
        
        file_names = os.listdir(self.data_dir)
        chunk_size = 100000  
        chunks = [file_names[i:i + chunk_size] for i in range(0, len(file_names), chunk_size)]
        
        with ThreadPoolExecutor() as executor:
            paths = []
            for chunk_paths in executor.map(process_chunk, chunks):
                paths.extend(chunk_paths)
        
        return np.array(paths)
        
    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.path_list) if self.path_list is not None else 0

    def read_lamost_spectra(self, filename):
        with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        rv = header['HELIO_RV']
        meta = {'RV': rv, 'wavelength': wv}
        # if self.store_wv:
        #     meta['wavelength'] = wv
        return x, meta
    
    def read_apogee_spectra(self, filename):
        with fits.open(filename) as hdul:
            data = hdul[1].data.astype(np.float32).squeeze()[None]
        meta = {}
        header = hdul[1].header
        # Create pixel array (1-indexed for FITS convention)
        pixels = np.arange(1, data.shape[-1] + 1)
        
        # Calculate log10(wavelength):
        # log_wave = CRVAL1 + CDELT1 * (pixel - CRPIX1)
        log_wavelength = header['CRVAL1'] + header['CDELT1'] * (pixels - header['CRPIX1'])
        
        # Convert to linear wavelength in Angstroms
        wv = 10**log_wavelength
        meta = {'wavelength': wv}
        return data, meta
    

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            obsid = row[self.id]
            info = row.to_dict()

            if self.id == 'obsid':
                obsdir = str(obsid)[:4]
                spectra_filename = os.path.join(self.data_dir, f'{obsdir}/{obsid}.fits')
                if not os.path.exists(spectra_filename):
                    raise SkipSample(f"missing file: {spectra_filename}")
                spectra, meta = self.read_lamost_spectra(spectra_filename)
                info.update(meta); info['obsid'] = obsid

            elif self.id == 'APOGEE_ID':
                spectra_filename = f"/home/ilay.kamai/work/apogee/data/aspcapStar-dr17-{obsid}.fits"
                if not os.path.exists(spectra_filename):
                    raise SkipSample(f"missing file: {spectra_filename}")
                spectra, meta = self.read_apogee_spectra(spectra_filename)
                info.update(meta); info['apogee_id'] = obsid

            else:
                raise SkipSample(f"unknown id type: {self.id}")

            # transforms
            if self.transforms:
                spectra, _, info = self.transforms(spectra, None, info)

            # masking
            spectra_masked, mask, _ = self.mask_transform(spectra, None, info)

            # force shapes — NO empty tensors get past this
            L = int(self.max_len)
            spectra        = _pad_or_trim_1d(spectra, L)
            spectra_masked = _pad_or_trim_1d(spectra_masked, L)
            mask           = _pad_or_trim_1d(mask.to(torch.bool), L)

            # sanitize
            spectra        = torch.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)
            spectra_masked = torch.nan_to_num(spectra_masked, nan=0.0, posinf=0.0, neginf=0.0)

            if 'wavelength' in info:
                wv = np.asarray(info['wavelength']).reshape(-1)
                if wv.size == 0: raise SkipSample("empty wavelength")
                if wv.size != L:
                    # pad wavelength with its last element
                    last = wv[-1]
                    if wv.size < L:
                        wv = np.pad(wv, (0, L - wv.size), mode='edge')
                    else:
                        wv = wv[:L]
                info['wavelength'] = wv

            target = torch.tensor([info[t] for t in self.labels], dtype=torch.float32)

            # small_info = {'id': obsid, 'snr': info['snr']}

            sample = (
                spectra_masked,        # [L]
                spectra,               # [L]
                target,                # [num_labels]
                mask,                  # [L]
                mask.clone(),          # whatever your code expects
                info
            )
            return sample, idx

        except SkipSample as e:
            print("skipping")
            # light log; DataLoader will drop it in collate
            # import logging
            # logging.getLogger().warning(f"Skip idx={idx}: {e}")
            return None  # collate will filter it out

        except Exception as e:
            # unexpected issues → skip too (but with full traceback)
            import logging, traceback
            logging.getLogger().exception(f"Error at idx={idx}: {e}")
            return None