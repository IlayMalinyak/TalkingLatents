import argparse
import yaml
import torch
import numpy as np
import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from astropy.io import fits

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from nn.spectra_model import SpectralViT, MultiTaskRegressor
from util.utils import Container
from collections import OrderedDict

# Import transforms
from data.transforms import *
# Need ensure_backend_config and setup for correct path handling if used
from src.simple_questions import setup

JSON_PATH = '/home/ilay.kamai/work/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json'
SPECTRA_CONFIG_PATH_v2 = '/home/ilay.kamai/work/MultiDESA/configs/lamost.yaml'
SPECTRA_WEIGHTS_PATH_v2 = 'pretrained_models/spectra.pth'
OUTPUT_PATH = '/home/ilay.kamai/work/TalkingLatents/logs/features_v2/features.npy'
OUTPUT_PATH_cls = '/home/ilay.kamai/work/TalkingLatents/logs/features_v2/features_cls.npy'

class AlignedSpectraDataset(Dataset):
    """
    Dataset that loads ALL samples from the JSON and sorts them by 'index'
    to ensure perfect alignment with the global dataframe index.
    Also extracts metadata for info dataframe.
    """
    def __init__(self, json_file, transforms=None):
        self.json_file = json_file
        self.transforms = transforms
        self.mask_transform = RandomMasking()
        
        print(f"Loading data from {json_file}...")
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
            
        print(f"Loaded {len(raw_data)} samples from JSON")
        
        # Sort by index to ensure alignment: arr[i] corresponds to sample with index=i
        self.data = [s for s in raw_data if 'index' in s]
        if len(self.data) < len(raw_data):
            print(f"Warning: {len(raw_data) - len(self.data)} samples missing 'index' field")
            
        self.data.sort(key=lambda x: x['index'])
        
        # Check continuity
        indices = [s['index'] for s in self.data]
        if not indices:
            raise ValueError("No valid data found with 'index' field")
            
        if indices[-1] != len(indices) - 1:
            print(f"Warning: Indices are not continuous! Max index: {indices[-1]}, Count: {len(indices)}")
            self.max_index = indices[-1]
        else:
            self.max_index = len(indices) - 1

        print(f"Dataset prepared. Max Index: {self.max_index}")

    def __len__(self):
        return len(self.data)

    def read_lamost_spectra(self, filename):
        try:
            with fits.open(filename) as hdulist:
                binaryext = hdulist[1].data
                header = hdulist[0].header
            spectra = torch.tensor(binaryext['FLUX'].astype(np.float32))
            wv = binaryext['WAVELENGTH'].astype(np.float32)
            rv = header['HELIO_RV']
            snr = header.get('SNRG', 0.0) # Try getting SNRG from header
            meta = {'RV': rv, 'wavelength': wv, 'snr': snr}
        except FileNotFoundError:
            spectra = torch.zeros(1, 4096, dtype=torch.float32)
            wv = np.linspace(3690, 9100, 4096, dtype=np.float32)
            meta = {'RV': 0.0, 'wavelength': wv, 'snr': 0.0}
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            spectra = torch.zeros(1, 4096, dtype=torch.float32)
            wv = np.linspace(3690, 9100, 4096, dtype=np.float32)
            meta = {'RV': 0.0, 'wavelength': wv, 'snr': 0.0}

        if self.transforms:
            spectra, _, meta = self.transforms(spectra, None, meta)
        
        spectra_masked, mask, _ = self.mask_transform(spectra, None, meta)
        
        if spectra.shape[-1] < 4096:
             pad = torch.zeros(1, 4096 - spectra.shape[-1])
             spectra = torch.cat([spectra, pad], dim=-1)
             spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
             
        return spectra, spectra_masked, meta

    def get_raw_spectra(self, obsid):
        obsdir = str(obsid)[:4]
        spectra_filename = os.path.join(f'/home/ilay.kamai/work/lamost/data', f'{obsdir}/{obsid}.fits')
        return self.read_lamost_spectra(spectra_filename)

    def __getitem__(self, idx):
        sample = self.data[idx]
        obsid = sample.get('obsid')
        global_index = sample.get('index')
        
        full_spectra, masked_spectra, meta = self.get_raw_spectra(obsid)
        
        # Extract metadata
        stellar_data = sample.get('stellar_data', {})
        
        return {
            'spectra': full_spectra,
            'masked_spectra': masked_spectra,
            'index': global_index,
            'obsid': str(obsid),
            'teff': stellar_data.get('Teff', np.nan),
            'logg': stellar_data.get('logg', np.nan),
            'feh': stellar_data.get('FeH', np.nan),
            'snr': meta.get('snr', np.nan)
        }


def load_model(config, weights_path, device):
    print("Loading model...")
    print("loading spectra model v2...")
    config = yaml.safe_load(open(SPECTRA_CONFIG_PATH_v2, 'r'))
    
    if 'SpectralViT' in config:
        print("Loading SpectralViT model...")
        model_args = Container(**config['SpectralViT'])
        conformer_args = Container(**config['Conformer'])
        model = SpectralViT(model_args, transformer_args=conformer_args)
        
        ckpt_path = model_args.checkpoint_path
        print(f"Loading checkpoint from {ckpt_path}")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location='cpu')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Generate V2 features aligned with JSON dataset")
    parser.add_argument('--json_file', type=str, default=JSON_PATH, help='Path to stellar descriptions JSON file')
    parser.add_argument('--model_config', type=str, default=SPECTRA_CONFIG_PATH_v2, help='Path to config file of the model')
    parser.add_argument('--weights_path', type=str, default=SPECTRA_WEIGHTS_PATH_v2, help='Path to weights file (spectra.pth)')
    parser.add_argument('--output_path', type=str, default=OUTPUT_PATH, help='Path to save features (.npy)')
    parser.add_argument('--output_path_cls', type=str, default=OUTPUT_PATH_cls, help='Path to save features (.npy)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    config = yaml.safe_load(open(args.model_config, 'r'))
    model = load_model(config, args.weights_path, device)

    lamost_transforms = Compose([GeneralSpectrumPreprocessor(
                                rv_norm=True,
                                plot_steps=False,
                                resample=False,
                                continuum_norm=True,),
                                ToTensor()
                            ])


    # 2. Create Dataset
    # Using aligned dataset
    print(f"Creating aligned dataset from {args.json_file}...")
    dataset = AlignedSpectraDataset(
        json_file=args.json_file,
        transforms=lamost_transforms
    )
    
    # 3. Generation Loop
    
    max_idx = dataset.max_index
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    results = [] # List of (index, feature_vec, metadata_dict)
    results_cls = [] # List of (index, feature_vec, metadata_dict)
    
    print("Starting generation...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            masked_spectra = batch['masked_spectra'].to(device)
            full_spectra = batch['spectra'].to(device)
            indices = batch['index']
            
            # Metadata batch handling
            # batch keys: 'obsid', 'teff', etc. are lists or tensors
            
            out = model(masked_spectra, full_spectra, return_all=True)
            
            # Using AVG TOKENS as requested
            features = out['tokens'].mean(1) # [B, Dim]
            features_cls = out['cls'] # [B, Dim]
            
            features_cpu = features.cpu().numpy()
            indices_cpu = indices.numpy()
            features_cls_cpu = features_cls.cpu().numpy()
            
            # Collect metadata
            obsids = batch['obsid'] # list of strings
            teffs = batch['teff'].numpy()
            loggs = batch['logg'].numpy()
            fehs = batch['feh'].numpy()
            snrs = batch['snr'].numpy()
            
            for i in range(len(indices_cpu)):
                meta = {
                    'obsid': obsids[i],
                    'teff': teffs[i],
                    'logg': loggs[i],
                    'feh': fehs[i],
                    'snr': snrs[i]
                }
                results.append((indices_cpu[i], features_cpu[i], 
                                features_cls_cpu[i], meta))
    
    # 4. Construct Final Array & DataFrame
    print("Constructing final aligned array and dataframe...")
    sample_feat = results[0][1]
    feat_dim = sample_feat.shape[0]
    
    # Allocate full array
    final_features = np.zeros((max_idx + 1, feat_dim), dtype=np.float32)
    final_features_cls = np.zeros((max_idx + 1, feat_dim), dtype=np.float32)
    
    # Allocate list for dataframe rows
    # Pre-fill with None/NaN so we align by index? 
    # Or just construct sorted DF and then reindex?
    # Reindexing is safer.
    
    meta_rows = []
    
    filled_count = 0
    for idx, feat, feat_cls, meta in results:
        if 0 <= idx <= max_idx:
            final_features[idx] = feat
            final_features_cls[idx] = feat_cls
            meta['index'] = idx
            meta_rows.append(meta)
            filled_count += 1
            
    print(f"Filled {filled_count} samples in array of size {final_features.shape}")
    
    # Save Features
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    np.save(args.output_path, final_features)
    np.save(args.output_path_cls, final_features_cls)
    print(f"Saved aligned features to {args.output_path}")
    
    # Save Info DataFrame
    info_path = os.path.join(output_dir, 'info.csv')
    df = pd.DataFrame(meta_rows)
    # Sort just in case
    df = df.sort_values('index')
    
    # Reindex to ensure 1:1 with features array? 
    # If features array has zeros for missing indices, df should probably have rows (maybe empty/NaN) too
    # to maintain "row N corresponds to index N".
    # Creating a comprehensive index
    full_index = pd.Index(range(max_idx + 1), name='index')
    df = df.set_index('index').reindex(full_index)
    
    df.to_csv(info_path)
    print(f"Saved info dataframe to {info_path}")

if __name__ == '__main__':
    main()
