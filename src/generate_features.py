import yaml
import pandas as pd
from torch.utils.data import DataLoader
import datetime
import argparse
import torch
import os
from collections import OrderedDict
os.system('pip install tiktoken fairscale fire blobfile torchdiffeq torchcfm transformers bitsandbytes accelerate')

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 
from nn.spectra_model import *
from nn.train import *
from util.visualization import *
from util.utils import *
from nn.optim import CQR
from data.transforms import *
from data.spectra_dataset import SpectraDataset
from nn.spectra_model import SpectralViT, MultiTaskRegressor


# these are just for my convenient. The actual paths are given as arguments
config_path = '/home/ilay.kamai/work/MultiDESA/configs/lamost.yaml'
weights_path = 'pretrained_models/spectra.pth'
lamost_data_dir = '/home/ilay.kamai/work/lamost/data'
simulation_data_dir = "data/dataset_noiseless/lamost"
simulation_df_path = "data/dataset_noiseless/simulation_properties.csv"
lamost_df_path = "home/ilay.kamai/work/lamost/lamost_local_catalog.csv"

BOUNDS = {'MAX_TEFF' : 7500, 'MIN_TEFF' : 3000, 'MAX_LOGG' : 5.0, 'MIN_LOGG' : 0, 'MAX_FEH' : 0.5, 'MIN_FEH' : -3}


def get_lamost_df(df_path):
    print("reading lamost dataframe...")
    lamost_catalog = pd.read_csv(df_path)
    print("Applying basic filters...")
    for s in ['snrg', 'snru', 'snrr', 'snri']:
        lamost_catalog = lamost_catalog[lamost_catalog[s] > 0]
    lamost_catalog['snr'] = lamost_catalog[['snrg', 'snru', 'snrr', 'snri']].mean(axis=1)
    lamost_catalog.rename(columns={'Teff':'teff', 'FeH':'feh'}, inplace=True)
    lamost_catalog = lamost_catalog.dropna(subset=['teff', 'logg', 'feh'])
    lamost_catalog['teff'] = lamost_catalog['teff'] * 5778 # remove old normalization
        
    print("\nNormalizing parameters...")
    print("values ranges: ")
    for c in ['teff', 'logg', 'feh']: 
        C = c.upper()
        if c != 'feh':
            lamost_catalog = lamost_catalog[lamost_catalog[c] > 0]
        print(c, lamost_catalog[c].min(), lamost_catalog[c].max(), "nans: ", lamost_catalog[c].isna().sum())
        lamost_catalog[c] = (lamost_catalog[c] - BOUNDS[f'MIN_{C}']) / (BOUNDS[f'MAX_{C}'] - BOUNDS[f'MIN_{C}'])
        print(c, lamost_catalog[c].min(), lamost_catalog[c].max(), "nans: ", lamost_catalog[c].isna().sum())
    
    lamost_catalog['snr'] = lamost_catalog['snr'] / lamost_catalog['snr'].max()
    return lamost_catalog

def get_simulation_df(df_path):
    print("reading simulation dataframe...")
    df = pd.read_csv(df_path)
    return df

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=config_path)
    parser.add_argument("--weights_path", type=str, default=weights_path)
    parser.add_argument("--lamost_data_dir", type=str, default=lamost_data_dir)
    parser.add_argument("--simulation_data_dir", type=str, default=simulation_data_dir)
    parser.add_argument("--simulation_df_path", type=str, default=simulation_df_path)
    parser.add_argument("--lamost_df_path", type=str, default=lamost_df_path)
    parser.add_argument("--output_path", type=str, default="logs/2025-07-29/features.npy")
    return parser.parse_args()


def load_model(config):
    print("loading model...")
    # config is already a dict from yaml.safe_load
    
    # Check for SpectralViT config first
    if 'SpectralViT' in config:
        print("Loading SpectralViT model...")
        model_args = Container(**config['SpectralViT'])
        # In MultiDESA/src/lamost.py, transformer_args uses Conformer config
        conformer_args = Container(**config['Conformer'])
        
        model = SpectralViT(model_args, transformer_args=conformer_args)
        
        # Load checkpoint
        ckpt_path = model_args.checkpoint_path
        print(f"Loading checkpoint from {ckpt_path}")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location='cpu')
            # Handle DDP prefix if present
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}")

    else:
        # Fallback to old behavior
        print("Loading MultiTaskRegressor model...")
        spectra_args = Container(**config['MultiTaskRegressor'])
        conformer_args = Container(**config['Conformer'])
        model = MultiTaskRegressor(spectra_args, conformer_args)
        # Weights for MTR are loaded via load_checkpoints_ddp in run() function usually,
        # but let's check run(). run() calls load_checkpoints_ddp AFTER load_model.
        # So we don't need to load weights here for MTR.
        
    model.eval()
    return model
def predict(model, test_dl, device, max_iter=100):
    loss_fn = CQR(quantiles=[0.1,0.25,0.5,0.75,0.9], reduction='none')
    ssl_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,
                                 weight_decay=1e-6)

    trainer = MaskedRegressorTrainer(model=model, optimizer=optimizer,
                                     criterion=loss_fn, ssl_criterion=ssl_loss_fn,
                                     output_dim=3, scaler=None,
                                     scheduler=None, train_dataloader=None,
                                     val_dataloader=None, device=device,
                                     num_quantiles=5,
                                    log_path=None, range_update=None,
                                     accumulation_step=1, max_iter=max_iter, w_name=None,
                                    exp_name=f"predict")

    return trainer.predict(test_dl, device=device)


def test_predictions(log_dir, quantile_labels=['teff', 'logg', 'feh'],
                     umap_labels=['teff', 'logg', 'feh'],
                     savedir='figs'
                     ):
    info_df = pd.read_csv(os.path.join(log_dir, 'info.csv'))
    preds = np.load(os.path.join(log_dir, 'preds.npy'))
    targets = np.load(os.path.join(log_dir, 'targets.npy'))
    features = np.load(os.path.join(log_dir, 'features.npy'))
    xs = np.load(os.path.join(log_dir, 'xs.npy'))
    decodes = np.load(os.path.join(log_dir, 'decodes.npy'))

    plot_quantiles(targets, preds, [0.1,0.25,0.5,0.75,0.9], quantile_labels, savedir=savedir)
    plot_decode(xs, decodes, info_df, num_sampels=10, savedir=savedir)
    plot_umap(features, info_df, umap_labels, savedir=savedir)


def run(config_path, weights_path, data_dir, df_path, simulation=True, max_iter=100):
    cur_time = datetime.date.today().strftime("%Y-%m-%d")
    if simulation:
        cur_time = cur_time + "-simulation"
    savedir = f'logs/{cur_time}/figs'
    os.makedirs(f'logs/{cur_time}', exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = yaml.safe_load(open(config_path, 'r'))
    model = load_model(cfg)
    model = load_checkpoints_ddp(model, weights_path)
    model.to(device)
    model = model.eval()
    train_df = get_simulation_df(df_path) if simulation else get_lamost_df(df_path)
    rv_norm = not simulation
    file_type = 'pqt' if simulation else 'fits'
    id = 'Simulation Number' if simulation else 'obsid'
    labels = ['teff', 'logg', 'feh']
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=rv_norm), ToTensor()])
    train_ds = SpectraDataset(cfg['Data']['spectra_dir'],
                            transforms=transf, df=train_df,
                            max_len=cfg['Data']['max_len'],
                            target_norm=cfg['Data']['target_norm'],
                            id='obsid',
                            labels=cfg['Data']['prediction_labels'],
                            store_wv=True
                            )
    train_dl = DataLoader(train_ds,
                          batch_size=16,
                          collate_fn=collate_with_idx,
                          )

    for i, (batch, idx) in enumerate(train_dl):
        if batch is not None:
             spectra_masked,spectra,target, mask,_, info = batch
             print(spectra_masked.shape, spectra.shape,target.shape)
        if i > 3:
            break
    preds, targets, features, tokens, decodes, xs, info, mean_loss = predict(model, train_dl, device, max_iter=max_iter)
    print(preds.shape, targets.shape, tokens.shape, features.shape, xs.shape, decodes.shape, info.keys())
    if info:
        max_length = max(len(v) for v in info.values())
        for key in info:
            current_length = len(info[key])
            if current_length < max_length:
                # Pad with NaN for numeric data, or None for mixed data
                info[key].extend([np.nan] * (max_length - current_length))

    info_df = pd.DataFrame(info)
    save_dir = f"/home/ilay.kamai/work/TalkingLatents/logs/{cur_time}"
    os.makedirs(save_dir, exist_ok=True)
    info_df.to_csv(f'{save_dir}/info.csv', index=False)
    np.save(f'{save_dir}/preds.npy', preds)
    np.save(f'{save_dir}/targets.npy', targets)
    np.save(f'{save_dir}/features.npy', features)
    np.save(f'{save_dir}/tokens.npy', tokens)
    np.save(f'{save_dir}/xs.npy', xs)
    np.save(f'{save_dir}/decodes.npy', decodes)

    print("outputs saved into: ", save_dir)

    test_predictions(f'{save_dir}', savedir=f'{save_dir}/figs')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specta Feature extraction.')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to config file of the model')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to weights file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data folder')
    parser.add_argument('--meta_file', type=str, required=True,
                        help='Path to csv with meta data (stellar parameters, ids, etc.)')
    parser.add_argument('--simulation', type=bool, default=False,
                        help='Run on simulated data (default: False)')
    parser.add_argument('--max_iter', type=int, default=np.inf,
                        help='Number of iterations (default: 100)')

    args = parser.parse_args()


    run(args.model_config, args.weights_path, args.data_dir, args.meta_file,
        simulation=args.simulation, max_iter=args.max_iter)