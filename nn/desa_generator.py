import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import json
from collections import OrderedDict
import warnings
import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from data.transforms import *
# from dataset.dataset import *
# from dataset.sampler import BalancedDistributedSampler
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.models import *
from nn.simsiam import MultiModalSimSiam, MultiModalSimCLR
from nn.moco import *
from nn.multi_modal import *
from nn.simsiam import SimSiam, projection_MLP
from nn.optim import CQR
from nn.utils import init_model, load_checkpoints_ddp, deepnorm_init
from util.utils import *
from nn.train import *
# from tests.test_unique_sampler import run_sampler_tests
from features import multimodal_umap, create_umap

MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder,
            'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

DATASETS = {'LightSpec': LightSpecDataset, 'FineTune': FineTuneDataset, 'Simulation': SimulationDataset, 'Kepler':
                KeplerDataset, 'Spectra': SpectraDataset}

SIMULATION_DATA_DIR = '/data/simulations/dataset_big/lc'
SIMULATION_LABELS_PATH = '/data/simulations/dataset_big/simulation_properties.csv'


def get_kepler_data(data_args, df, transforms):

    return KeplerDataset(df=df,
                        transforms=transforms,
                        target_transforms=transforms,
                        npy_path = '/data/lightPred/data/raw_npy',
                        seq_len=int(data_args.max_len_lc),
                        masked_transforms = data_args.masked_transform,
                        use_acf=data_args.use_acf,
                        use_fft=data_args.use_fft,
                        scale_flux=data_args.scale_flux,
                        labels=data_args.prediction_labels_lc,
                        dims=data_args.dim_lc,
                                )
def get_lamost_data(data_args, df, transforms):

    return SpectraDataset(data_args.spectra_dir, transforms=transforms, df=df, 
                                 max_len=int(data_args.max_len_spectra),
                                 target_norm=data_args.target_norm,)

def get_lightspec_data(data_args, df, light_transforms, spec_transforms):

    return LightSpecDataset(df=df,
                            light_transforms=light_transforms,
                            spec_transforms=spec_transforms,
                            npy_path = '/data/lightPred/data/raw_npy',
                            spec_path = data_args.spectra_dir,
                            light_seq_len=int(data_args.max_len_lc),
                            spec_seq_len=int(data_args.max_len_spectra),
                            use_acf=data_args.use_acf,
                            use_fft=data_args.use_fft,
                            meta_columns=data_args.meta_columns_lightspec,
                            scale_flux=data_args.scale_flux,
                            labels=data_args.prediction_labels_lightspec
                            )

def get_finetune_data(data_args, df, light_transforms, spec_transforms):

    return FineTuneDataset(df=df,
                            light_transforms=light_transforms,
                            spec_transforms=spec_transforms,
                            npy_path = '/data/lightPred/data/raw_npy',
                            spec_path = data_args.spectra_dir,
                            light_seq_len=int(data_args.max_len_lc),
                            spec_seq_len=int(data_args.max_len_spectra),
                            use_acf=data_args.use_acf,
                            use_fft=data_args.use_fft,
                            meta_columns=data_args.meta_columns_finetune,
                            scale_flux=data_args.scale_flux,
                            labels=data_args.prediction_labels_finetune
                            )

def get_simulation_data(data_args, df, light_transforms, spec_transforms):
    return SimulationDataset(df=df,
                        light_transforms=light_transforms,
                        spec_transforms=spec_transforms,
                        npy_path = '/data/lightPred/data/raw_npy',
                        spec_path = data_args.spectra_dir,
                        light_seq_len=int(data_args.max_len_lc),
                        spec_seq_len=int(data_args.max_len_spectra),
                        use_acf=data_args.use_acf,
                        use_fft=data_args.use_fft,
                        meta_columns=data_args.meta_columns_simulation,
                        scale_flux=data_args.scale_flux,
                        labels=data_args.prediction_labels_simulation
                        )


def get_data(data_args, data_generation_fn, dataset_name='FineTune', config=None):


    light_transforms = Compose([ RandomCrop(int(data_args.max_len_lc)),
                            MovingAvg(13),
                            ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                            FFT(seq_len=int(data_args.max_len_lc)),
                            Normalize(['mag_median', 'std']),
                            ToTensor(), ])
    spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                                ToTensor()
                            ])
    if dataset_name == 'Kepler':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_lc)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        spec_transforms = None
        train_dataset = get_kepler_data(data_args, train_df, light_transforms)
        val_dataset = get_kepler_data(data_args, val_df, light_transforms)
        test_dataset = get_kepler_data(data_args, test_df, light_transforms)
    elif dataset_name == 'Spectra':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_spec)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                                ToTensor()
                            ])
        light_transforms = None
        train_dataset = get_lamost_data(data_args, train_df, spec_transforms)
        val_dataset = get_lamost_data(data_args, val_df, spec_transforms)
        test_dataset = get_lamost_data(data_args, test_df, spec_transforms)

    elif dataset_name == 'LightSpec':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_lightspec, 
                                                only_main_seq=data_args.only_main_seq)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        # mask flags for testing
        test_df['planet_flag'] = -1
        test_df['flag_Binary_Union'] = False
        
        train_dataset = get_lightspec_data(data_args, train_df, light_transforms, spec_transforms)
        val_dataset = get_lightspec_data(data_args, val_df, light_transforms, spec_transforms)
        test_dataset = get_lightspec_data(data_args, test_df, light_transforms, spec_transforms)
    elif dataset_name == 'FineTune':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_finetune)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        train_dataset = get_finetune_data(data_args, train_df, light_transforms, spec_transforms)
        val_dataset = get_finetune_data(data_args, val_df, light_transforms, spec_transforms)
        test_dataset = get_finetune_data(data_args, test_df, light_transforms, spec_transforms)
    
    elif dataset_name == 'Simulation':
        light_transforms = Compose([
                            # AvgDetrend(100*48),
                            RandomCrop(int(data_args.max_len_lc)),
                            MovingAvg(13),
                            ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                            # FFT(seq_len=int(data_args.max_len_lc)),
                            Normalize(['std']),
                            ToTensor(), ])
        spec_transforms = Compose([LAMOSTSpectrumPreprocessor(rv_norm=False, continuum_norm=data_args.continuum_norm, plot_steps=False),
                                ToTensor()
                            ])
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_simulation)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        train_dataset = get_simulation_data(data_args, train_df, light_transforms, spec_transforms)
        val_dataset = get_simulation_data(data_args, val_df, light_transforms, spec_transforms)
        test_dataset = get_simulation_data(data_args, test_df, light_transforms, spec_transforms)

    else: 
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    if config is None:
        config = {}
    config.update({
        "light_transforms": str(light_transforms),
        "spec_transforms": str(spec_transforms),
        "train_dataset": str(train_dataset),
        "val_dataset": str(val_dataset),
        "test_dataset": str(test_dataset)
    }
    )
    return train_dataset, val_dataset, test_dataset, config

def get_model(data_args,
            args_dir,
            config,
             local_rank,
             load_individuals=False,
             freeze=False):
    
    light_model_name = data_args.light_model_name
    spec_model_name = data_args.spec_model_name
    light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{light_model_name}_lc'])
    spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{spec_model_name}_spec'])
    conformer_args_spec = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_spec'])
    astroconformer_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['AstroConformer_lc'])
    cnn_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['CNNEncoder_lc'])
    simsiam_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiModalSimSiam'])
    # lightspec_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiEncoder_lightspec'])
    transformer_args_lightspec = Container(**yaml.safe_load(open(args_dir, 'r'))['Transformer_lightspec'])
    # transformer_args_jepa = Container(**yaml.safe_load(open(args_dir, 'r'))['Transformer_jepa'])
    dual_former_args = Container(**yaml.safe_load(open(args_dir, 'r'))['dual_former'])
    projector_args = Container(**yaml.safe_load(open(args_dir, 'r'))['projector'])
    predictor_args = Container(**yaml.safe_load(open(args_dir, 'r'))['predictor'])
    moco_pred_args = Container(**yaml.safe_load(open(args_dir, 'r'))['reg_predictor'])
    loss_args = Container(**yaml.safe_load(open(args_dir, 'r'))['loss'])
    moco_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MoCo'])
    optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
    tuner_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Tuner'])

    latent_dim = len(data_args.prediction_labels_lightspec) if data_args.use_latent else 0
    dual_former_args.latent_dim = latent_dim
    print("latent dim: ", latent_dim)
    print("quantiles: ", optim_args.quantiles)

    light_encoder1 = CNNEncoder(cnn_args_lc)
    light_encoder2 = Astroconformer(astroconformer_args_lc)
    deepnorm_init(light_encoder2, astroconformer_args_lc)
    light_backbone = DoubleInputRegressor(light_encoder1, light_encoder2, light_model_args)
    light_model = MultiTaskSimSiam(light_backbone, light_model_args)
    light_model = init_model(light_model, light_model_args)

    num_params_lc_all = sum(p.numel() for p in light_model.parameters() if p.requires_grad)
    print(f"Number of trainble parameters in light model: {num_params_lc_all}")
    num_params_lc = sum(p.numel() for p in light_model.simsiam.encoder.parameters() if p.requires_grad)
    print(f"Number of trainble parameters in lc encoder: {num_params_lc}")

    spec_model = MODELS[spec_model_name](spec_model_args, conformer_args=conformer_args_spec)
    
    spec_model = init_model(spec_model, spec_model_args)

    num_params_spec_all = sum(p.numel() for p in spec_model.parameters() if p.requires_grad)
    print(f"Number of trainble parameters in spec model: {num_params_spec_all}")
    num_params_spec = sum(p.numel() for p in spec_model.encoder.parameters() if p.requires_grad)
    print(f"Number of trainble parameters in spec encoder: {num_params_spec}")

    light_model_state = light_model.state_dict()
    spec_model_state = spec_model.state_dict()



    if data_args.approach == 'dual_former':
        optim_args.quantiles = [0.5]
    lc_reg_args = predictor_args.get_dict().copy()
    spectra_reg_args = predictor_args.get_dict().copy()
    lc_reg_args['in_dim'] = light_model.simsiam.encoder.output_dim
    lc_reg_args['out_dim'] = len(data_args.prediction_labels_lc) * len(optim_args.quantiles)
    lc_reg_args['w_dim'] = 0
    spectra_reg_args['in_dim'] = spec_model.encoder.output_dim
    spectra_reg_args['out_dim'] = len(data_args.prediction_labels_spec) * len(optim_args.quantiles)
    spectra_reg_args['w_dim'] = 0
    moco_args.freeze_lightcurve = data_args.freeze_backbone
    moco_args.freeze_spectra = data_args.freeze_backbone

    if data_args.approach == 'dual_former' or 'unimodal' in data_args.approach:

        print("Using DualNet with transformer args: ", dual_former_args)

        model = DualNet(light_model.simsiam.encoder, spec_model.encoder, dual_former_args.get_dict(),
                        lc_reg_args, spectra_reg_args, freeze_backbone=data_args.freeze_backbone).to(local_rank)

        print("Applying DeepNorm initialization to DualNet transformer components")
        deepnorm_init(model.dual_former, dual_former_args)

        tuner_args.in_dim = (dual_former_args.embed_dim + latent_dim) * 2

    
    elif data_args.approach == 'jepa':
        
        print("Using JEPA with transformer args: ", transformer_args_lightspec)
        print("Using projector args: ", projector_args)
        print("Using predictor args: ", predictor_args.get_dict())

        model = PredictiveMoco(spec_model.encoder, light_model.simsiam.encoder,
                             transformer_args_lightspec,
                              predictor_args.get_dict(),
                              loss_args,
                               **moco_args.get_dict()).to(local_rank)

        
        
        # Apply DeepNorm initialization to the JEPA transformer components
        deepnorm_init(model.vicreg_predictor, transformer_args_jepa)
    
    elif data_args.approach == 'moco':

        print("Using MoCo with transformer args: ", transformer_args_lightspec)
        model = MultimodalMoCo(spec_model.encoder, light_model.simsiam.encoder,
                             transformer_args_lightspec,
                             **moco_args.get_dict()).to(local_rank)
        # Apply DeepNorm initialization to the MoCo transformer components
        # Only the shared encoder is a transformer architecture
        print("Applying DeepNorm initialization to MoCo transformer components")
        print(f"Using transformer_args_lightspec with num_layers={transformer_args_lightspec.num_layers}, beta={getattr(transformer_args_lightspec, 'beta', 1.0)}")
        deepnorm_init(model.shared_encoder_q, transformer_args_lightspec)
        deepnorm_init(model.shared_encoder_k, transformer_args_lightspec)
    
    elif data_args.approach == 'simsiam':
        backbone = Transformer(transformer_args_lightspec)
        model = MultiModalSimSiam(backbone, light_model.simsiam.encoder, spec_model.encoder, simsiam_args).to(local_rank)
        tuner_args.in_dim = light_model.simsiam.encoder.output_dim


    if data_args.load_checkpoint:
        datetime_dir = os.path.basename(os.path.dirname(data_args.checkpoint_path))
        exp_num = os.path.basename(data_args.checkpoint_path).split('.')[0].split('_')[-1]
        print(datetime_dir)
        print("loading checkpoint from: ", data_args.checkpoint_path)
        model = load_checkpoints_ddp(model, data_args.checkpoint_path)
        print("loaded checkpoint from: ", data_args.checkpoint_path)
        

    if data_args.approach=='finetune':
        model = MocoTuner(model.moco_model, tuner_args.get_dict(), freeze_moco=freeze).to(local_rank)
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainble parameters: {num_params}")
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {all_params}")

    if config is None:
        config = {}
    config.update(
        {
    "data_args": data_args.__dict__,
    "light_model_args": light_model_args.__dict__,
    "light_astroconformer_args": astroconformer_args_lc.__dict__,
    "light_cnn_args": cnn_args_lc.__dict__,
    "spec_model_args": spec_model_args.__dict__,
    "conformer_args_spec": conformer_args_spec.__dict__,
    "transformer_args_lightspec": transformer_args_lightspec.__dict__,
    "moco_args": moco_args.__dict__,
    "predictor_args": predictor_args.__dict__,
    "moco_pred_args": moco_pred_args.__dict__,
    "loss_args": loss_args.__dict__,
    "tuner_args": tuner_args.__dict__,
    "optim_args": optim_args.__dict__,
    "num_params": num_params,
    "model_structure": str(model),
    }
    )
    return model, optim_args, tuner_args, config, light_model, spec_model




