import torch
from collections import OrderedDict
from typing import List
import pandas as pd

def collate_with_idx(batch):
    # drop skipped items
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None  # DataLoader will hand this to the loop; you can skip the step

    data, idxs = zip(*batch)
    # use your original collate for tensors
    out = kepler_collate_fn(data)  # or collate_triplets(data)
    # attach indices for debugging
    return out, torch.as_tensor(idxs, dtype=torch.long)

def kepler_collate_fn(batch:List):
    """
    Collate function for the Kepler dataset.
    """
    # Separate the elements of each sample tuple (x, y, mask, info) into separate lists
    x1, x2, y, x1_target, x2_target, infos = zip(*batch)

    # Convert lists to tensors
    x1_tensor = torch.stack(x1, dim=0)
    x2_tensor = torch.stack(x2, dim=0)
    y_tensor = torch.stack(y, dim=0)
    x1_target_tensor = torch.stack(x1_target, dim=0)
    x2_target_tensor = torch.stack(x2_target, dim=0)

    return x1_tensor, x2_tensor, y_tensor, x1_target_tensor, x2_target_tensor, infos
class Container(object):
    '''A container class that can be used to store any attributes.'''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def load_dict(self, dict):
        for key, value in dict.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

    def print_attributes(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")

    def get_dict(self):
        return self.__dict__


def load_checkpoints_ddp(model, checkpoint_path, prefix='', load_backbone=False):
    print(f"****Loading  checkpoint - {checkpoint_path}****")
    state_dict = torch.load(f'{checkpoint_path}', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # print(key)
        while key.startswith('module.'):
            key = key[7:]
        if load_backbone:
            if key.startswith('backbone.'):
                key = key[9:]
            else:
                continue
        key = prefix + key
        # print(key, value.shape)
        new_state_dict[key] = value
    state_dict = new_state_dict

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("number of keys in state dict and model: ", len(state_dict), len(model.state_dict()))
    print("number of missing keys: ", len(missing))
    print("number of unexpected keys: ", len(unexpected))
    print("missing keys: ", missing[:10] if len(missing) > 10 else missing)
    print("unexpected keys: ", unexpected[:10] if len(unexpected) > 10 else unexpected)
    
    # Critical validation for physics components
    physics_components = ['feature_encoder', 'evolution_model', 'feature_decoder']
    for component in physics_components:
        component_keys = [k for k in model.state_dict().keys() if component in k]
        loaded_keys = [k for k in state_dict.keys() if component in k]
        if component_keys and not loaded_keys:
            print(f"WARNING: {component} exists in model but no weights loaded!")
        elif component_keys and loaded_keys:
            print(f"✓ {component}: {len(loaded_keys)}/{len(component_keys)} parameters loaded")
    
    return model

def merge_dataframes_and_filter_array(df1, df2, right_on, left_on,  array):
    """
    Merge two dataframes on all columns and filter corresponding numpy array
    Parameters:
    - df1: First dataframe (berger_lamost)
    - df2: Second dataframe (features_meta)
    - array: Numpy array aligned with df2 (features)
    Returns:
    - merged_df: Merged dataframe
    - filtered_array: Filtered numpy array
    """
    # Track original indices
    df2_indexed = df2.reset_index().rename(columns={'index': 'original_idx'})
    # Merge on all original columns from df2
    merged = pd.merge(df1, df2_indexed, right_on=right_on, left_on=left_on, how='inner', suffixes=['', '_2'])
    # Filter array based on surviving indices
    surviving_indices = merged['original_idx'].values
    filtered_array = array[surviving_indices]
    # Clean up merged dataframe
    final_df = merged.drop('original_idx', axis=1).reset_index(drop=True)
    return final_df, filtered_array