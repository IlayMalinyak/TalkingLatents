# Stellar Feature Prediction Dataset

## Overview

`dataset_feature_pred.py` provides a dataset for training models to predict latent spectral features from stellar parameters. This inverts the typical stellar parameter prediction task.

## Task Description

**Input (Question)**: Stellar parameters (Teff, logg, FeH)
**Output (Answer)**: Latent features + physical description

### Example

**Question:**
```
Describe a star with Teff 4964.09 K, logg 2.97, and FeH -0.40.
```

**Answer:**
```
The latent features of this star are -0.0000 0.0000 -0.0000 ... [2048 values].
This star exhibits characteristics suggesting it is a red giant with low metallicity...
```

## Usage

### Basic Usage

```python
from data.dataset_feature_pred import (
    StellarFeaturePredictionDataset,
    create_feature_prediction_dataloaders
)
import numpy as np

# Load features array (required)
features_array = np.load('path/to/features.npy')

# Create dataset
dataset = StellarFeaturePredictionDataset(
    json_file='path/to/stellar_data.json',
    features_array=features_array,
    split='train',
    tokenizer_path='path/to/tokenizer.model',
    max_length=1024,
    feature_precision=4,  # decimal places for features
    feature_separator=' '  # separator between feature values
)

# Get a sample
sample = dataset[0]
print(sample['input_text'])   # Question with stellar parameters
print(sample['target_text'])  # Features + description
print(sample['features'].shape)  # [2048] tensor with actual features
```

### Creating DataLoaders

```python
train_loader, val_loader, test_loader = create_feature_prediction_dataloaders(
    json_file='data/dataset/stellar_descriptions_questions_short.json',
    features_array=features_array,
    batch_size=8,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
    num_workers=4,
    cache_dir='./cache',
    tokenizer_path='path/to/tokenizer.model',
    max_length=1024
)
```

## Parameters

### StellarFeaturePredictionDataset

- `json_file` (str): Path to JSON file with stellar data
- `features_array` (np.ndarray): **Required** - Spectral features array
- `split` (str): 'train', 'val', or 'test'
- `train_ratio` (float): Training set proportion (default: 0.7)
- `val_ratio` (float): Validation set proportion (default: 0.15)
- `test_ratio` (float): Test set proportion (default: 0.15)
- `random_state` (int): Random seed for reproducibility (default: 42)
- `filter_valid_params` (bool): Filter samples without valid parameters (default: True)
- `cache_dir` (str): Directory for caching splits (default: None)
- `tokenizer_path` (str): Path to tokenizer model (default: None)
- `max_length` (int): Maximum sequence length (default: 1024)
- `feature_precision` (int): Decimal places for feature values (default: 4)
- `feature_separator` (str): Separator between features (default: ' ')
- `normalize_features` (bool): Standardise feature vectors using train split stats (default: True)
- `feature_stats` (dict): Optional precomputed `{"mean": ..., "std": ...}` to reuse existing scaling
- `feature_norm_epsilon` (float): Floor applied to std dev when normalising (default: 1e-6)

### Feature Normalization

- Feature vectors are z-scored (per-dimension) using the training split by default so the model sees well-scaled inputs.
- Reuse the same scaling for validation/test by passing the dict returned from `dataset.get_feature_normalization_stats()`.
- Convert predictions back to the original scale with `dataset.denormalize_features(tensor)` when logging or exporting results.

## Sample Structure

Each sample returned by the dataset contains:

```python
{
    'input_ids': torch.Tensor,          # Tokenized question + answer [max_length]
    'target_ids': torch.Tensor,         # Targets with question masked [-100s]
    'input_length': int,                # Length of question tokens
    'question_start_idx': int,          # Start index of question (0)
    'answer_start_idx': int,            # Start index of answer
    'target_length': int,               # Length of answer tokens
    'input_text': str,                  # Original question text
    'target_text': str,                 # Original answer text (features + description)
    'features': torch.Tensor,           # Actual feature tensor [feature_dim]
    'stellar_params': dict,             # {'Teff': float, 'logg': float, 'FeH': float}
    'stellar_data': dict,               # Full stellar data from JSON
    'obsid': int,                       # Observation ID
    'df_index': int,                    # Index in features array
    'sample_index': int,                # Index in dataset
    'y_numeric': torch.Tensor,          # Normalized [Teff, logg, FeH] in [0,1]
}
```

## Data Requirements

### JSON File Format

The JSON file should contain entries with:
```json
{
  "index": 0,
  "obsid": 154001001,
  "description": "{\"Question\": \"...\", \"Description\": \"...physical description...\"}",
  "stellar_data": {
    "Teff": 6509.64,
    "logg": 4.172,
    "FeH": -0.054,
    ...
  }
}
```

### Features Array

- Shape: `[n_samples, feature_dim]` (e.g., `[116543, 2048]`)
- Format: NumPy array (`.npy`)
- The `index` field in JSON must match the row index in the features array

## Testing

Run the test script to verify the dataset works:
```bash
python test_feature_pred_dataset.py
```

This will:
1. Load the dataset
2. Display sample outputs
3. Test dataloaders
4. Verify batch shapes

## Notes

- **Feature Length**: With 2048 features at 4 decimal places, the feature string alone is ~15,000 characters. Consider using `max_length=1024` or higher.
- **Memory**: Feature arrays can be large (e.g., 116k samples × 2048 features × 4 bytes = ~950 MB)
- **Normalization**: Stellar parameters are normalized to [0,1] range based on bounds:
  - Teff: [3000, 7500] K
  - logg: [0.0, 5.0]
  - FeH: [-3.0, 0.5]

## Differences from StellarQuestionsDataset

| Feature | StellarQuestionsDataset | StellarFeaturePredictionDataset |
|---------|------------------------|----------------------------------|
| Input | Spectral features | Stellar parameters (Teff, logg, FeH) |
| Output | Parameter-conditioned description | Features + description |
| Feature injection | Model injects features | Model generates features in text |
| Use case | Parameter prediction | Feature generation/prediction |
| Max length | 512 (typical) | 1024+ (features are long) |
