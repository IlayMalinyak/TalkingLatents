# Generation Results Analysis

This document explains how to use the `analyze_generation_results.py` module to process and visualize the results from ablation study generation comparisons.

## Overview

The analysis pipeline consists of two main steps:

1. **Generate comparison data**: Run `ablation_postprocess.py` with `--mode followup` to generate a JSON file containing model responses for test samples
2. **Analyze and plot**: Run `analyze_generation_results.py` to extract metrics and create comparison plots

## Step 1: Generate Comparison Data

```bash
python src/ablation_postprocess.py \
    --ablation_dir /path/to/ablation/results \
    --mode followup \
    --max_samples 100 \
    --num_followups 2 \
    --followup_output followup_comparison.json
```

This creates a JSON file with the following structure:

```json
[
  {
    "experiment_name": "ablation_s_fup0_none",
    "config": {
      "followup_trained": false,
      "feature_pred": "none"
    },
    "samples": [
      {
        "stellar_params": {
          "Teff": 5800.0,
          "logg": 4.5,
          "FeH": -0.2
        },
        "original_question": "describe a star with Teff 5800.0 K...",
        "target_answer": "This is a main sequence star...",
        "description_response": "The star has Teff=5800K, logg=4.5...",
        "followup_qa": [...]
      },
      ...
    ]
  },
  ...
]
```

## Step 2: Analyze and Create Plots

```bash
python src/analyze_generation_results.py \
    --json_path /path/to/ablation/results/followup_comparison.json \
    --output_dir /path/to/output/plots
```

### What Gets Analyzed

The analysis extracts and compares:

1. **Stellar Parameters (Teff, logg, FeH)**:
   - Extracts numeric values from generated text using regex patterns
   - Computes Mean Absolute Error (MAE) compared to ground truth
   - Calculates extraction success rate (how often the model mentions parameters)

2. **Stellar Type Classification**:
   - Extracts stellar type from generated text (main sequence, giant, supergiant, white dwarf, subdwarf)
   - Compares with stellar type mentioned in target answer
   - Calculates accuracy and extraction rate

### Generated Plots

The analysis creates 4 plots:

#### 1. `parameter_extraction_rates.png`
Bar chart showing the percentage of samples where each parameter (Teff, logg, FeH) was successfully extracted from the generated text.

**Interpretation**: Higher is better. Shows how often the model mentions specific parameter values.

#### 2. `parameter_mae_comparison.png`
Three bar charts (one per parameter) showing the Mean Absolute Error between predicted and true values.

**Interpretation**: Lower is better. Shows prediction accuracy when the model does provide values.

- Blue bars: Models trained with followup augmentation
- Orange bars: Models trained without followup augmentation

#### 3. `stellar_type_accuracy.png`
Bar chart comparing:
- **Accuracy**: Percentage of correct stellar type predictions
- **Extraction Rate**: Percentage of samples where stellar type was mentioned

**Interpretation**:
- High accuracy + high extraction = model reliably predicts stellar types
- High accuracy + low extraction = model is accurate when it predicts, but often doesn't mention type
- Low accuracy = model frequently misclassifies stars

#### 4. `summary_combined.png`
Combined multi-panel figure with:
- Top row: MAE for each parameter
- Bottom left: Stellar type accuracy
- Bottom middle: Parameter extraction rates
- Bottom right: Summary statistics table

**Interpretation**: Provides a comprehensive overview of all metrics for quick comparison.

## Parameter Extraction Details

### Teff (Effective Temperature)
Looks for patterns like:
- "Teff = 5800 K"
- "Teff=5800K"
- "effective temperature of 5800 K"
- "temperature of 5800 K"

Expected range: 3000-7500 K

### logg (Surface Gravity)
Looks for patterns like:
- "logg = 4.5"
- "log g = 4.5"
- "log(g) = 4.5"
- "surface gravity of 4.5"

Expected range: 0.0-5.0

### FeH (Metallicity)
Looks for patterns like:
- "FeH = -0.5"
- "[Fe/H] = -0.5"
- "metallicity of -0.5"

Expected range: -3.0 to +0.5

## Stellar Type Classification

The analysis recognizes these stellar types:

- **main_sequence**: "main sequence", "main-sequence", "ms star", "dwarf star", "hydrogen burning"
- **giant**: "giant", "evolved", "red giant", "subgiant", "rgb"
- **supergiant**: "supergiant", "super-giant"
- **white_dwarf**: "white dwarf", "white-dwarf", "wd"
- **subdwarf**: "subdwarf", "sub-dwarf", "sdss"

## Example Usage

Full pipeline example:

```bash
# 1. Generate comparison data (may take hours for large max_samples)
python src/ablation_postprocess.py \
    --ablation_dir logs/ablation_study \
    --mode followup \
    --max_samples 50 \
    --num_followups 2 \
    --max_new_tokens 200 \
    --temperature 0.7 \
    --followup_output comparison_50samples.json

# 2. Analyze and create plots
python src/analyze_generation_results.py \
    --json_path logs/ablation_study/comparison_50samples.json \
    --output_dir logs/ablation_study/analysis_plots
```

## Interpreting Results

### What to look for:

1. **Does followup training improve parameter prediction?**
   - Compare MAE between `fup0` (no followup) vs `fup1` (with followup) experiments
   - Lower MAE = better prediction accuracy

2. **Does feature prediction (cross-star training) help?**
   - Compare experiments with different `feature_pred` settings:
     - `none`: No feature prediction task
     - `pred_pair_random`: Random star pairing
     - `pred_pair_nn`: Nearest neighbor pairing
   - Look for improvements in both parameter accuracy and extraction rates

3. **Are parameters being mentioned?**
   - High extraction rate = model consistently mentions parameters
   - Low extraction rate = model often provides qualitative descriptions without numbers

4. **Is stellar type classification reliable?**
   - Accuracy shows correctness when type is mentioned
   - Extraction rate shows how often type is mentioned

### Common patterns:

- **Followup training typically improves**: Parameter extraction rates and accuracy
- **Feature prediction may help**: Models learn better stellar parameter understanding
- **Trade-offs**: Some models may have high accuracy but low extraction (more conservative)

## Customization

### Adding new stellar types:

Edit `STELLAR_TYPE_KEYWORDS` in `analyze_generation_results.py`:

```python
STELLAR_TYPE_KEYWORDS = {
    'main_sequence': ['main sequence', 'main-sequence', 'ms star'],
    'your_new_type': ['keyword1', 'keyword2', 'keyword3'],
}
```

### Modifying parameter extraction patterns:

Edit the regex patterns in `extract_stellar_params_from_text()` function.

### Changing plot styles:

Modify the `create_comparison_plots()` function. The plots use matplotlib and seaborn for styling.

## Troubleshooting

### Issue: "No parameters extracted"
- Check that generated text actually contains parameter values
- Review the regex patterns in `extract_stellar_params_from_text()`
- Examine a few samples manually to see the output format

### Issue: "Low stellar type accuracy"
- Check if target answers actually mention stellar types
- Review the keyword lists in `STELLAR_TYPE_KEYWORDS`
- Consider that some descriptions may be too vague for classification

### Issue: "Missing experiments in plots"
- Ensure all checkpoint files exist in the ablation directory
- Check the console output for warnings about missing experiments
- Verify experiment names match the expected format

## Output Files

After running the analysis, you'll find:

```
output_dir/
├── parameter_extraction_rates.png    # Extraction rate comparison
├── parameter_mae_comparison.png      # Prediction accuracy (MAE)
├── stellar_type_accuracy.png         # Type classification performance
└── summary_combined.png              # Combined overview
```

All plots are saved at 300 DPI for publication quality.
