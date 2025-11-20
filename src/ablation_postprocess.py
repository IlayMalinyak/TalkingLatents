"""
Postprocessing and analysis for ablation study results.
"""
import json
import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
os.system('pip install tiktoken fairscale fire blobfile torchdiffeq torchcfm transformers bitsandbytes accelerate')


# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average with given window size"""
    if window <= 1:
        return values
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')


def load_ablation_results(ablation_dir: Path, experiments: List[str]) -> Dict[str, dict]:
    """
    Load JSON results from all ablation experiments.

    Args:
        ablation_dir: Directory containing ablation results
        experiments: List of experiment names (without .json extension)

    Returns:
        Dictionary mapping experiment names to their data
    """
    data = {}
    for exp in experiments:
        json_path = ablation_dir / f"{exp}.json"
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping")
            continue

        with open(json_path, 'r') as f:
            data[exp] = json.load(f)

        print(f"Loaded {exp}: train_loss={len(data[exp]['train_loss'])}, val_loss={len(data[exp]['val_loss'])}")

    return data


def plot_loss_comparison(
    ablation_dir: Path,
    experiments: Optional[List[str]] = None,
    labels: Optional[Dict[str, str]] = None,
    train_window: int = 50,
    val_window: int = 10,
    output_suffix: str = "",
) -> None:
    """
    Plot training and validation loss comparison across ablation experiments.

    Args:
        ablation_dir: Directory containing ablation results
        experiments: List of experiment names. If None, uses default 6 experiments
        labels: Dictionary mapping experiment names to display labels
        train_window: Window size for smoothing training loss
        val_window: Window size for smoothing validation loss
        output_suffix: Optional suffix to add to output filenames
    """
    # Default experiments if not provided
    if experiments is None:
        experiments = [
            "ablation_s_fup0_none",
            "ablation_s_fup0_pred_pair_random",
            "ablation_s_fup0_pred_pair_nn",
            "ablation_s_fup1_none",
            "ablation_s_fup1_pred_pair_random",
            "ablation_s_fup1_pred_pair_nn",
        ]

    # Default labels if not provided
    if labels is None:
        labels = {
            "ablation_s_fup0_none": "No Followup, No Feature Pred",
            "ablation_s_fup0_pred_pair_random": "No Followup, Random Pairing",
            "ablation_s_fup0_pred_pair_nn": "No Followup, NN Pairing",
            "ablation_s_fup1_none": "Followup, No Feature Pred",
            "ablation_s_fup1_pred_pair_random": "Followup, Random Pairing",
            "ablation_s_fup1_pred_pair_nn": "Followup, NN Pairing",
        }

    # Load data
    data = load_ablation_results(ablation_dir, experiments)

    if not data:
        print("No data loaded, exiting")
        return

    # Create combined figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Training Loss
    print("\nPlotting training losses...")
    for exp in experiments:
        if exp not in data:
            continue

        train_loss = np.array(data[exp]['train_loss'])
        # Remove NaN/inf values
        train_loss = train_loss[~np.isnan(train_loss) & ~np.isinf(train_loss)]

        # Compute moving average
        smoothed_loss = moving_average(train_loss, train_window)

        # Plot
        x = np.arange(len(smoothed_loss))
        ax1.plot(x, smoothed_loss, label=labels.get(exp, exp), linewidth=2, alpha=0.8)
        print(f"  {exp}: min={smoothed_loss.min():.4f}, final={smoothed_loss[-1]:.4f}")

    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training Loss Comparison (smoothed over {train_window} steps)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    print("\nPlotting validation losses...")
    for exp in experiments:
        if exp not in data:
            continue

        val_loss = np.array(data[exp]['val_loss'])
        # Remove NaN/inf values
        val_loss = val_loss[~np.isnan(val_loss) & ~np.isinf(val_loss)]

        # Compute moving average
        smoothed_loss = moving_average(val_loss, val_window)

        # Plot
        x = np.arange(len(smoothed_loss))
        ax2.plot(x, smoothed_loss, label=labels.get(exp, exp), linewidth=2, alpha=0.8)
        print(f"  {exp}: min={smoothed_loss.min():.4f}, final={smoothed_loss[-1]:.4f}")

    ax2.set_xlabel('Validation Steps', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'Validation Loss Comparison (smoothed over {val_window} steps)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)

    # Save combined plot
    plt.tight_layout()
    output_name = f"ablation_loss_comparison{output_suffix}.png"
    output_path = ablation_dir / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot to: {output_path}")
    plt.close()

    # Plot training loss separately
    fig1, ax = plt.subplots(figsize=(10, 6))
    for exp in experiments:
        if exp not in data:
            continue

        train_loss = np.array(data[exp]['train_loss'])
        train_loss = train_loss[~np.isnan(train_loss) & ~np.isinf(train_loss)]
        smoothed_loss = moving_average(train_loss, train_window)
        x = np.arange(len(smoothed_loss))
        ax.plot(x, smoothed_loss, label=labels.get(exp, exp), linewidth=2, alpha=0.8)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Loss Comparison (smoothed over {train_window} steps)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    train_output_name = f"ablation_train_loss{output_suffix}.png"
    train_output_path = ablation_dir / train_output_name
    plt.savefig(train_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training plot to: {train_output_path}")
    plt.close()

    # Plot validation loss separately
    fig2, ax = plt.subplots(figsize=(10, 6))
    for exp in experiments:
        if exp not in data:
            continue

        val_loss = np.array(data[exp]['val_loss'])
        val_loss = val_loss[~np.isnan(val_loss) & ~np.isinf(val_loss)]
        smoothed_loss = moving_average(val_loss, val_window)
        x = np.arange(len(smoothed_loss))
        ax.plot(x, smoothed_loss, label=labels.get(exp, exp), linewidth=2, alpha=0.8)

    ax.set_xlabel('Validation Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Validation Loss Comparison (smoothed over {val_window} steps)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    val_output_name = f"ablation_val_loss{output_suffix}.png"
    val_output_path = ablation_dir / val_output_name
    plt.savefig(val_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved validation plot to: {val_output_path}")
    plt.close()

    print("\nDone!")


def compare_followup_responses(
    ablation_dir: Path,
    experiments: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    max_samples: int = 5,
    num_followups: int = 2,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    output_filename: str = "followup_comparison.json",
) -> None:
    """
    Compare followup question responses across all ablation experiments.

    Loads each trained model and generates responses for both description questions
    and followup questions, even for experiments that didn't use followup during training.

    Args:
        ablation_dir: Directory containing ablation results
        experiments: List of experiment names. If None, uses default 6 experiments
        config_path: Optional path to config JSON. If None, loads from checkpoint dir
        max_samples: Maximum number of test samples to evaluate
        num_followups: Number of followup questions to generate per sample
        max_new_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        seed: Random seed for reproducibility
        output_filename: Name of output JSON file
    """
    # Import required modules
    from src.simple_questions import setup, get_model_path
    from src.simple_questions_multitok import (
        build_model_multitok,
        create_datasets_and_loaders,
        ensure_backend_config,
    )
    from src.training_ablation import make_base_args, apply_ablation_to_args, AblationConfig
    from llama3.llama.tokenizer import Tokenizer
    from src.follow_up_templates import create_follow_up_specs, PARAM_KEY_ALIASES

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Default experiments
    if experiments is None:
        experiments = [
            "ablation_s_fup0_none",
            "ablation_s_fup0_pred_pair_random",
            "ablation_s_fup0_pred_pair_nn",
            "ablation_s_fup1_none",
            "ablation_s_fup1_pred_pair_random",
            "ablation_s_fup1_pred_pair_nn",
        ]

    # Map experiment names to AblationConfig
    exp_to_config = {
        "ablation_s_fup0_none": AblationConfig("short-fup0-none", "short", False, "none"),
        "ablation_s_fup0_pred_pair_random": AblationConfig("short-fup0-pred_pair_random", "short", False, "pred_pair_random"),
        "ablation_s_fup0_pred_pair_nn": AblationConfig("short-fup0-pred_pair_nn", "short", False, "pred_pair_nn"),
        "ablation_s_fup1_none": AblationConfig("short-fup1-none", "short", True, "none"),
        "ablation_s_fup1_pred_pair_random": AblationConfig("short-fup1-pred_pair_random", "short", True, "pred_pair_random"),
        "ablation_s_fup1_pred_pair_nn": AblationConfig("short-fup1-pred_pair_nn", "short", True, "pred_pair_nn"),
    }

    # Load followup JSON file for description-based followups
    followup_data = None
    followup_json_path = '/data/TalkingLatents/data/dataset/stellar_descriptions_questions.json'
    if os.path.exists(followup_json_path):
        with open(followup_json_path, 'r') as f:
            followup_data = json.load(f)
        print(f"Loaded {len(followup_data)} samples from followup JSON: {followup_json_path}")
    else:
        print(f"Warning: Followup JSON not found at {followup_json_path}, will only use stellar type questions")

    # Setup device
    device, _, _ = setup()
    print(f"Using device: {device}")

    # Create base args (we'll modify for each experiment)
    base_args = make_base_args(output_dir=str(ablation_dir))

    # Helper function to parse description text
    def parse_description_text(description: str) -> dict:
        """Parse description text to extract question and answer"""
        result = {"question": "", "answer": ""}
        if not description:
            return result

        # Try to split on common delimiters
        for delimiter in ['\nAnswer:', '\nA:', 'Answer:', 'A:']:
            if delimiter in description:
                parts = description.split(delimiter, 1)
                if len(parts) == 2:
                    question = parts[0].replace('Question:', '').replace('Q:', '').strip()
                    answer = parts[1].strip()
                    result["question"] = question
                    result["answer"] = answer
                    return result

        # If no delimiter found, assume entire text is the answer
        result["answer"] = description.strip()
        return result

    # Helper function to get followup from long JSON
    def get_followup_from_json(sample_idx: int) -> Optional[Tuple[str, str]]:
        """Get followup Q&A from long JSON file at the same index"""
        if followup_data is None or sample_idx >= len(followup_data):
            return None

        followup_sample = followup_data[sample_idx]
        description = followup_sample.get("description", "")

        if not description:
            return None

        parsed = parse_description_text(description)
        question = parsed.get("question", "")
        answer = parsed.get("answer", "")

        if not question or not answer:
            return None

        return question, answer

    # Results storage
    all_results = []

    # Process each experiment
    for exp_name in experiments:
        print(f"\n{'='*80}")
        print(f"Processing experiment: {exp_name}")
        print(f"{'='*80}")

        # Get checkpoint path
        checkpoint_path = ablation_dir / f"{exp_name}_resume_best.pth"
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint {checkpoint_path} not found, skipping")
            continue

        # Get config for this experiment
        cfg = exp_to_config.get(exp_name)
        if cfg is None:
            print(f"Warning: No config found for {exp_name}, skipping")
            continue

        # Apply ablation config to args (but disable followup for inference)
        args = apply_ablation_to_args(base_args, cfg)
        # Force disable followup augmentation during dataset loading for clean test set
        args.enable_followup_augmentation = False

        # Ensure backend config
        backend_config = ensure_backend_config(args)

        # Load tokenizer
        _, tokenizer_path = get_model_path(args)
        tokenizer = Tokenizer(model_path=tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")

        # Create test dataloader (without followup augmentation)
        print("Creating test dataloader...")
        _, _, test_loader = create_datasets_and_loaders(args, device, backend_config)
        print(f"Test loader has {len(test_loader.dataset)} samples")

        # Load model
        print(f"Loading model from {checkpoint_path}")
        model = build_model_multitok(args, device, world_size=1, backend_config=backend_config)

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        del checkpoint

        # Clean state dict (remove 'module.' prefix if present)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            cleaned_state_dict[new_key] = value

        # Load into model
        target_model = model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
        missing, unexpected = target_model.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys (showing up to 5): {missing[:5]}")
        if unexpected:
            print(f"Warning: Unexpected keys (showing up to 5): {unexpected[:5]}")

        model.eval()
        model.to(device)
        print("Model loaded successfully")

        # Generate responses on test samples
        exp_results = {
            'experiment_name': exp_name,
            'config': {
                'followup_trained': cfg.followup,
                'feature_pred': cfg.feature_pred,
            },
            'samples': []
        }

        samples_processed = 0
        print("max sampes: ", max_samples)
        for batch in tqdm(test_loader, desc=f"Generating for {exp_name}", total=len(test_loader)):
            if samples_processed >= max_samples:
                break

            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device)

            batch_size = batch['input_ids'].shape[0]

            for sample_idx in range(batch_size):
                if samples_processed >= max_samples:
                    break

                # Extract sample metadata
                sample_data = {}

                # Get stellar parameters for followup question generation
                # First try to use pre-extracted params (from feature prediction datasets)
                stellar_params = {}
                if 'stellar_params_star2' in batch and isinstance(batch['stellar_params_star2'], list):
                    # Feature prediction dataset - use target star params
                    if sample_idx < len(batch['stellar_params_star2']):
                        stellar_params = batch['stellar_params_star2'][sample_idx] or {}
                elif 'stellar_data' in batch and isinstance(batch['stellar_data'], list):
                    # Regular dataset - extract from stellar_data
                    if sample_idx < len(batch['stellar_data']):
                        raw_params = batch['stellar_data'][sample_idx] or {}
                        for canonical, aliases in PARAM_KEY_ALIASES.items():
                            val = None
                            for key in aliases:
                                if key in raw_params and raw_params[key] is not None:
                                    try:
                                        val = float(raw_params[key])
                                        break
                                    except Exception:
                                        continue
                            stellar_params[canonical] = val

                sample_data['stellar_params'] = stellar_params
                sample_data['sample_idx'] = samples_processed

                # Get original question and target
                # Handle both singular and plural field names (different datasets use different conventions)
                if 'input_texts' in batch and isinstance(batch['input_texts'], list):
                    if sample_idx < len(batch['input_texts']):
                        sample_data['original_question'] = batch['input_texts'][sample_idx]
                elif 'input_text' in batch and isinstance(batch['input_text'], list):
                    if sample_idx < len(batch['input_text']):
                        sample_data['original_question'] = batch['input_text'][sample_idx]

                if 'target_texts' in batch and isinstance(batch['target_texts'], list):
                    if sample_idx < len(batch['target_texts']):
                        sample_data['target_answer'] = batch['target_texts'][sample_idx]
                elif 'target_text' in batch and isinstance(batch['target_text'], list):
                    if sample_idx < len(batch['target_text']):
                        sample_data['target_answer'] = batch['target_text'][sample_idx]

                # Generate response for original description question
                response, _, _, _ = model.generate_response_from_batch(
                    batch_data=batch,
                    batch_idx=sample_idx,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                sample_data['description_response'] = response

                # Generate followup questions and responses
                sample_data['followup_qa'] = []

                # Get the original sample index from the dataset
                original_idx = samples_processed  # This maps to the same index in the long JSON

                # Try to get followup from long JSON first
                followup_specs = []
                qa_pair = get_followup_from_json(original_idx)
                if qa_pair is not None:
                    question, answer = qa_pair
                    followup_specs.append({
                        'question': question,
                        'expected_answer': answer,
                        'type': 'description_from_long_json'
                    })

                # Fall back to stellar type questions if we don't have enough followups
                if len(followup_specs) < num_followups and stellar_params:
                    stellar_followups = create_follow_up_specs(
                        stellar_params,
                        random.Random(seed + samples_processed),
                        max_pairs=num_followups - len(followup_specs),
                        include_answers=True
                    )
                    for spec in stellar_followups:
                        followup_specs.append({
                            'question': spec.get('question', ''),
                            'expected_answer': spec.get('answer', ''),
                            'type': spec.get('type', 'stellar_parameter')
                        })

                # Generate responses for each followup
                for spec in followup_specs[:num_followups]:
                    followup_q = spec.get('question', '')
                    if not followup_q:
                        continue

                    # Get the original input text (question)
                    original_question = sample_data.get('original_question', '')

                    # Build conversation: original Q + generated A + followup Q
                    conversation_text = f"{original_question}\n{response}\n\nFollow-up question: {followup_q}\nAnswer:"

                    # Tokenize the conversation
                    conv_tokens = tokenizer.encode(conversation_text, bos=True, eos=False)

                    # The answer starts right after the conversation (at the end of conv_tokens)
                    # Since we end with "Answer:", the model should generate after that
                    answer_start_idx = len(conv_tokens)

                    # Create modified batch with conversation context
                    followup_batch = {
                        'input_ids': torch.tensor([conv_tokens], dtype=torch.long).to(device),
                        'input_texts': [conversation_text],
                        'answer_start_indices': [answer_start_idx],
                    }

                    # Copy all feature-related fields from the original batch
                    # These are required by generate_response_from_batch

                    # Single-star mode fields
                    if 'masked_spectra' in batch:
                        followup_batch['masked_spectra'] = batch['masked_spectra'][sample_idx:sample_idx+1].to(device)
                    elif 'features' in batch:
                        # Use 'features' as 'masked_spectra' if that's what the batch has
                        followup_batch['masked_spectra'] = batch['features'][sample_idx:sample_idx+1].to(device)

                    if 'feature_start_indices' in batch:
                        followup_batch['feature_start_indices'] = batch['feature_start_indices'][sample_idx:sample_idx+1]

                    # Two-star mode fields (if they exist)
                    if 'masked_spectra_a' in batch:
                        followup_batch['masked_spectra_a'] = batch['masked_spectra_a'][sample_idx:sample_idx+1].to(device)
                    if 'masked_spectra_b' in batch:
                        followup_batch['masked_spectra_b'] = batch['masked_spectra_b'][sample_idx:sample_idx+1].to(device)
                    if 'star_a_feature_indices' in batch:
                        followup_batch['star_a_feature_indices'] = batch['star_a_feature_indices'][sample_idx:sample_idx+1].to(device)
                    if 'star_b_feature_indices' in batch:
                        followup_batch['star_b_feature_indices'] = batch['star_b_feature_indices'][sample_idx:sample_idx+1].to(device)

                    # Mode information
                    if 'mode' in batch:
                        followup_batch['mode'] = [batch['mode'][sample_idx]] if isinstance(batch['mode'], list) else batch['mode'][sample_idx:sample_idx+1]
                    if 'mode_mask_comparative' in batch:
                        followup_batch['mode_mask_comparative'] = batch['mode_mask_comparative'][sample_idx:sample_idx+1]

                    # Attention mask
                    if 'attention_mask' in batch:
                        # Create attention mask for the conversation
                        followup_batch['attention_mask'] = torch.ones(1, len(conv_tokens), dtype=torch.long).to(device)

                    # Generate followup response
                    followup_response, _, _, _ = model.generate_response_from_batch(
                        batch_data=followup_batch,
                        batch_idx=0,  # We've already sliced to a single sample
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    sample_data['followup_qa'].append({
                        'question': followup_q,
                        'type': spec.get('type', 'unknown'),
                        'expected_answer': spec.get('expected_answer', ''),
                        'generated_response': followup_response,
                        'source': 'long_json' if spec['type'] == 'description_from_long_json' else 'stellar_template'
                    })

                exp_results['samples'].append(sample_data)
                samples_processed += 1
                print("samples processed: ", samples_processed)

        all_results.append(exp_results)

        # Clean up to free memory
        # Explicitly delete LLM components if they exist
        target_model = model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
        if hasattr(target_model, 'llm'):
            del target_model.llm
        if hasattr(target_model, 'embedding'):
            del target_model.embedding

        # Delete model and related objects
        del model
        del target_model
        del test_loader
        del state_dict
        del cleaned_state_dict

        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(f"Memory cleaned up after {exp_name}")

    # Save all results
    output_path = ablation_dir / output_filename
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Saved followup comparison results to: {output_path}")
    print(f"Total experiments processed: {len(all_results)}")
    print(f"{'='*80}")


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess ablation study results")
    parser.add_argument("--ablation_dir", type=str, required=True,
                       help="Directory containing ablation results")
    parser.add_argument("--mode", type=str, choices=["loss", "followup", "both", "analyze"], default="loss",
                       help="Analysis mode: 'loss' for loss plots, 'followup' for followup comparison, "
                            "'analyze' to analyze existing followup JSON, 'both' for loss+followup")

    # Loss plotting args
    parser.add_argument("--train_window", type=int, default=50,
                       help="Window size for smoothing training loss (default: 50)")
    parser.add_argument("--val_window", type=int, default=10,
                       help="Window size for smoothing validation loss (default: 10)")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="Optional suffix for output filenames")

    # Followup comparison args
    parser.add_argument("--max_samples", type=int, default=5,
                       help="Maximum number of test samples for followup comparison (default: 50)")
    parser.add_argument("--num_followups", type=int, default=2,
                       help="Number of followup questions per sample (default: 2)")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum tokens to generate per response (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling parameter (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--followup_output", type=str, default="followup_comparison.json",
                       help="Output filename for followup comparison (default: followup_comparison.json)")

    # Analysis args
    parser.add_argument("--analysis_input", type=str, default="followup_comparison.json",
                       help="Input JSON file for analysis mode (default: followup_comparison.json)")
    parser.add_argument("--analysis_output_dir", type=str, default=None,
                       help="Output directory for analysis plots (default: same as ablation_dir)")

    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    if not ablation_dir.exists():
        print(f"Error: {ablation_dir} does not exist")
        return

    if args.mode in ["loss", "both"]:
        print("\n" + "="*80)
        print("RUNNING LOSS COMPARISON")
        print("="*80)
        plot_loss_comparison(
            ablation_dir=ablation_dir,
            train_window=args.train_window,
            val_window=args.val_window,
            output_suffix=args.output_suffix,
        )

    if args.mode in ["followup", "both"]:
        print("\n" + "="*80)
        print("RUNNING FOLLOWUP COMPARISON")
        print("="*80)
        compare_followup_responses(
            ablation_dir=ablation_dir,
            max_samples=args.max_samples,
            num_followups=args.num_followups,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            output_filename=args.followup_output,
        )

    if args.mode == "analyze":
        print("\n" + "="*80)
        print("ANALYZING GENERATION RESULTS")
        print("="*80)

        # Import analysis module
        from src.analyze_generation_results import analyze_generation_results

        # Construct full path to JSON file
        json_path = ablation_dir / args.analysis_input
        if not json_path.exists():
            print(f"Error: Analysis input file not found: {json_path}")
            print(f"Please run with --mode followup first to generate the comparison JSON")
            return

        # Determine output directory
        output_dir = args.analysis_output_dir
        if output_dir is None:
            output_dir = ablation_dir / "analysis_plots"

        print(f"Input JSON: {json_path}")
        print(f"Output directory: {output_dir}")
        print()

        # Run analysis
        stats = analyze_generation_results(
            json_path=str(json_path),
            output_dir=str(output_dir),
        )

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
