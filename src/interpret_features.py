#!/usr/bin/env python3
"""
src/interpret_features.py

Standalone script to explore relationships between generated tokens and input spectral features.
Implements:
1. Attention Map Visualization: "Where is the model looking?"
2. Gradient-based Saliency: "Which features caused this output?"

Usage:
    python src/interpret_features.py --checkpoint_path ... --sample_idx 0 --target_token "temperature"
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.follow_up_inference import (
    load_model,
    create_datasets_and_loaders,
    ensure_backend_config,
    load_config_from_checkpoint_dir,
    create_args_from_config
)
from src.tokenizer_adapter import load_tokenizer_adapter
from nn.llm_hf_multi import HuggingFaceMultimodalModel

# =============================================================================
# MONKEY PATCHING FOR ATTENTION EXTRACTION
# =============================================================================

# Global storage for attention maps
LATEST_ATTENTIONS = []

def patched_forward_with_attention(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Monkey-patched forward method for HuggingFaceMultimodalModel
    that forces output_attentions=True and captures them.
    """
    # 1. Prepare embeddings (same as original)
    token_embeddings, cfm_targets = self._prepare_embeddings(batch)
    attention_mask = batch.get('attention_mask', None)
    
    # 2. Force output_attentions=True
    hf_kwargs = {
        "inputs_embeds": token_embeddings,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "output_attentions": True,  # <--- CHANGED
        "use_cache": False,
    }
    hf_kwargs = {k: v for k, v in hf_kwargs.items() if v is not None}
    
    # 3. Call base model
    outputs = self.base_model(**hf_kwargs)
    hidden_states = getattr(outputs, "last_hidden_state", outputs[0])
    
    # 4. CAPTURE ATTENTIONS
    if hasattr(outputs, "attentions"):
        self.latest_attentions = outputs.attentions
    else:
        print("Warning: Model output did not contain 'attentions'.")

    # 5. Logits (same as original)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        lm_head = getattr(self.base_model, "lm_head", None)
        if lm_head is None:
            raise ValueError("HF model output lacks logits and no lm_head is available.")
        logits = lm_head(hidden_states)
    
    logits = logits.float()
    out_dict = {"logits": logits, "h": hidden_states, "attentions": getattr(outputs, "attentions", None)}
    
    return self._add_common_outputs(hidden_states, out_dict, batch, cfm_targets)


def patched_llama_attention_forward(self, x, start_pos, freqs_cis, mask, cache_rows=None, use_cache=True):
    """
    Monkey-patched forward for Llama Attention module to capture scores.
    """
    # ... (Logic copied from llama3.llama.model.Attention.forward) ...
    # DEBUG: Check if we are running
    if not hasattr(self, '_debug_printed'):
        print(f"DEBUG: Patched forward called for Attention module!")
        self._debug_printed = True

    bsz, seqlen, _ = x.shape
    if cache_rows is not None:
        row_indices = [int(r) for r in cache_rows]
    else:
        row_indices = list(range(bsz))
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    from llama3.llama.model import apply_rotary_emb, repeat_kv
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    if use_cache:
        # Simplified cache logic for analysis (we assume training mode / full sequence mostly)
        if self.cache_k is None:
            self.cache_k = torch.zeros((self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim), device=xq.device, dtype=xq.dtype)
            self.cache_v = torch.zeros((self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim), device=xq.device, dtype=xq.dtype)
        
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        for i, row in enumerate(row_indices):
            self.cache_k[row, start_pos : start_pos + seqlen] = xk[i]
            self.cache_v[row, start_pos : start_pos + seqlen] = xv[i]

        keys = torch.stack([self.cache_k[row, : start_pos + seqlen] for row in row_indices], dim=0)
        values = torch.stack([self.cache_v[row, : start_pos + seqlen] for row in row_indices], dim=0)
    else:
        keys = xk
        values = xv

    keys = repeat_kv(keys, self.n_rep)
    values = repeat_kv(values, self.n_rep)

    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None:
        scores = scores + mask

    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    
    # --- CAPTURE HERE ---
    # Store scores in gloabl list or attach to model if we can reference it.
    # Since 'self' is the Attention module layer, we can't easily reach the top model.
    # We'll use a global list and clear it before forward.
    if len(LATEST_ATTENTIONS) < 32: # Avoid memory explosion if looping
         LATEST_ATTENTIONS.append(scores.detach().cpu()) 
    # --------------------

    output = torch.matmul(scores, values)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)


# =============================================================================
# HOOKS
# =============================================================================

class SaliencyHooks:
    def __init__(self):
        self.projected_tokens_grad = None
        self.projected_tokens_val = None
    
    def projector_hook(self, module, inputs, output):
        """
        Hook for the MLP projector. 
        Output shape: (Batch, K_tokens, Embedding_Dim)
        """
        # Retain gradient on the output of the projector (the "concept tokens")
        output.retain_grad()
        self.projected_tokens_val = output
        
        # Register a backward hook on the tensor itself to capture the gradient
        def capture_grad_hook(grad):
            self.projected_tokens_grad = grad
        output.register_hook(capture_grad_hook)

# =============================================================================
# MAIN LOGIC
# =============================================================================

# =============================================================================
# CONCEPTS DEFINITION
# =============================================================================
CONCEPTS = {
    'Temperature': ['temperature', 'Temperature', 'Teff', 'teff', 'hot', 'cool', 'Kelvin', ' K ', 'warm'],
    'Gravity': ['gravity', 'Gravity', 'logg', 'log(g)', 'surface', 'pressure', 'dense', 'compact'],
    'Metallicity': ['metallicity', 'Metallicity', 'Fe/H', '[Fe/H]', 'metal', 'rich', 'poor', 'abundance'],
    'Main Sequence': ['main-sequence', 'Main Sequence', 'dwarf', 'Dwarf', 'V'],
    'Giant': ['giant', 'Giant', 'evolved', 'III', 'radius', 'expanded']
}

def find_concept_indices(input_ids: List[int], decoded_text: str, tokenizer, concept_keywords: List[str]) -> List[int]:
    """
    Finds all token indices that participate in any of the concept keywords.
    This is heuristic: we check if the decoded token forms part of a keyword.
    """
    indices = []
    
    # 1. Simple Token-level check (fast but might miss split subwords)
    for idx, token_id in enumerate(input_ids):
        # Skip special tokens
        if token_id < 0: continue
        
        token_text = tokenizer.decode([token_id])
        # Clean checking
        for kw in concept_keywords:
            if kw.lower() in token_text.lower():
                indices.append(idx)
                break
    
    # 2. Phrase matching (more expensive but accurate for multi-token words)
    # This is tricky without alignment. We'll stick to per-token check + sliding window if needed.
    # For now, let's trust that keywords like "Temperature" are usually single or few tokens 
    # and at least one token will capture the root.
    
    return sorted(list(set(indices)))

# =============================================================================
# MAIN LOGIC
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interpret TalkingLatents Model Features")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of sample in dataset to analyze')
    parser.add_argument('--target_token', type=str, default=None, 
                        help='Specific token/concept to analyze. If None, runs all defined concepts. If "Last", runs last token.')
    parser.add_argument('--output_dir', type=str, default='interpretation_results', help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy args for loading logic
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    
    cli_args = parser.parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)
    
    # 1. Load Configuration
    print(f"Loading configuration from {cli_args.checkpoint_path}...")
    try:
        config = load_config_from_checkpoint_dir(cli_args.checkpoint_path)
    except FileNotFoundError:
        print("Warning: training_config.json not found, using minimal defaults.")
        config = {}
        
    device = torch.device(cli_args.device)
    
    # 2. Create Args Namespace (merging CLI and Config)
    dummy_args = argparse.Namespace()
    dummy_args.batch_size = 1
    dummy_args.predict_features = False 
    args = create_args_from_config(config, dummy_args)
    
    args.batch_size = 1
    args.num_workers = 0
    args.freeze_llm = False 
    args.freeze_spectral = False
    
    # 3. Initialize Fairscale
    try:
        from fairscale.nn.model_parallel.initialize import initialize_model_parallel, model_parallel_is_initialized
        import torch.distributed as dist
        
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group("gloo", rank=0, world_size=1)
            
        if not model_parallel_is_initialized():
            initialize_model_parallel(1)
            print("Initialized Fairscale for model parallel group.")
    except ImportError:
        pass

    # 4. Load Model
    print("Loading model...")
    model = load_model(cli_args.checkpoint_path, args, device)
    
    for param in model.parameters():
        param.requires_grad = True
    
    # 5. Patch the Model for Attention
    patched_count = 0
    if isinstance(model, HuggingFaceMultimodalModel):
        print("Monkey-patching HuggingFaceMultimodalModel.forward...")
        model.forward = patched_forward_with_attention.__get__(model, HuggingFaceMultimodalModel)
        patched_count += 1
    else:
        # Patch Llama
        try:
            from llama3.llama.model import Attention
            print("Monkey-patching Llama Attention.forward...")
            
            # Explicitly traverse base_model layers if possible
            modules_to_patch = []
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'layers'):
                print("Found base_model.layers, iterating...")
                for layer in model.base_model.layers:
                    if hasattr(layer, 'attention'):
                        modules_to_patch.append(layer.attention)
            
            # Fallback to general search if specific path failed
            if not modules_to_patch:
                print("Fallback: Searching all modules for Attention...")
                for name, module in model.named_modules():
                    if isinstance(module, Attention):
                        modules_to_patch.append(module)
            
            for module in modules_to_patch:
                module.forward = patched_llama_attention_forward.__get__(module, Attention)
                patched_count += 1
                
            print(f"✓ Patched {patched_count} Llama Attention modules.")
        except ImportError:
            print("Warning: Could not import Llama Attention.")
            
    if patched_count == 0:
        print("WARNING: No attention modules were patched! Attention maps will be empty.")

    # 5. Load Data
    print("Creating dataloaders...")
    backend_config = ensure_backend_config(args)
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, device, backend_config)
    dataset = val_loader.dataset 
    print(f"Loaded dataset with {len(dataset)} samples. Selecting index {cli_args.sample_idx}.")
    
    sample = dataset[cli_args.sample_idx]
    
    from torch.utils.data.dataloader import default_collate
    batch = default_collate([sample])
    
    # Prepare batch
    if 'masked_spectra' in batch:
        batch['masked_spectra'] = batch['masked_spectra'].to(device).float()
        batch['masked_spectra'].requires_grad_(True)
        
    if 'feature_start_idx' in batch and 'feature_start_indices' not in batch:
        batch['feature_start_indices'] = batch['feature_start_idx']
    
    for k, v in batch.items():
        if k != 'masked_spectra' and torch.is_tensor(v):
            batch[k] = v.to(device)

    # 6. Setup Saliency Hooks
    saliency_hooks = SaliencyHooks()
    if hasattr(model, 'projector'):
        model.projector.register_forward_hook(saliency_hooks.projector_hook)
        print("Registered hook on model.projector")
    else:
        print("Error: Could not find model.projector!")

    # 7. Forward Pass
    print("Running forward pass...")
    
    global LATEST_ATTENTIONS
    LATEST_ATTENTIONS = []
    
    outputs = model(batch)
    logits = outputs['logits'] # (B, Seq, Vocab)
    
    if LATEST_ATTENTIONS:
        model.latest_attentions = LATEST_ATTENTIONS
        print(f"Captured {len(LATEST_ATTENTIONS)} attention maps from forward pass.")
    else:
        print("No attention maps captured during forward pass.")

    # 8. Decode Text
    tokenizer_path = backend_config.tokenizer_path or "Qwen/Qwen2.5-32B-Instruct"
    tokenizer = load_tokenizer_adapter(
        backend=backend_config.tokenizer_backend,
        tokenizer_path=tokenizer_path, 
        trust_remote_code=True
    )
    
    input_ids = batch['input_ids'][0]
    if torch.is_tensor(input_ids):
        input_ids_list = input_ids.tolist()
    else:
        input_ids_list = input_ids
        
    valid_ids = [t for t in input_ids_list if t >= 0]
    decoded_text = tokenizer.decode(valid_ids)
    print(f"\nFull Sequence Text:\n{decoded_text}\n{'='*40}")
    
    # 9. Determine Concepts to Analyze
    concepts_to_run = {}
    if cli_args.target_token:
        if cli_args.target_token.lower() == 'last':
             concepts_to_run['Last Token'] = [len(valid_ids) - 1]
        elif cli_args.target_token in CONCEPTS:
             concepts_to_run[cli_args.target_token] = CONCEPTS[cli_args.target_token]
        else:
             # Treat as custom keyword list (single item)
             concepts_to_run[cli_args.target_token] = [cli_args.target_token]
    else:
        # Run all default concepts
        concepts_to_run = CONCEPTS

    # 10. Run Analysis Loop
    feature_dim_start = int(batch['feature_start_indices'][0].item())
    feature_dim_end = feature_dim_start + args.num_spectral_features
    
    for concept_name, keywords in concepts_to_run.items():
        print(f"\nAnalyzing Concept: {concept_name}")
        
        # Find indices
        if concept_name == 'Last Token':
            indices = keywords
        else:
            indices = find_concept_indices(valid_ids, decoded_text, tokenizer, keywords)
            
        # Shift indices if necessary? 
        # input_ids_list might have -100s. valid_ids has them removed.
        # But logits align with input_ids (full sequence).
        # We need indices into the Logits/Attention sequence.
        # The indices from 'valid_ids' map to positions in the UNPADDED/UNMASKED sequence.
        # We assume input_ids_list is roughly [Features, Tokens, Padding].
        # So 'indices' into valid_ids should map 1:1 to logits positions (ignoring padding at end).
        # Let's verify: logits shape is (B, Seq, V). input_ids shape is (B, Seq).
        # If valid_ids excludes padding, it might shift if padding was in middle (unlikely for causal LM).
        # Padding is usually at end or left.
        # Assumption: valid_idx corresponds to input_id_idx unless there's masking elsewhere.
        
        # Filter indices to ignore the prompt/input if desired, or keep all.
        # usually we care about the Generated part? Or all occurrences?
        # User said "look for all their occurrence".
        
        if not indices:
            print(f"  No tokens found for keywords {keywords}. Skipping.")
            continue
            
        print(f"  Found {len(indices)} occurrences at indices: {indices}")
        
        # A) ATTENTION AGGREGATION
        if hasattr(model, 'latest_attentions') and model.latest_attentions:
            last_layer_attn = model.latest_attentions[-1] # (B, H, Seq, Seq)
            # Average heads
            avg_attn_matrix = last_layer_attn[0].mean(dim=0).detach().cpu().numpy() # (Seq, Seq)
            
            # Collect rows for target indices
            # Be careful with index bounds
            valid_attn_rows = []
            for idx in indices:
                if idx < avg_attn_matrix.shape[0]:
                    valid_attn_rows.append(avg_attn_matrix[idx, :])
            
            if valid_attn_rows:
                mean_attn_row = np.mean(valid_attn_rows, axis=0)
                
                # Extract spectral attention
                spectral_attn = mean_attn_row[feature_dim_start:feature_dim_end]
                
                plt.figure(figsize=(10, 4))
                plt.plot(range(len(spectral_attn)), spectral_attn, marker='o')
                plt.title(f"Avg Attention to Spectral Features\nConcept: {concept_name} (n={len(indices)})")
                plt.xlabel("Spectral Token Index")
                plt.ylabel("Attention Weight")
                plt.grid(True, alpha=0.3)
                safe_name = concept_name.replace(" ", "_").replace("/", "_")
                plt.savefig(os.path.join(cli_args.output_dir, f"attn_{safe_name}.png"))
                plt.close()
                print(f"  Saved attn_{safe_name}.png")

        # B) GRADIENT AGGREGATION
        # We want grad of sum(logits[indices])
        target_log_probs = []
        
        # For each occurrence at position `i`, the prediction comes from `i-1`.
        # So we look at logit at `i-1` for token `input_ids[i]`.
        
        valid_grad_indices = []
        for idx in indices:
            pred_pos = idx - 1
            if pred_pos >= 0 and pred_pos < logits.shape[1]:
                token_id = input_ids_list[idx] # Use original list to get correct ID
                if token_id >= 0:
                     target_log_probs.append(logits[0, pred_pos, token_id])
                     valid_grad_indices.append(idx)
        
        if not target_log_probs:
            print("  No valid positions for gradient computation (indices too early?).")
            continue
            
        # Sum target log probs
        combined_target = torch.stack(target_log_probs).sum()
        
        model.zero_grad()
        combined_target.backward(retain_graph=True) # Retain graph in case loops overlap? Actually separated by .zero_grad() but we only backward once per concept.
        
        # Plot Saliency
        # 1. Projected Tokens
        if saliency_hooks.projected_tokens_grad is not None:
            grad = saliency_hooks.projected_tokens_grad[0].cpu().numpy()
            token_saliency = np.linalg.norm(grad, axis=1)
            
            plt.figure(figsize=(10, 4))
            plt.plot(token_saliency, color='red', marker='x')
            plt.title(f"Projected Token Saliency\nConcept: {concept_name} (n={len(valid_grad_indices)})")
            plt.xlabel("Spectral Token Index")
            plt.ylabel("Gradient Norm")
            plt.grid(True, alpha=0.3)
            safe_name = concept_name.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(cli_args.output_dir, f"saliency_proj_{safe_name}.png"))
            plt.close()
            
        # 2. Raw Spectra
        if batch['masked_spectra'].grad is not None:
            raw_grad = batch['masked_spectra'].grad[0].cpu().numpy()
            if raw_grad.ndim > 1:
               raw_saliency = np.sum(np.abs(raw_grad), axis=0)
            else:
               raw_saliency = np.abs(raw_grad)
            
            # Reset grad for next concept?
            # Yes, essential!
            batch['masked_spectra'].grad.zero_()
            if saliency_hooks.projected_tokens_grad is not None:
                 saliency_hooks.projected_tokens_grad.zero_() # Or just let hook overwrite? Hook overwrites if capture_grad_hook logic is simple assign.
                 # Actually hook does `self.projected_tokens_grad = grad`. So it will be overwritten next backward.
                 # But input spectra grad accumulates if not zeroed?
                 # batch['masked_spectra'].grad IS accumulated by autograd. So we must zero it.
            
            # Plot
            raw_input = batch['masked_spectra'][0].detach().cpu().numpy()
            if raw_input.ndim > 1: raw_input = raw_input[0]
            
            fig, ax1 = plt.subplots(figsize=(12, 5))
            color = 'tab:blue'
            ax1.set_xlabel('Bin')
            ax1.set_ylabel('Spectrum', color=color)
            ax1.plot(raw_input, color=color, alpha=0.5)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Saliency', color=color)
            ax2.plot(raw_saliency, color=color, alpha=0.9, lw=1)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title(f"Raw Spectrum Saliency\nConcept: {concept_name}")
            safe_name = concept_name.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(cli_args.output_dir, f"saliency_raw_{safe_name}.png"))
            plt.close()
            print(f"  Saved saliency_{safe_name}.png")

if __name__ == "__main__":
    main()
