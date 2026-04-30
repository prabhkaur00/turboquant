"""
turbo_gsm8k_analysis.py — GSM8K-based KV-cache quantization analysis for TurboQuant.

Produces three plots (see ANALYSIS.md for interpretation):
  1. Layer-wise relative error — Sensitivity Map
  2. Per-config attention KL-divergence from FP16 — Logic Check
  3. Bit-rate vs. GSM8K accuracy — Efficiency Pareto

Local usage:
    python turbo_gsm8k_analysis.py
    python turbo_gsm8k_analysis.py --configs k4v2 k2v4 k4v4 --num-samples 50
    python turbo_gsm8k_analysis.py --model Qwen/Qwen2.5-7B-Instruct --num-samples 100

Google Colab quick-start:
    # Cell 1 — install
    !pip install torch transformers accelerate datasets scipy matplotlib
    !git clone <repo-url> turboquant && cd turboquant && pip install -e . -q

    # Cell 2 — run
    %cd turboquant
    !python turbo_gsm8k_analysis.py --num-samples 50 --num-calib 10 --output-dir /content/plots
"""

import argparse
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; works in Colab and remote servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.compressors_v3 import TurboQuantV3

# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

ALL_CONFIGS = {
    "fp16": {"key_bits": 16, "value_bits": 16, "avg_bits": 16.0},
    "k8v4": {"key_bits":  8, "value_bits":  4, "avg_bits":  6.0},
    "k4v8": {"key_bits":  4, "value_bits":  8, "avg_bits":  6.0},
    "k4v4": {"key_bits":  4, "value_bits":  4, "avg_bits":  4.0},
    "k4v2": {"key_bits":  4, "value_bits":  2, "avg_bits":  3.0},
    "k2v4": {"key_bits":  2, "value_bits":  4, "avg_bits":  3.0},
}

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_RESIDUAL_WINDOW = 128
# Cap sequence length used for KL-div computation to control peak memory.
_KL_SEQ_CAP = 256


# ---------------------------------------------------------------------------
# Qwen2 attention patching
# ---------------------------------------------------------------------------

def _get_qwen2_helpers():
    """Import Qwen2-specific attention helpers. Raises a clear error if unavailable."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        return apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS
    except ImportError as e:
        sys.exit(
            f"Could not import Qwen2 attention helpers: {e}\n"
            "This script requires transformers >= 4.40 with Qwen2 support.\n"
            "Run: pip install -U transformers"
        )


def make_patched_forward(attn, compressor, stats_bucket):
    """
    Return a patched self-attention forward that:
      1. Projects Q/K/V and applies RoPE as normal.
      2. Compresses K/V with TurboQuantV3 and decompresses.
      3. If stats_bucket is not None, records per-call layer error and attention KL-div.
      4. Runs normal attention and output projection.           

    stats_bucket: dict with lists  {"layer_error_k", "layer_error_v", "attn_kl"}
                  or None to disable recording (used during accuracy eval).
    """
    apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS = _get_qwen2_helpers()

    def patched_forward(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Save unquantized tensors for stat computation
        k_fp16 = key_states.detach()
        v_fp16 = value_states.detach()

        # Quantize and reconstruct
        compressed_k, compressed_v = compressor.compress_kv(key_states, value_states)
        key_hat, value_hat = compressor.decompress_kv(compressed_k, compressed_v)
        key_hat   = key_hat.to(hidden_states.dtype)
        value_hat = value_hat.to(hidden_states.dtype)

        # Record stats (calibration pass only)
        if stats_bucket is not None:
            with torch.no_grad():
                # Relative Frobenius error
                e_k = (torch.norm(key_hat   - k_fp16) / (torch.norm(k_fp16) + 1e-8)).item()
                e_v = (torch.norm(value_hat - v_fp16) / (torch.norm(v_fp16) + 1e-8)).item()
                stats_bucket["layer_error_k"].append(e_k)
                stats_bucket["layer_error_v"].append(e_v)

                # Attention KL-divergence: KL(P_fp16 || P_quant)
                # Only compute for sequences longer than a trivial length
                S = k_fp16.shape[2]
                if S >= 4:
                    T  = min(S, _KL_SEQ_CAP)
                    B, H, _, D = k_fp16.shape
                    # Flatten batch and heads: (B*H, T, D)
                    q  = query_states[:, :, :T, :].reshape(B * H, T, D).float()
                    kf = k_fp16[:, :, :T, :].reshape(B * H, T, D).float()
                    kq = key_hat[:, :, :T, :].reshape(B * H, T, D).float()
                    scale = math.sqrt(D)

                    p_fp16  = F.softmax(torch.bmm(q, kf.transpose(1, 2)) / scale, dim=-1)
                    p_quant = F.softmax(torch.bmm(q, kq.transpose(1, 2)) / scale, dim=-1)

                    eps = 1e-8
                    kl  = (p_fp16 * ((p_fp16 + eps).log() - (p_quant + eps).log())).sum(-1).mean().item()
                    stats_bucket["attn_kl"].append(kl)

        # Accumulate into HF's KV cache if present
        key_out, val_out = key_hat, value_hat
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_out, val_out = past_key_values.update(
                key_out, val_out, attn.layer_idx, cache_kwargs
            )

        attn_iface = eager_attention_forward
        if attn.config._attn_implementation != "eager":
            attn_iface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

        attn_output, attn_weights = attn_iface(
            attn, query_states, key_out, val_out, attention_mask,
            dropout=0.0, scaling=attn.scaling,
            sliding_window=getattr(attn, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    return patched_forward


def save_originals(model):
    return {i: layer.self_attn.forward for i, layer in enumerate(model.model.layers)}


def patch_model(model, key_bits, value_bits, n_layers, residual_window, stats):
    """
    Patch all self-attention layers with TurboQuantV3.
    stats: list of n_layers dicts, or None to disable stat recording.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        device = str(next(layer.parameters()).device)
        compressor = TurboQuantV3(
            head_dim=layer.self_attn.head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            residual_window=residual_window,
            layer_idx=layer_idx,
            n_layers=n_layers,
            protected_layers=0,
            seed=42,
            device=device,
        )
        bucket = stats[layer_idx] if stats is not None else None
        layer.self_attn.forward = make_patched_forward(layer.self_attn, compressor, bucket)


def unpatch_model(model, originals):
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.forward = originals[i]


# ---------------------------------------------------------------------------
# Calibration (layer stats collection — single prefill per example)
# ---------------------------------------------------------------------------

def run_diagnostic_pass(model, tokenizer, dataset, num_examples, device):
    """
    Run a single prefill forward pass per example to collect per-layer stats.

    This is NOT calibration in the ML sense — the quantizer (rotation matrix +
    Lloyd-Max codebook) is fixed at init time from a random seed, with no
    dependence on real data. TurboQuant's random rotation guarantees the input
    to the codebook is approximately Gaussian regardless of the layer or model,
    so the codebook never needs to be adapted.

    This pass purely *measures* how much reconstruction error the fixed quantizer
    produces on realistic activations, for the diagnostic plots. Nothing changes.
    Uses use_cache=False to get a full-sequence prefill (not token-by-token decode).
    """
    model.eval()
    for i, example in enumerate(dataset):
        if i >= num_examples:
            break
        prompt = _build_prompt(example["question"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            model(**inputs, use_cache=False)
        if (i + 1) % 5 == 0:
            print(f"  diagnostic [{i+1}/{num_examples}]")


# ---------------------------------------------------------------------------
# GSM8K evaluation
# ---------------------------------------------------------------------------

def _build_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Write your final numeric answer after '####'.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def _extract_answer(text: str):
    # Prefer the #### pattern (standard GSM8K format)
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fall back to last number in the text
    nums = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def evaluate_gsm8k(model, tokenizer, dataset, num_samples, device, max_new_tokens=256):
    model.eval()
    correct = 0
    total   = 0
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        prompt = _build_prompt(example["question"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        pred  = _extract_answer(generated)
        truth = _extract_answer(example["answer"])
        if pred is not None and truth is not None and pred == truth:
            correct += 1
        total += 1
        if total % 10 == 0:
            print(f"  [{total}/{num_samples}]  acc = {correct/total:.3f}")
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_layer_error(layer_stats_by_config, n_layers, output_dir):
    """Plot 1 — per-layer K and V relative reconstruction error."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors
    color_idx = 0

    for cfg_name, stats in layer_stats_by_config.items():
        if cfg_name == "fp16" or stats is None:
            continue
        c = colors[color_idx % len(colors)]
        color_idx += 1
        for ax, key, ylabel in zip(
            axes,
            ["layer_error_k", "layer_error_v"],
            ["Key Relative Error  (e_k)", "Value Relative Error  (e_v)"],
        ):
            errors = [
                float(np.mean(stats[l][key])) if stats[l][key] else 0.0
                for l in range(n_layers)
            ]
            ax.plot(range(n_layers), errors, marker="o", markersize=3,
                    linewidth=1.5, label=cfg_name, color=c)

    for ax, title in zip(axes, ["Key (e_k) — Sensitivity Map", "Value (e_v) — Sensitivity Map"]):
        ax.set_xlabel("Layer Index", fontsize=11)
        ax.set_ylabel("Relative Error", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "1_layer_wise_error.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_kl_divergence(kl_by_config, output_dir):
    """Plot 2 — mean attention KL-divergence per config (bar chart)."""
    configs = list(kl_by_config.keys())
    values  = [kl_by_config[c] for c in configs]
    colors  = plt.cm.tab10.colors[:len(configs)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(configs, values, color=colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=9)
    ax.set_xlabel("Quantized Config", fontsize=11)
    ax.set_ylabel("Mean KL-Divergence from FP16 Attention", fontsize=11)
    ax.set_title("Attention KL-Divergence — Logic Check", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "2_attention_kl_divergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_pareto(accuracy_by_config, output_dir):
    """Plot 3 — GSM8K accuracy vs. average bits per KV element (scatter)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = plt.cm.tab10.colors

    for i, (cfg_name, acc) in enumerate(accuracy_by_config.items()):
        avg_bits = ALL_CONFIGS[cfg_name]["avg_bits"]
        ax.scatter(avg_bits, acc * 100, s=140, zorder=5,
                   color=colors[i % len(colors)], label=cfg_name, edgecolors="black", linewidths=0.6)
        ax.annotate(cfg_name, (avg_bits, acc * 100),
                    textcoords="offset points", xytext=(7, 5), fontsize=10)

    ax.set_xlabel("Average Bits per KV Element", fontsize=11)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax.set_title("Bit-Rate vs. Accuracy — Efficiency Pareto", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "3_bitrate_vs_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate TurboQuant KV configs on GSM8K and produce three analysis plots. "
            "See ANALYSIS.md for interpretation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path.",
    )
    p.add_argument(
        "--configs", nargs="+",
        default=list(ALL_CONFIGS.keys()),
        choices=list(ALL_CONFIGS.keys()),
        metavar="CONFIG",
        help=f"KV configs to evaluate. Choices: {list(ALL_CONFIGS.keys())}",
    )
    p.add_argument(
        "--num-samples", type=int, default=200,
        help="Number of GSM8K test examples for accuracy evaluation.",
    )
    p.add_argument(
        "--num-calib", type=int, default=20,
        help="Examples from the train split for the diagnostic pass (layer error + KL-div measurement). Does not affect the quantizer.",
    )
    p.add_argument(
        "--residual-window", type=int, default=DEFAULT_RESIDUAL_WINDOW,
        help="Recent tokens kept in FP16 per layer (residual window).",
    )
    p.add_argument(
        "--output-dir", default="plots",
        help="Directory to write plots and results.json.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()

    n_layers = len(model.model.layers)
    model_type = getattr(model.config, "model_type", "unknown")
    print(f"  architecture: {model_type}, layers: {n_layers}")

    if model_type != "qwen2":
        print(
            f"Warning: model_type is '{model_type}', not 'qwen2'. "
            "The attention patcher imports Qwen2-specific helpers and may fail for other architectures."
        )

    print("Loading GSM8K splits ...")
    gsm8k_test  = load_dataset("gsm8k", "main", split="test")   # accuracy eval only
    gsm8k_train = load_dataset("gsm8k", "main", split="train")  # calibration only — never scored

    active_configs = {k: ALL_CONFIGS[k] for k in args.configs}
    accuracy_by_config    = {}
    layer_stats_by_config = {}
    kl_by_config          = {}

    for cfg_name, cfg in active_configs.items():
        kb, vb = cfg["key_bits"], cfg["value_bits"]
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name}  (key={kb}b  value={vb}b  avg={cfg['avg_bits']}b)")
        print(f"{'='*60}")

        layer_stats = [
            {"layer_error_k": [], "layer_error_v": [], "attn_kl": []}
            for _ in range(n_layers)
        ]

        if cfg_name != "fp16":
            # ── Pass 1: diagnostic pass to measure layer stats ────────────
            print(f"  Diagnostic pass ({args.num_calib} examples, train split) ...")
            orig = save_originals(model)
            patch_model(model, kb, vb, n_layers, args.residual_window, stats=layer_stats)
            run_diagnostic_pass(model, tokenizer, gsm8k_train, args.num_calib, device)
            unpatch_model(model, orig)

            # ── Pass 2: accuracy eval (no stats overhead) ──────────────────
            print(f"  Accuracy eval ({args.num_samples} examples) ...")
            orig = save_originals(model)
            patch_model(model, kb, vb, n_layers, args.residual_window, stats=None)
            acc = evaluate_gsm8k(model, tokenizer, gsm8k_test, args.num_samples, device)
            unpatch_model(model, orig)

            # Aggregate KL div: mean over all layers and calibration examples
            all_kl = [v for l in range(n_layers) for v in layer_stats[l]["attn_kl"]]
            kl_by_config[cfg_name] = float(np.mean(all_kl)) if all_kl else 0.0
            print(f"  Mean attn KL-div : {kl_by_config[cfg_name]:.5f}")
        else:
            # FP16 baseline — no patching needed
            print(f"  Accuracy eval ({args.num_samples} examples) ...")
            acc = evaluate_gsm8k(model, tokenizer, gsm8k_test, args.num_samples, device)
            layer_stats = None  # no quantization error for fp16

        accuracy_by_config[cfg_name]    = acc
        layer_stats_by_config[cfg_name] = layer_stats
        print(f"  GSM8K accuracy   : {acc:.3f}  ({acc*100:.1f}%)")

    # ── Save raw results ──────────────────────────────────────────────────────
    results = {
        "model":          args.model,
        "num_samples":    args.num_samples,
        "num_calib":      args.num_calib,
        "residual_window": args.residual_window,
        "accuracy":       accuracy_by_config,
        "kl_divergence":  kl_by_config,
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_layer_error(layer_stats_by_config, n_layers, args.output_dir)
    if kl_by_config:
        plot_kl_divergence(kl_by_config, args.output_dir)
    else:
        print("Skipping KL-div plot (no quantized configs selected).")
    plot_pareto(accuracy_by_config, args.output_dir)

    print(f"\nAll outputs written to ./{args.output_dir}/")


if __name__ == "__main__":
    main()
