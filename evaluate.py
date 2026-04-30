"""
evaluate.py — perplexity eval for Step 1 (baselines) and Step 5 (adaptive configs).

Usage:
    python evaluate.py --mode fp16
    python evaluate.py --mode k6v4
    python evaluate.py --mode custom --layer-config layer_config.json
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from turboquant.compressors_v3 import TurboQuantV3

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
N_LAYERS = 28
RESIDUAL_WINDOW = 128  # fixed across all experiments per the plan
STRIDE = 512
MAX_LENGTH = 1024


def patch_model(model, layer_config: dict):
    """
    Patch every layer listed in layer_config with a TurboQuant forward.
    layer_config: {layer_idx (int): {"key_bits": int, "value_bits": int}}
    Layers not in layer_config are left as fp16.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in layer_config:
            continue

        kb = layer_config[layer_idx]["key_bits"]
        vb = layer_config[layer_idx]["value_bits"]
        layer_device = str(next(layer.parameters()).device)

        compressor = TurboQuantV3(
            head_dim=layer.self_attn.head_dim,
            key_bits=kb,
            value_bits=vb,
            residual_window=RESIDUAL_WINDOW,
            layer_idx=layer_idx,
            n_layers=N_LAYERS,
            protected_layers=0,  # uniform — no automatic protection
            seed=42,
            device=layer_device,
        )

        layer.self_attn.forward = _make_tq_forward(layer.self_attn, compressor)


def _make_tq_forward(attn, compressor):
    def tq_forward(
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
        key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        compressed_k, compressed_v = compressor.compress_kv(key_states, value_states)
        key_states, value_states = compressor.decompress_kv(compressed_k, compressed_v)
        key_states = key_states.to(hidden_states.dtype)
        value_states = value_states.to(hidden_states.dtype)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if attn.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=attn.scaling,
            sliding_window=attn.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    return tq_forward


def compute_perplexity(model, tokenizer, text, device):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, STRIDE):
        end = min(begin + MAX_LENGTH, seq_len)
        trg_len = end - prev_end

        chunk = input_ids[:, begin:end].to(device)
        labels = chunk.clone()
        labels[:, :-trg_len] = -100

        with torch.no_grad():
            out = model(chunk, labels=labels)

        nlls.append(out.loss.item())
        prev_end = end

        if begin % 10000 == 0:
            print(f"  tokens {begin}/{seq_len}  running ppl={torch.exp(torch.tensor(nlls).mean()):.4f}")

        if end == seq_len:
            break

    return torch.exp(torch.tensor(nlls).mean()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp16", "k6v4", "custom"], required=True)
    parser.add_argument("--layer-config", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "custom" and args.layer_config is None:
        parser.error("--mode custom requires --layer-config")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {MODEL_NAME} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    if args.mode == "k6v4":
        layer_config = {i: {"key_bits": 6, "value_bits": 4} for i in range(N_LAYERS)}
        print("Patching all layers: K6/V4 ...")
        patch_model(model, layer_config)

    elif args.mode == "custom":
        with open(args.layer_config) as f:
            raw = json.load(f)
        layer_config = {int(k): v for k, v in raw.items()}
        print(f"Patching layers from {args.layer_config} ...")
        patch_model(model, layer_config)

    print("Loading WikiText-2 test set ...")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    print(f"Evaluating perplexity (stride={STRIDE}, max_length={MAX_LENGTH}) ...")
    ppl = compute_perplexity(model, tokenizer, text, device)

    label = args.mode if args.mode != "custom" else args.layer_config
    print(f"\nppl_{label} = {ppl:.4f}")


if __name__ == "__main__":
    main()
