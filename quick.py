# run: python phase1_quick.py
import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from turboquant.turboquant import generate_rotation_matrix
from scipy.stats import kurtosis

model = GPT2Model.from_pretrained("gpt2")
tok   = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

text = "The quick brown fox jumps over the lazy dog. " * 20
inputs = tok(text, return_tensors="pt", max_length=128, truncation=True)

with torch.no_grad():
    out = model(**inputs, use_cache=True, output_attentions=False)

cache = out.past_key_values
# DynamicCache (transformers >= 4.38) supports __len__ but not layer subscripting.
# to_legacy_cache() converts it to ((K0,V0), (K1,V1), ...) which works everywhere.
if hasattr(cache, "to_legacy_cache"):
    cache = cache.to_legacy_cache()

n_layers   = len(cache)
n_kv_heads = cache[0][0].shape[1]
head_dim   = cache[0][0].shape[-1]

print(f"GPT-2: {n_layers} layers, {n_kv_heads} heads, head_dim={head_dim}")
print(f"{'Layer':>5} {'Head':>4}  {'kurt_raw':>10}  {'kurt_rot':>10}")

for layer_idx in range(n_layers):
    K = cache[layer_idx][0]  # (1, heads, seq, head_dim)
    Pi = generate_rotation_matrix(head_dim, seed=layer_idx * 1000)

    for h in range(n_kv_heads):
        vecs = K[0, h]  # (seq, head_dim)
        norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vecs_norm = vecs / norms
        vecs_rot  = vecs_norm @ Pi.T  # same op as MSECompressor.compress()

        coords_raw = vecs_norm.reshape(-1).numpy()
        coords_rot = vecs_rot.reshape(-1).numpy()

        k_raw = kurtosis(coords_raw, fisher=True)  # 0 = Gaussian
        k_rot = kurtosis(coords_rot, fisher=True)
        print(f"{layer_idx:>5} {h:>4}  {k_raw:>10.3f}  {k_rot:>10.3f}")