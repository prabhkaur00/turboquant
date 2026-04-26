#!/usr/bin/env python3
"""
TurboQuant Outlier Analysis
Covers all 5 measurements from outlier-analysis.md in one script.

  M1. Coordinate histograms after rotation vs theoretical Gaussian
  M2. Excess kurtosis + KS-test per head (distributional fit)
  M3. Quantization MSE per head
  M4. Attention KL-divergence per head (softmax before/after compression)
  M5. Needle-in-haystack recall at multiple context lengths

Usage:
  python outlier_analysis.py                                    # Qwen2.5-3B (default)
  python outlier_analysis.py --model Qwen/Qwen2.5-7B-Instruct
  python outlier_analysis.py --model meta-llama/Llama-3.1-8B-Instruct
  python outlier_analysis.py --model Qwen/Qwen2.5-3B-Instruct --calib 8 --needle-ctx 1024 2048

Time estimates (4-bit, default settings):
  Qwen2.5-3B  T4:  ~40 min   A100: ~12 min
  Qwen2.5-7B  T4:  ~90 min   A100: ~25 min
  Llama-3.1-8B T4: ~100 min  A100: ~28 min
"""

import argparse
import gc
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turboquant.compressors_v3 import MSECompressor
from turboquant.turboquant import generate_rotation_matrix

# ─────────────────────────────────────────────────────────────────────
# CONFIG  (override via CLI args below)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL    = "Qwen/Qwen2.5-3B-Instruct"
BITS_CONFIGS     = [
    {"key_bits": 8, "value_bits": 8, "label": "K8/V8"},
    {"key_bits": 4, "value_bits": 4, "label": "K4/V4"},
    {"key_bits": 4, "value_bits": 2, "label": "K4/V2"},
    {"key_bits": 3, "value_bits": 2, "label": "K3/V2"},
]
N_CALIB          = 16     # forward passes to collect KV statistics
CALIB_LEN        = 1024   # tokens per calibration passage
N_CLASS          = 8      # prompts for head-type classification
CLASS_LEN        = 256    # tokens per classification prompt (short saves memory)
RETRIEVAL_TOP    = 0.30   # top fraction by attention variance  → retrieval heads
SINK_TOP         = 0.30   # top fraction by attention-to-pos-0 → sink heads
NEEDLE_CONTEXTS  = [1024, 2048, 4096]
NEEDLE_TEXT      = "The secret project code name is AURORA-7749."
NEEDLE_EXPECTED  = "AURORA-7749"
FILLER = (
    "The quarterly financial review meeting covered several topics including budget "
    "allocations for the upcoming fiscal year, departmental spending reports, and projected "
    "revenue streams from various business units. The committee discussed infrastructure "
    "upgrades planned for the western regional offices and noted that maintenance schedules "
    "should be coordinated with the facilities management team. Several action items were "
    "assigned to team leads for follow-up before the next meeting cycle.\n\n"
)
CALIB_TEXTS = [
    "Artificial intelligence has become increasingly integrated into daily life, transforming industries from healthcare to transportation. Machine learning algorithms analyze vast datasets to identify patterns and make predictions, enabling applications such as medical diagnosis, autonomous vehicles, and personalized recommendations. The field continues to advance rapidly, with researchers developing more sophisticated models.",
    "The history of computing spans several decades, beginning with early mechanical calculators and evolving through mainframes, personal computers, and mobile devices. Each generation brought increased processing power, reduced costs, and new capabilities. Smartphones now contain more computing power than room-sized computers of the 1960s.",
    "Climate change represents one of the most significant challenges facing humanity. Rising global temperatures, driven primarily by greenhouse gas emissions, are causing widespread environmental disruption. Scientists have documented increases in extreme weather events, rising sea levels, and disruptions to ecosystems worldwide.",
    "The human brain contains approximately one hundred billion neurons, each forming thousands of connections. These neural networks process sensory information, coordinate movement, store memories, and generate conscious experience. Understanding the brain has been a central goal of neuroscience for over a century.",
    "Economic globalization has transformed patterns of trade, investment, and migration. Supply chains now span multiple continents, with components manufactured in different countries. This integration has reduced costs for consumers while creating new vulnerabilities and distributional challenges.",
    "The discovery of antibiotics transformed medicine, enabling treatment of bacterial infections that had previously been fatal. However, widespread misuse has led to antibiotic-resistant bacteria, posing a growing threat. Researchers are working to develop new antimicrobial agents and treatment strategies.",
    "Quantum mechanics describes the behavior of matter and energy at the smallest scales, revealing a world fundamentally different from everyday experience. Particles can exist in superpositions, become entangled with distant partners, and tunnel through barriers. These phenomena have practical applications in computing and cryptography.",
    "The internet has fundamentally altered how people communicate, access information, and conduct commerce. Originally developed for military and academic use, it has grown into a global network connecting billions of devices. Social media platforms have created new forms of community while raising concerns about privacy and misinformation.",
    "Renewable energy technologies have advanced dramatically, with solar and wind power becoming cost-competitive with fossil fuels in many markets. The transition to clean energy is essential for mitigating climate change, but requires significant investment in grid infrastructure and storage systems.",
    "The field of genetics has been transformed by advances in DNA sequencing, enabling researchers to read the complete genetic code quickly and inexpensively. This has revealed insights into human health and evolutionary history. Gene editing technologies now allow precise modifications to DNA sequences.",
    "Urban planning shapes how cities grow and function, influencing transportation, housing, and quality of life. Effective design balances competing demands for density and space, accessibility and privacy. Cities worldwide are grappling with rapid population growth and the need for sustainable infrastructure.",
    "International trade agreements have shaped economic relationships between countries, reducing tariffs and barriers to exchange of goods and services. These agreements generate economic benefits while displacing workers in affected industries. Negotiating trade policy requires balancing interests of different economic sectors.",
    "Space exploration has expanded humanity's knowledge of the universe and produced numerous technological spinoffs. Robotic spacecraft have visited every planet in the solar system. Human missions have traveled to the Moon and maintained continuous presence on the International Space Station for decades.",
    "The gig economy has created new forms of work and employment relationships, enabling flexible arrangements that benefit some workers while raising concerns about job security and benefits. Digital platforms connect workers with customers in real time, transforming industries from transportation to professional services.",
    "Vaccine development has been one of the most successful public health interventions in history, dramatically reducing the incidence of diseases that once caused widespread suffering. The rapid development of COVID-19 vaccines demonstrated both the capabilities of modern biotechnology and the importance of international scientific cooperation.",
    "The philosophy of mind addresses fundamental questions about consciousness and the relationship between mental states and physical processes. Debates about the nature of subjective experience, the possibility of artificial consciousness, and the basis of personal identity have occupied philosophers for centuries.",
]
CLASS_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What is the capital of France and why is it historically significant?",
    "Describe the process of photosynthesis step by step.",
    "How does the internet work? Explain the key protocols involved.",
    "What are the main causes of climate change?",
    "Explain recursion in programming with an example.",
    "What is quantum computing and how does it differ from classical computing?",
    "Describe the water cycle and its importance to life on Earth.",
]

# ─────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────

def load_model(model_name, use_4bit=True):
    print(f"Loading {model_name} ({'4-bit' if use_4bit else 'fp16'})…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(device_map="auto", dtype=torch.float16, attn_implementation="eager")
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    cfg = model.config
    arch = dict(
        n_layers  = cfg.num_hidden_layers,
        n_kv      = cfg.num_key_value_heads,
        n_q       = cfg.num_attention_heads,
        head_dim  = cfg.hidden_size // cfg.num_attention_heads,
        q_per_kv  = cfg.num_attention_heads // cfg.num_key_value_heads,
    )
    print(f"  layers={arch['n_layers']}, Q-heads={arch['n_q']}, KV-heads={arch['n_kv']}, "
          f"head_dim={arch['head_dim']}, GPU={torch.cuda.memory_allocated()//1024//1024} MB")
    return model, tok, arch


def tokenize(tok, text, max_len, device="cuda"):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    return {k: v.to(device) for k, v in enc.items()}


def get_kv(cache, layer_idx):
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
    return cache[layer_idx][0], cache[layer_idx][1]


def flush():
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────
# STEP 1  Head classification
# ─────────────────────────────────────────────────────────────────────

def classify_heads(model, tok, arch, device="cuda"):
    """
    Run N short prompts with output_attentions=True.
    Score each KV head for: attention to position 0 (sink), variance of
    attention distribution (retrieval). Label top fractions accordingly.
    Falls back to 'local' for all heads if output_attentions fails (e.g. OOM).
    """
    print("\n[1/5] Head classification…", flush=True)
    L, Hkv, q_per_kv = arch["n_layers"], arch["n_kv"], arch["q_per_kv"]

    sink_acc  = np.zeros((L, Hkv))
    var_acc   = np.zeros((L, Hkv))
    n_done    = 0

    for prompt in CLASS_PROMPTS[:N_CLASS]:
        inputs = tokenize(tok, prompt, CLASS_LEN, device)
        try:
            with torch.no_grad():
                out = model(**inputs, output_attentions=True)
            attentions = out.attentions  # tuple[layer] of (1, n_q, S, S)
        except Exception as e:
            print(f"  output_attentions failed ({e}) — defaulting all heads to 'local'")
            labels = {(l, h): "local" for l in range(L) for h in range(Hkv)}
            return labels, sink_acc, var_acc

        if attentions is None:
            print("  output_attentions returned None — defaulting all heads to 'local'")
            labels = {(l, h): "local" for l in range(L) for h in range(Hkv)}
            return labels, sink_acc, var_acc

        for li, attn in enumerate(attentions):
            attn = attn[0].float()  # (n_q, S, S)
            for kv_h in range(Hkv):
                qs = attn[kv_h * q_per_kv : (kv_h + 1) * q_per_kv].mean(0)  # (S, S)
                sink_acc[li, kv_h] += qs[:, 0].mean().item()
                var_acc[li, kv_h]  += qs.var(dim=-1).mean().item()

        del out, attentions
        flush()
        n_done += 1

    sink_acc /= max(n_done, 1)
    var_acc  /= max(n_done, 1)

    sink_thresh = np.percentile(sink_acc, (1 - SINK_TOP) * 100)
    var_thresh  = np.percentile(var_acc,  (1 - RETRIEVAL_TOP) * 100)

    labels = {}
    counts = {"retrieval": 0, "sink": 0, "local": 0}
    for l in range(L):
        for h in range(Hkv):
            if sink_acc[l, h] >= sink_thresh:
                labels[(l, h)] = "sink";      counts["sink"]      += 1
            elif var_acc[l, h] >= var_thresh:
                labels[(l, h)] = "retrieval"; counts["retrieval"] += 1
            else:
                labels[(l, h)] = "local";     counts["local"]     += 1

    print(f"  {counts}")
    return labels, sink_acc, var_acc


# ─────────────────────────────────────────────────────────────────────
# STEPS 2-4  Per-head statistics (kurtosis, MSE, KL-div)
# ─────────────────────────────────────────────────────────────────────

def compute_stats(model, tok, arch, bits_cfgs, head_labels, device="cuda"):
    """
    For each calibration passage:
      - One forward pass to capture past_key_values
      - Per head: rotate coordinates, measure kurtosis + KS-test (M1/M2)
      - Per head × bit config: compress/decompress, measure MSE + KL-div (M3/M4)
    Compressors are created once per (layer, config) and reused across passages.
    """
    print("\n[2-4/5] Per-head statistics…", flush=True)
    L, Hkv, D = arch["n_layers"], arch["n_kv"], arch["head_dim"]
    cfg_labels = [c["label"] for c in bits_cfgs]

    # Accumulators
    kurt_acc = {(l, h): [] for l in range(L) for h in range(Hkv)}
    ksp_acc  = {(l, h): [] for l in range(L) for h in range(Hkv)}
    mse_acc  = {lbl: {(l, h): [] for l in range(L) for h in range(Hkv)} for lbl in cfg_labels}
    kl_acc   = {lbl: {(l, h): [] for l in range(L) for h in range(Hkv)} for lbl in cfg_labels}

    # Build compressors once — seed matches TurboQuantV3 convention
    # key_compressor seed = 42 + layer_idx * 1000
    key_comps = {}   # (layer, cfg_label) -> MSECompressor for keys
    for l in range(L):
        seed_base = 42 + l * 1000
        for cfg in bits_cfgs:
            key_comps[(l, cfg["label"])] = MSECompressor(
                D, cfg["key_bits"], seed=seed_base, device=device
            )

    # Rotation matrices (same as key_compressor.Pi but we surface them for M1/M2)
    Pi = {l: key_comps[(l, cfg_labels[0])].Pi for l in range(L)}

    for pi, text in enumerate(CALIB_TEXTS[:N_CALIB]):
        print(f"  passage {pi+1}/{N_CALIB}…", end=" ", flush=True)
        inputs = tokenize(tok, text, CALIB_LEN, device)

        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        cache = out.past_key_values

        for l in range(L):
            keys, values = get_kv(cache, l)   # (1, Hkv, S, D)
            keys_f   = keys.float()
            values_f = values.float()
            S = keys_f.shape[2]

            for h in range(Hkv):
                kh = keys_f[0, h]              # (S, D)

                # Normalize to unit sphere (mirrors MSECompressor.compress)
                norms = kh.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                kh_n  = kh / norms

                # Rotate
                rotated = kh_n @ Pi[l].T       # (S, D)  — same rotation as compressor
                coords  = rotated.cpu().numpy().flatten()

                # M2: excess kurtosis (0 = Gaussian)
                kurt_acc[(l, h)].append(float(scipy_stats.kurtosis(coords, fisher=True)))

                # M2: KS-test vs standard normal (coordinates are ~N(0, 1/sqrt(D)) so rescale)
                sample = coords * math.sqrt(D)   # rescale to N(0,1) for interpretability
                if len(sample) > 8000:
                    sample = np.random.choice(sample, 8000, replace=False)
                _, pval = scipy_stats.kstest(sample, "norm")
                ksp_acc[(l, h)].append(float(pval))

            # M3 & M4 per bit config
            for cfg in bits_cfgs:
                lbl    = cfg["label"]
                kcomp  = key_comps[(l, lbl)]

                compressed = kcomp.compress(keys_f)
                keys_recon = kcomp.decompress(compressed).float()   # (1, Hkv, S, D)

                for h in range(Hkv):
                    kh = keys_f[0, h]      # (S, D)
                    kr = keys_recon[0, h]  # (S, D)

                    # M3: MSE
                    mse_acc[lbl][(l, h)].append(((kh - kr) ** 2).mean().item())

                    # M4: KL-divergence — use last token as proxy query
                    q      = kh[-1:, :]                              # (1, D)
                    scale  = math.sqrt(D)
                    p_raw  = F.softmax((q @ kh.T / scale).squeeze(0), dim=0)
                    p_comp = F.softmax((q @ kr.T / scale).squeeze(0), dim=0)
                    kl = F.kl_div((p_comp + 1e-10).log(), p_raw, reduction="sum").item()
                    kl_acc[lbl][(l, h)].append(kl)

        del out, cache
        flush()
        print("✓")

    def avg(d):
        return {k: float(np.mean(v)) if v else 0.0 for k, v in d.items()}

    return {
        "kurtosis": avg(kurt_acc),
        "ks_pval":  avg(ksp_acc),
        "mse":      {lbl: avg(mse_acc[lbl]) for lbl in cfg_labels},
        "kl":       {lbl: avg(kl_acc[lbl])  for lbl in cfg_labels},
    }


# ─────────────────────────────────────────────────────────────────────
# STEP 5  Needle-in-haystack recall
# ─────────────────────────────────────────────────────────────────────

class _V3Cache(DynamicCache):
    """Minimal V3Cache — inline so this script is self-contained."""
    def __init__(self, key_bits, value_bits, residual_window, n_layers):
        super().__init__()
        self.key_bits, self.value_bits = key_bits, value_bits
        self.rw, self.n_layers = residual_window, n_layers
        self._comps = {}; self._ck = {}; self._cv = {}
        self._rbk = {}; self._rbv = {}; self._total = {}

    def _comp(self, li, D, dev):
        if li not in self._comps:
            seed = 42 + li * 1000
            from turboquant.compressors_v3 import TurboQuantV3
            self._comps[li] = TurboQuantV3(D, self.key_bits, self.value_bits,
                                           0, li, self.n_layers, 0, 8, 42, str(dev))
        return self._comps[li]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        B, H, S_new, D = key_states.shape
        dev = key_states.device
        comp = self._comp(layer_idx, D, dev)
        if layer_idx not in self._ck:
            self._ck[layer_idx] = []; self._cv[layer_idx] = []
            self._rbk[layer_idx] = []; self._rbv[layer_idx] = []
            self._total[layer_idx] = 0
        self._total[layer_idx] += S_new
        self._rbk[layer_idx].append(key_states)
        self._rbv[layer_idx].append(value_states)
        rk = torch.cat(self._rbk[layer_idx], dim=2)
        rv = torch.cat(self._rbv[layer_idx], dim=2)
        rw = self.rw
        if rw == 0:
            ck, cv = comp.compress_kv(rk, rv)
            self._ck[layer_idx].append(ck); self._cv[layer_idx].append(cv)
            self._rbk[layer_idx] = []; self._rbv[layer_idx] = []
        elif rk.shape[2] > rw:
            ov = rk.shape[2] - rw
            ck, cv = comp.compress_kv(rk[:, :, :ov, :], rv[:, :, :ov, :])
            self._ck[layer_idx].append(ck); self._cv[layer_idx].append(cv)
            self._rbk[layer_idx] = [rk[:, :, ov:, :]]; self._rbv[layer_idx] = [rv[:, :, ov:, :]]
        parts_k, parts_v = [], []
        for ck, cv in zip(self._ck[layer_idx], self._cv[layer_idx]):
            dk, dv = comp.decompress_kv(ck, cv)
            parts_k.append(dk.to(key_states.dtype))
            parts_v.append(dv.to(value_states.dtype))
        if self._rbk[layer_idx]:
            parts_k.append(torch.cat(self._rbk[layer_idx], dim=2))
            parts_v.append(torch.cat(self._rbv[layer_idx], dim=2))
        full_k = torch.cat(parts_k, dim=2)
        full_v = torch.cat(parts_v, dim=2)
        while len(self.layers) <= layer_idx:
            from transformers.cache_utils import DynamicLayer
            self.layers.append(DynamicLayer())
        return full_k, full_v

    def get_seq_length(self, layer_idx=0):
        return self._total.get(layer_idx, 0)


def _needle_prompt(tok, target_tokens, model_family):
    filler_len = len(tok.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = n_reps // 2
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE_TEXT}\n--- End ---\n\n")
        parts.append(FILLER)
    hay = "".join(parts)
    q = "What is the secret project code name? Answer with just the code name."
    if model_family == "llama":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{hay}\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{hay}\n{q}<|im_end|>\n<|im_start|>assistant\n")


def run_needle_recall(model, tok, arch, bits_cfgs, needle_ctxs, model_family, device="cuda"):
    print("\n[5/5] Needle recall…", flush=True)
    results = {}
    all_configs = [{"fp16": True, "label": "FP16"}] + bits_cfgs

    for ctx in needle_ctxs:
        results[ctx] = {}
        prompt = _needle_prompt(tok, ctx, model_family)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=ctx + 512)
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        for cfg in all_configs:
            lbl = cfg["label"]
            cache = None if cfg.get("fp16") else _V3Cache(
                cfg["key_bits"], cfg["value_bits"], cfg.get("residual_window", 128), arch["n_layers"]
            )
            print(f"  ctx={ctx} [{lbl}]…", end=" ", flush=True)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=mask, max_new_tokens=32,
                                     do_sample=False, past_key_values=cache, use_cache=True)
            resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            found = NEEDLE_EXPECTED.lower() in resp.lower()
            results[ctx][lbl] = found
            print("FOUND" if found else "MISS")
            del out
            flush()

    return results


# ─────────────────────────────────────────────────────────────────────
# PLOTS & SUMMARY
# ─────────────────────────────────────────────────────────────────────

COLORS = {"retrieval": "#e74c3c", "sink": "#3498db", "local": "#2ecc71"}
HEAD_TYPES = ["retrieval", "sink", "local"]


def _by_type(d, head_labels, t):
    return [v for (l, h), v in d.items() if head_labels.get((l, h)) == t]


def _bar_by_type(ax, data_by_type, active_types, ylabel, title=""):
    """Bar chart with error bars — works for any number of non-empty groups."""
    means  = [np.mean(data_by_type[t]) for t in active_types]
    stds   = [np.std(data_by_type[t])  for t in active_types]
    xs     = range(len(active_types))
    bars   = ax.bar(xs, means, yerr=stds, capsize=4,
                    color=[COLORS[t] for t in active_types], alpha=0.8)
    ax.set_xticks(list(xs)); ax.set_xticklabels(active_types)
    ax.set_ylabel(ylabel); ax.set_title(title)


def plot_all(hstats, head_labels, arch, bits_cfgs, needle_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    L, Hkv = arch["n_layers"], arch["n_kv"]
    cfg_labels = [c["label"] for c in bits_cfgs]

    # Only plot head types that actually have data
    active_types = [t for t in HEAD_TYPES
                    if len(_by_type(hstats["kurtosis"], head_labels, t)) > 0]

    # ── Figure 1: Kurtosis heatmap + bar by head type ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mat = np.array([[hstats["kurtosis"][(l, h)] for h in range(Hkv)] for l in range(L)])
    im = axes[0].imshow(mat, aspect="auto", cmap="RdYlGn_r")
    axes[0].set_xlabel("KV head"); axes[0].set_ylabel("Layer")
    axes[0].set_title("Excess Kurtosis (0 = Gaussian, ↑ = heavier tails)")
    plt.colorbar(im, ax=axes[0])

    data_by_type = {t: _by_type(hstats["kurtosis"], head_labels, t) for t in active_types}
    _bar_by_type(axes[1], data_by_type, active_types,
                 ylabel="Excess Kurtosis (mean ± std)", title="Kurtosis by Head Type")
    axes[1].axhline(0, color="k", linestyle="--", alpha=0.4, label="Gaussian baseline")
    axes[1].legend()
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig1_kurtosis.png", dpi=150); plt.close()

    # ── Figure 2: MSE and KL per head type per bit config ────────────────
    n = len(cfg_labels)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    for ci, lbl in enumerate(cfg_labels):
        for ri, (key, ylabel) in enumerate([("mse", "Quantization MSE"), ("kl", "Attention KL Divergence")]):
            ax = axes[ri, ci]
            data_by_type = {t: _by_type(hstats[key][lbl], head_labels, t) for t in active_types}
            active = [t for t in active_types if len(data_by_type[t]) > 0]
            _bar_by_type(ax, data_by_type, active, ylabel=ylabel if ci == 0 else "",
                         title=lbl if ri == 0 else "")
    plt.suptitle("Compression Error by Head Type", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig2_errors_by_head.png", dpi=150); plt.close()

    # ── Figure 3: MSE vs KL correlation (proxy comparison) ──────────────
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ci, lbl in enumerate(cfg_labels):
        ax = axes[0, ci]
        for t in HEAD_TYPES:
            mse_t = _by_type(hstats["mse"][lbl], head_labels, t)
            kl_t  = _by_type(hstats["kl"][lbl],  head_labels, t)
            ax.scatter(mse_t, kl_t, alpha=0.4, color=COLORS[t], label=t, s=12)
        all_mse = list(hstats["mse"][lbl].values())
        all_kl  = list(hstats["kl"][lbl].values())
        if len(all_mse) > 2:
            r, p = scipy_stats.pearsonr(all_mse, all_kl)
            ax.set_title(f"{lbl}  r={r:.3f} (p={p:.3f})")
        ax.set_xlabel("MSE"); ax.set_ylabel("KL Divergence")
        if ci == 0: ax.legend()
    plt.suptitle("MSE vs KL Divergence — are they measuring the same thing?", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig3_proxy_correlation.png", dpi=150); plt.close()

    # ── Figure 4: Needle recall table ───────────────────────────────────
    row_labels = ["FP16"] + cfg_labels
    col_labels  = [str(c) for c in sorted(needle_results.keys())]
    cell_text   = []
    cell_colors = []
    for rl in row_labels:
        row = []; cols = []
        for ctx in sorted(needle_results.keys()):
            found = needle_results[ctx].get(rl)
            row.append("FOUND" if found else "MISS")
            cols.append("#27ae60" if found else "#e74c3c")
        cell_text.append(row); cell_colors.append(cols)

    fig, ax = plt.subplots(figsize=(3 + 2 * len(col_labels), 1 + 0.5 * len(row_labels)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.4, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif c >= 0:
            cell.set_facecolor(cell_colors[r - 1][c])
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Needle-in-Haystack Recall (context length in tokens)", fontsize=12, pad=16)
    plt.tight_layout(); plt.savefig(f"{out_dir}/fig4_needle_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {out_dir}/")


def print_summary(hstats, head_labels, bits_cfgs, needle_results):
    cfg_labels = [c["label"] for c in bits_cfgs]
    counts = {t: sum(1 for v in head_labels.values() if v == t) for t in HEAD_TYPES}
    ctxs   = sorted(needle_results.keys())

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Head types: {counts}\n")

    print(f"  {'':22}  {'kurtosis':>10}  {'MSE retr':>10}  {'MSE sink':>10}  "
          f"{'KL retr':>10}  {'KL sink':>10}  {'needle':>10}")
    print("  " + "-" * 86)

    for lbl in cfg_labels:
        mr = np.mean(_by_type(hstats["mse"][lbl], head_labels, "retrieval") or [0])
        ms = np.mean(_by_type(hstats["mse"][lbl], head_labels, "sink") or [0])
        kr = np.mean(_by_type(hstats["kl"][lbl],  head_labels, "retrieval") or [0])
        ks = np.mean(_by_type(hstats["kl"][lbl],  head_labels, "sink") or [0])
        ku = np.mean(list(hstats["kurtosis"].values()))
        recall = sum(needle_results.get(c, {}).get(lbl, False) for c in ctxs)
        print(f"  {lbl:<22}  {ku:>10.4f}  {mr:>10.6f}  {ms:>10.6f}  "
              f"{kr:>10.6f}  {ks:>10.6f}  {recall}/{len(ctxs)}")

    # FP16 needle
    fp16_recall = sum(needle_results.get(c, {}).get("FP16", False) for c in ctxs)
    print(f"  {'FP16 (baseline)':<22}  {'—':>10}  {'—':>10}  {'—':>10}  "
          f"{'—':>10}  {'—':>10}  {fp16_recall}/{len(ctxs)}")
    print("=" * 90)

    # Kurtosis by head type
    print("\nMean excess kurtosis by head type:")
    for t in HEAD_TYPES:
        vals = _by_type(hstats["kurtosis"], head_labels, t)
        if vals:
            print(f"  {t:<12}: {np.mean(vals):+.4f} ± {np.std(vals):.4f}")

    # KS-test
    low_pval = sum(1 for v in hstats["ks_pval"].values() if v < 0.05)
    total = len(hstats["ks_pval"])
    print(f"\nKS-test: {low_pval}/{total} heads reject Gaussian (p<0.05) — "
          f"{'rotation is helping' if low_pval < total * 0.5 else 'significant non-Gaussianity remains'}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    global N_CALIB, NEEDLE_CONTEXTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--no-4bit",    action="store_true")
    parser.add_argument("--calib",      type=int, default=N_CALIB,
                        help="Number of calibration passages (default 16, min 4)")
    parser.add_argument("--needle-ctx", type=int, nargs="+", default=NEEDLE_CONTEXTS)
    parser.add_argument("--out",        default="outlier_results")
    args = parser.parse_args()

    N_CALIB         = max(4, args.calib)
    NEEDLE_CONTEXTS = args.needle_ctx

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit   = not args.no_4bit and device == "cuda"
    model_fam  = "llama" if "llama" in args.model.lower() else "qwen"
    out_dir    = args.out

    os.makedirs(out_dir, exist_ok=True)

    model, tok, arch = load_model(args.model, use_4bit)

    labels, sink_scores, var_scores = classify_heads(model, tok, arch, device)

    hstats = compute_stats(model, tok, arch, BITS_CONFIGS, labels, device)

    needle = run_needle_recall(model, tok, arch, BITS_CONFIGS, NEEDLE_CONTEXTS, model_fam, device)

    print("\nGenerating plots…", flush=True)
    plot_all(hstats, labels, arch, BITS_CONFIGS, needle, out_dir)
    print_summary(hstats, labels, BITS_CONFIGS, needle)

    # Save raw numbers for further analysis
    with open(f"{out_dir}/raw_stats.json", "w") as f:
        json.dump({
            "model": args.model,
            "arch":  arch,
            "head_labels": {f"{l}_{h}": v for (l, h), v in labels.items()},
            "kurtosis": {f"{l}_{h}": v for (l, h), v in hstats["kurtosis"].items()},
            "ks_pval":  {f"{l}_{h}": v for (l, h), v in hstats["ks_pval"].items()},
            "mse":      {lbl: {f"{l}_{h}": v for (l, h), v in d.items()}
                         for lbl, d in hstats["mse"].items()},
            "kl":       {lbl: {f"{l}_{h}": v for (l, h), v in d.items()}
                         for lbl, d in hstats["kl"].items()},
            "needle": {str(k): v for k, v in needle.items()},
        }, f, indent=2)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
