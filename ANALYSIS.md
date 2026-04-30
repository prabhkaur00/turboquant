# TurboQuant KV-Cache Sensitivity Analysis

Three diagnostic plots that characterize how asymmetric K/V bit-width choices affect model quality — from where error lives in the network all the way to real GSM8K reasoning accuracy.

---

## The Three Analyses

### 1. Layer-wise Error — Sensitivity Map

**Plot:** Line chart, one line per quantization config  
**X-axis:** Layer index (0 → N-1)  
**Y-axis:** Relative reconstruction error

```
e_k[l] = ||K̂_l − K_l||_F  /  ||K_l||_F
e_v[l] = ||V̂_l − V_l||_F  /  ||V_l||_F
```

This map shows *where* quantization error concentrates in the network. Middle transformer layers typically spike because they encode the highest-level semantic relationships and carry the most discriminative attention patterns.

**What to look for:**
- Configs that spike sharply in middle layers will tend to fail on multi-step reasoning even if their average error looks acceptable
- Comparing `K4V2` vs `K2V4` at the same 3-bit average shows which tensor type drives the middle-layer spike
- A flat error curve across layers is a strong signal of stable quantization

---

### 2. Attention KL-Divergence — Logic Check

**Plot:** Grouped bar chart, one bar per quantization config  
**X-axis:** Config label (K8V4, K4V8, K4V4, K4V2, K2V4)  
**Y-axis:** Mean KL-divergence from FP16 attention

```
KL(P_fp16 || P_quant) = Σ P_fp16 · log(P_fp16 / P_quant)
```

Measures how much the model's *attention focus* shifted after quantization. High KL divergence predicts downstream task failure: if the model is attending to different tokens, its reasoning chain is broken regardless of what cosine-similarity metrics say.

**What to look for:**
- Asymmetric configs (more bits on keys) should consistently show lower KL than symmetric configs at the same average bit budget, because keys determine *which* tokens are attended to
- Any config above ~0.01 mean KL typically fails on multi-hop reasoning benchmarks
- K4V2 should dominate K2V4 at 3-bit average, confirming the key/value asymmetry thesis

---

### 3. Bit-Rate vs. Accuracy — Efficiency Pareto

**Plot:** Scatter plot with labeled points  
**X-axis:** Average bits per KV element  
**Y-axis:** GSM8K accuracy (%)

The "money shot." For a given memory budget, which K/V asymmetry gives the best reasoning accuracy? This is the Pareto frontier — points toward the upper-left are strictly better.

| Config | Avg Bits | Expected compression vs FP16 |
|--------|----------|-------------------------------|
| FP16   | 16.0     | 1×                            |
| K8V4   | 6.0      | ~2.7×                         |
| K4V8   | 6.0      | ~2.7×                         |
| K4V4   | 4.0      | ~4×                           |
| K4V2   | 3.0      | ~5.3×                         |
| K2V4   | 3.0      | ~5.3×                         |

**Expected finding:** `K4V2` should sit above and to the left of `K2V4` (same memory, higher accuracy), confirming that keys need more precision than values. `K8V4` should dominate `K4V8` at the 6-bit budget for the same reason.

---

## Interpreting the Three Plots Together

The plots tell a unified story:

1. **Where does error live?** (Plot 1) — Middle layers are the bottleneck. Configs that allocate more bits to keys show flatter error curves because key reconstruction quality directly governs which tokens the model attends to.

2. **Does error translate to attention shift?** (Plot 2) — Not always 1-to-1. A config can have moderate reconstruction error but low KL divergence if errors cancel in softmax. But *high KL always means broken reasoning* — this is the gating check.

3. **Does it matter for task performance?** (Plot 3) — The ground truth. GSM8K tests multi-step arithmetic reasoning, the hardest downstream signal for KV compression quality. A model that still extracts exact numeric answers after compression is genuinely preserving its computational graph.

---

## Running the Analysis

### Google Colab (step by step)

Paste each block into its own cell and run in order.

**Cell 1 — Install dependencies**
```python
!pip install -q transformers accelerate datasets scipy matplotlib
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
```

**Cell 2 — Clone the repo and install the package**
```python
!git clone https://github.com/tonbistudio/turboquant-pytorch.git
%cd turboquant-pytorch
!pip install -q -e .
```

**Cell 3 — Quick run (3 configs, 50 samples — ~15 min on T4)**
```python
!python turbo_gsm8k_analysis.py \
    --configs fp16 k4v2 k2v4 \
    --num-samples 50 \
    --num-calib 10 \
    --output-dir /content/plots
```

**Cell 4 — View the plots inline**
```python
from IPython.display import Image, display
import glob

for path in sorted(glob.glob("/content/plots/*.png")):
    print(path)
    display(Image(path))
```

**Cell 5 — Full run (all 6 configs, 200 samples — ~90 min on T4)**
```python
!python turbo_gsm8k_analysis.py \
    --configs fp16 k8v4 k4v8 k4v4 k4v2 k2v4 \
    --num-samples 200 \
    --num-calib 20 \
    --output-dir /content/plots
```

> **Tip:** Colab free tier (T4, 15 GB VRAM) fits Qwen2.5-3B-Instruct in FP16 with room to spare. If you hit OOM, add `--num-samples 25 --num-calib 5` or switch to the 1.5B model: `--model Qwen/Qwen2.5-1.5B-Instruct`.

### Local

```bash
# Full run — all configs, 200 GSM8K samples (default)
python turbo_gsm8k_analysis.py

# Quick test — subset of configs, 50 samples
python turbo_gsm8k_analysis.py --num-samples 50 --configs fp16 k4v2 k2v4 k4v4

# Larger model
python turbo_gsm8k_analysis.py --model Qwen/Qwen2.5-7B-Instruct --num-samples 100
```

### CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model ID or local path |
| `--configs` | all six | Space-separated list of configs to test |
| `--num-samples` | `200` | GSM8K test examples for accuracy eval |
| `--num-calib` | `20` | Calibration examples for layer stats |
| `--residual-window` | `128` | Recent tokens kept in FP16 per layer |
| `--output-dir` | `./plots` | Directory for saved plots and `results.json` |
| `--seed` | `42` | Random seed |

### Outputs

All files are written to `--output-dir`:

```
plots/
  1_layer_wise_error.png       # per-layer K and V relative error (two subplots)
  2_attention_kl_divergence.png  # mean KL divergence per config (bar chart)
  3_bitrate_vs_accuracy.png    # GSM8K accuracy vs. avg bits (scatter)
  results.json                 # raw accuracy and KL numbers
```

---

## Dependencies

```bash
pip install torch transformers accelerate datasets scipy matplotlib
```

For CUDA PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

This script requires the `turboquant` package in the same repo:
```bash
pip install -e .   # from the repo root
```
