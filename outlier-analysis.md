# TurboQuant KV Outlier Analysis — Research README

## The Central Question

TurboQuant's rotation step is supposed to make every coordinate of every KV vector
follow the same Beta ≈ Gaussian distribution, justifying a single shared codebook.
The math proves this in expectation, over random R, for a perfectly normalized vector,
in the aggregate.

This project asks: does it actually hold, per head, per layer, in a real model?
And if not — is the deviation large enough to matter for generation quality,
and systematic enough to exploit?

---

## Why This Is Worth Doing

The proof has three hidden assumptions that break in practice:

1. The rotation works in expectation over random R. In deployment, R is fixed.
   For any fixed R, input distributions that are poorly spread can persist.

2. The paper evaluates aggregate metrics (perplexity, needle recall). It never
   looks inside at which heads contribute to quantization error.

3. The community already found empirically that K needs more bits than V even
   after rotation. That means something survives rotation. Nobody has characterized
   what — channel structure, head type, layer position, or norm variance.

Your contribution: characterize *where* rotation fails and *why*, stratified by
head type (retrieval / sink / local), and test whether knowing this lets you
allocate bits more efficiently than uniform K6/V4.

---

## What You Need to Know Before Starting

**Grouped Query Attention (GQA):**
Qwen2.5-7B has 28 query heads but only 8 KV heads. The KV cache stores 8 heads
per layer, not 28. When you extract K and V tensors, you get 8 vectors per token
per layer. When you read attention weights, there are 28 (each query head has its
own weights, but groups of query heads share one KV head). Always be explicit about
which you're working with.

**What TurboQuant actually stores:**
For each token, per KV head, per layer:
- One scalar norm (fp16)
- head_dim centroid indices, bit-packed

Decompression: unpack indices → look up centroids → multiply by norm.
The rotation matrix R is fixed and shared — it is not stored per token.

**Head types (the independent variable for your whole analysis):**
- Retrieval heads: attention pattern changes based on query content. These do
  the "meaningful" retrieval work. High attention score variance across inputs.
- Sink heads: always attend heavily to token 0 (BOS), regardless of content.
  A known LLM artifact. Low attention variance, stereotyped key vectors.
- Local heads: attend primarily to recent tokens. Somewhere between the two.

The hypothesis is that retrieval heads produce more content-sensitive key vectors,
which means more residual structure survives the rotation, which means the shared
codebook fits them worse.

---

## What to Measure

These five measurements form a complete story from "does the math hold"
to "does it matter for real inference":

**1. Coordinate histograms after rotation**
After applying R to a normalized K or V vector, plot the empirical distribution
of coordinate values. Compare to the theoretical Beta/Gaussian curve TurboQuant
assumes. Do this separately per head type. If the distributions match, the math
holds and you have a null result (still publishable). If retrieval heads show
heavier tails or skew, you have your main finding.

**2. Fit to the Beta/Gaussian model**
Quantify how well the empirical coordinate distribution matches the theoretical
one. Use excess kurtosis (0 = Gaussian, higher = heavier tails) and a KS-test
p-value against the theoretical Beta CDF. This gives you a scalar per head that
summarizes "how well does TurboQuant's assumption hold for this head."

**3. Quantization MSE**
Apply quantization and dequantization to the rotated K/V vectors using the
standard TurboQuant codebook. Measure mean squared error between original and
reconstructed vectors, per head. If retrieval heads have worse distributional fit,
they should also have higher MSE. This links the statistical finding to the
actual compression error.

**4. Attention-logit error**
This is the most important measurement and the one most papers skip.
Compute attention logits q @ k.T / sqrt(d) using raw K, then using
quantized-dequantized K. Measure the difference as KL divergence between
the resulting softmax distributions. This matters because softmax is exponential:
small errors in logits get amplified nonlinearly. A head with 5% higher
quantization MSE might have 30% higher attention-logit error if its logits are
in a high-sensitivity regime. Cosine similarity of K vectors (what the current
repo uses) misses this entirely.

**5. End-task quality**
Perplexity on WikiText-2 and needle-in-haystack recall at multiple context lengths.
This is your ground truth. Everything above is a proxy — measurement 4 is only
valuable if it correlates with this better than measurement 3 does. Your proxy
metric comparison is the methodological contribution: you're not just running an
experiment, you're also telling future researchers which proxy to trust.

---

## The Systems Angle (What Makes This Engineering, Not Just Statistics)

For each head, you will have a distributional fit score, a quantization MSE,
and an attention-logit error. Ask: is the variation across heads large enough
to justify paying for per-head codebooks or per-head bit allocation?

Per-head codebooks cost extra metadata: you need to store which codebook each
head uses. The question is whether the accuracy gain justifies that overhead.

Concretely: if retrieval heads have 2x higher MSE than sink heads under the
shared codebook, and you can close that gap by giving retrieval heads +1 bit
while giving sink heads -1 bit to keep the budget constant, you have a practical
deployable recommendation. That is an engineering decision backed by measurement.

If the distributional separation between head types is small (< 10% difference
in MSE), the overhead is not worth it — and that is also a valid, useful finding
for anyone building on TurboQuant.

---

## Evaluation Plan

| What you measure | How | Expected finding |
|---|---|---|
| Coordinate histograms | Plot empirical vs theoretical Beta per head type | Retrieval heads: heavier tails |
| Distributional fit | Excess kurtosis + KS-test per head | Retrieval heads: higher kurtosis |
| Quantization MSE | Quantize + dequantize, measure L2 error per head | Retrieval heads: higher MSE |
| Attention-logit error | KL divergence of softmax(raw K) vs softmax(quant K) | Higher than MSE alone predicts |
| End-task quality | Perplexity + needle recall at 1K / 2K / 4K / 8K | Proxy 4 predicts this better than proxy 3 |
| Proxy metric comparison | Correlate each proxy with perplexity degradation | KL div > cosine sim > MSE as predictors |
| Systems tradeoff | MSE gap between head types vs metadata overhead | Quantify break-even for per-head codebooks |

---

## Minimal Experimental Design

**Model:** Qwen2.5-7B-Instruct (primary). Llama-3.1-8B-Instruct (secondary, for generalization).
Use 7B not 3B — the paper notes quality degrades on very small models and your
findings need to be credible at production-relevant scale. Both fit on A100-80GB.

**Calibration data:** 128 passages from WikiText-103, each ~2K tokens.
Enough for stable channel-level statistics. Use the same 128 passages for all
measurements so results are directly comparable.

**Evaluation data:** WikiText-2 test set for perplexity. 20 needle-in-haystack
prompts at context lengths 1K / 2K / 4K / 8K for generation quality.

**Bit configurations to sweep:**
K6/V4 (the repo default), K5/V4, K4/V4, K6/V3, K4/V3, K3/V3.
For each config, run uniform allocation and retrieval-biased allocation
(retrieval heads get +1 K bit, sink heads get -1, same total budget).
12 conditions total — feasible in a weekend of GPU time.

**Head classification:** Run 100 diverse prompts, compute attention score variance
per head, label top-30% variance as retrieval, top-30% attention-to-position-0 as
sink, rest as local. Cite DuoAttention (Xiao et al. 2024) for the methodology.

---

## Phases

**Phase 1 — Establish the baseline measurement**

What happens: Extract raw K and V tensors at every layer and head using forward
hooks on the model's attention projections. Apply TurboQuant's rotation to the
extracted tensors (using the same fixed R the repo uses). Save both pre- and
post-rotation tensors to disk.

What you look at first: Plot coordinate histograms for three representative
heads — one retrieval, one sink, one local — from one middle layer. Sanity
check: do post-rotation coordinates look more Gaussian than pre-rotation?
Then measure kurtosis across all 28 layers × 8 heads and plot it as a heatmap.

Gate to Phase 2: The heatmap shows non-uniform kurtosis across heads.
If it is perfectly uniform, the rotation works as claimed and you pivot to
studying layer depth instead of head type.

**Phase 2 — Stratify by head type**

What happens: Classify all heads using attention score variance. Re-examine
your Phase 1 kurtosis values grouped by head type. Compute KS-test against
the theoretical Beta CDF per head. Plot kurtosis distributions as violin plots
stratified by head type (retrieval / sink / local).

What you look at: Does retrieval head kurtosis sit systematically higher than
sink head kurtosis? Is the separation statistically significant? Compute the
mean post-rotation kurtosis per type and the p-value of the difference.

Gate to Phase 3: If retrieval heads show meaningfully higher kurtosis
(say, > 20% higher than sink heads), proceed. The residual structure is real.

**Phase 3 — Connect to attention-logit error**

What happens: For each bit configuration, apply quantization and dequantization
to the saved K tensors. Compute two things per head: quantization MSE, and
attention-logit KL divergence. Then run the model under each bit config on the
evaluation set and record perplexity. Correlate each proxy (MSE, cosine sim,
KL divergence) against perplexity degradation. Compute R² for each.

What you look at: Which proxy has the highest R² against perplexity? If KL
divergence wins, you've shown the community is using a suboptimal proxy metric.
Also look at whether retrieval heads have disproportionately high KL divergence
relative to their MSE — this would confirm the softmax amplification effect.

Gate to Phase 4: If KL divergence on retrieval heads is disproportionately
high relative to MSE, adaptive allocation is worth testing.

**Phase 4 — Test adaptive allocation**

What happens: For each bit config, run two allocation strategies. Uniform:
same K and V bits for every head. Retrieval-biased: retrieval heads get +1
K bit, sink heads get -1 K bit, same total budget. Measure attention-logit KL
divergence and perplexity for both strategies.

What you look at: Does retrieval-biased allocation reduce mean KL divergence?
By how much? What is the perplexity delta? Compute the break-even: how large
does the KL gap need to be before the overhead of storing per-head bit configs
is worth paying? This is one number — the minimum KL improvement per bit of
metadata — and it makes the project feel like an engineering decision.

**Phase 5 — Second model**

Repeat Phases 1–3 on Llama-3.1-8B-Instruct. Compare the kurtosis patterns
and head type sensitivity between the two architectures. If the same head types
show the same patterns, the finding is architecture-general and much stronger.
If they differ, characterize what structural difference explains it — this is
often the most interesting part of a two-model study.

---

## What "Done" Looks Like

You can write this paragraph:

> "We find that TurboQuant's rotation reduces excess kurtosis by X% on average,
> but the reduction is non-uniform: retrieval heads retain Y% higher post-rotation
> kurtosis than sink heads (p < 0.01, KS-test). This residual structure translates
> to Z% higher attention-logit KL divergence under K4/V4 quantization. We show that
> attention-logit KL divergence predicts perplexity degradation with R²=A, versus
> R²=B for cosine similarity of K vectors, confirming it as a more reliable
> evaluation proxy. A retrieval-biased bit allocation (+1 bit K for retrieval heads,
> same total budget) closes C% of the attention-logit gap at no additional memory cost."

Every sentence has a specific experiment behind it.
That is a complete, credible research contribution for a master's student
targeting MLE roles at frontier labs.
