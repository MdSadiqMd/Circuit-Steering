# Toolkit findings, ready to file upstream

Six defects found in the Lexsi track repos while running this study, each with a measurement, the
check that reproduces it and a proposed fix. None is filed yet: filing is the owner's step. Versions:
CircuitKIT 0.2.0, AuditKIT 1.1.3, SafeTune 0.1.10. Details and dates are in `docs/notes.md`.

## CircuitKIT

### 1. The built-in IOI task attributes the wrong position, on a doubly-BOS'd prompt

- **Where:** `circuitkit/tasks/builtins/ioi.py::_build_eap_dataloader`, `_ioi_logit_diff`.
- **What happens:** `get_ioi_data_only` marks `end_idxs` at the "to" position, but the EAP path
  writes `model.to_string(tokens)` to its CSV, answer token and literal `<PAD>` text included, and
  the metric reads `input_length − 1`, which is the indirect-object name itself.
  `tokenize_plus(templated=False)` then prepends a second BOS to text that already starts with one.
- **Measured** (`checks/check_ioi_position.py`, CPU fp32, 16 pairs): the metric position is the IO
  token on 16/16 pairs and the BOS is doubled on 16/16. Clean / corrupt logit difference is 2.00 /
  −0.30 there against 11.16 / −1.01 at the "to" position. GPT-2 shows the same (0.50 / −0.43 against
  3.31 / +0.07).
- **Effect:** the IOI "circuit" is a circuit for "which name follows the indirect object". On Tiny
  Aya it scores below random node sets (patching 0.02–0.20 against 0.24–0.30).
- **Fix:** write the prompt up to `end_idxs` and register the text as templated. Our corrected task
  (`ioi_fixed`, `data/ioi_fixed.py`) is necessary, restores 2–5 % at 10 % of nodes with random sets
  at 0.000, and has stability 0.86.

### 2. EAP-IG scores scale as 1/batch_size when the metric is a batch mean

- **Where:** `circuitkit/backends/eap/attribute.py`.
- **What happens:** the code accumulates Σ Δa · ∂metric/∂a over examples and divides by
  `total_items`. With a metric that already returns the batch mean, every per-example gradient
  carries 1/B, so scores come out as mean attribution / B.
- **Measured:** edges computed at batch 4 against batch 2 have a median ratio of exactly 2.000.
- **Effect:** rankings are unaffected, magnitudes are not comparable across batch sizes. Any
  distance between two score sets computed at different batch sizes (CAD, weighted Jaccard) is
  wrong; in this study it produced CAD ≈ 1 for every cell of one seed until rescored (A13).
- **Fix:** divide by the number of batches when the metric is a mean, or require `mean=False`
  metrics and document it.

### 3. Pillar 5 random and magnitude baselines are degenerate at node level

- **Where:** `Pillar5_Baselines._build_random_circuit`.
- **What happens:** nodes are sampled among all nodes including `input`, and only edges between
  selected nodes are active. Without `input` no path carries clean activations, so the baseline
  sits on the corrupt floor with zero variance. The discovered circuit is rebuilt by
  `_reconstruct_circuit_graph`, which pins `input` and `logits`, so the two are not comparable.
- **Measured:** b1_en, five draws: exactly −11.4375 each (the corrupt metric), std 0.
- **Fix:** build baseline circuits through the same reconstruction path as the real one. Our
  control does this (`aya_circuits/random_baseline.py`).

### 4. MEMIT "success" is not top-1 correctness

- **Where:** `memit_wrapper.py`: success = `confidence_increase > 0`.
- **Measured:** layer-0 edits report 49–60 of 60 successes with the mean target probability moving
  from 0.003 to 0.08; top-1 efficacy of the same edits is 0.37.
- **Fix:** report top-1 (or exact-match) efficacy next to the confidence change.

## AuditKIT

### 5. `paired_bootstrap` p-value is not a test of zero difference

- **Where:** `auditkit/_bootstrap.py::paired_bootstrap`.
- **What happens:** it counts resamples whose |mean| is at least the observed |mean|, but the
  resamples are drawn from the observed differences without centring them, so the resampled means
  are centred on the observed effect, not on zero. The returned p is close to 0.5 for any effect
  size and `significant` is almost never true.
- **Measured:** 200 pairs with a true shift of 2 standard deviations (observed Δ = 2.13) return
  p = 0.512, `significant: False`; a shift of 0 returns p = 0.542.
- **Fix:** centre the differences before resampling (`d − mean(d)`), or return the percentile
  interval and derive significance from it. We use its `delta` and means and compute percentile
  intervals ourselves (`audit/analyze.py::ci`).

## SafeTune

### 6. Refusal judges need vLLM and CUDA; the regex detector is English-only

- **What happens:** the WildGuard path is vLLM/CUDA-only and gated, so on Apple Silicon or CPU the
  only available judge is the regex behind `OverRefusalMetric`, which needs ≥ 20 characters of
  English. Non-English refusals are missed.
- **Measured (A22, Tiny Aya replies to harmful prompts):** see `docs/results/a22_b2.md` §5 for the
  regex judge against an LLM judge per language.
- **Fix:** a transformers / llama.cpp backend for the judge, and multilingual refusal patterns
  (ours for hi/ar/fr/sw are in `neb/judge.py`).

## Not a toolkit defect, but worth a note in the docs

Merging a LoRA adapter into bf16 weights rounds 31–43 % of its entries to zero
(`docs/audit-2026-11-08.md` B.1). Any evaluation of a merged adapter on a bf16 model is an
evaluation of a different adapter. We apply adapters as exact fp32 hooks (A18).
