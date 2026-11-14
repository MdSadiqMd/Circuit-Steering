# Technical report: circuit-level interventions on Tiny Aya, verified (fifth audit closed by amendment A18, 2026-11-09)

Status: **final (2026-11-09): amendments A15–A18 run and reported; the fifth audit's one pipeline
defect (bf16-merged LoRA adapters) fixed and every LoRA cell re-audited; report v0.18.** This document
is the owner's requested deliverable; it summarises and points into the research report
(`docs/report.md`, v0.18), the four audit records
(`docs/audit-2026-11-0{5,6,7,8}.md`; the first audit has none), the frozen pre-registration with its amendments
(`docs/preregistration.md`), the running notes (`docs/notes.md`) and the decision log
(`docs/decisions.md`). Every number here is traceable to a file under `docs/results/`, `checks/` or
`runs/`; nothing is quoted from memory.

## 1. Problem statement verification

The brief (`docs/project-brief.md` §1) asks for three things on `CohereLabs/tiny-aya-global`: discover
a circuit causally responsible for a behaviour; intervene only there by circuit steering,
circuit-restricted fine-tuning and knowledge editing; verify with controls that nothing else broke.
Its §3.1 sentence makes the measurement the goal: whether confining the three interventions to a
discovered circuit changes the target behaviour with less collateral than confining them to
size-matched random, inverse, gradient-selected or no components, plus the reusable five-layer audit.

| brief requirement | status after this audit | where |
|---|---|---|
| §1.1 discover a causally responsible circuit | met for fact recall (joint-ranked set: patching 0.33–0.56 en, random sets 0.000; necessary: ablated r′ ≈ 0) and language-ID; **not met for the per-type set the pre-registered cells intervened on** (patching 0.004–0.03) until A15; not met for refusal at 10 % (patching ≈ 0 under every definition); the IOI control was mis-specified by the toolkit (A16) | report §4.1, audit-07 §B.1, §C.7 |
| §1.2 intervene only there, three mechanisms | met: node steering, head-masked LoRA (q/o per head, k/v per GQA group), MEMIT at the circuit's MLP; A15 adds LoRA and steering on the faithful joint set | report §4.2, §5 |
| §1.3 verify nothing else broke, with controls | met on five layers with random (5 draws), inverse, Taylor, unrestricted and norm-matched perturbation controls in every table; L4's sensitivity measured (power check: under-sampled, not blind) | report §4.3–4.4 |
| §3.2 pre-registered H1–H5 with thresholds before the first run | met: frozen 2026-11-02 08:43 IST, first cell 08:43:40; 17 dated amendments, the frozen text never edited | preregistration.md |
| §3.3.1 B1 ≥ 200 triples in 5 languages | 180 triples (owner decision 2026-11-06), 5 languages | report §5 |
| §3.3.2 three bystander circuits + language-ID | met; the built-in IOI task was mis-specified by the toolkit and is replaced by `ioi_fixed` (A16: necessary, 2–5 % sufficient at 10 %, random sets 0.000, stability 0.86) | §4.1 |
| §3.3.3 3 interventions × 5 conditions × ≥ 3 seeds, ≥ 5 random draws, weights norm-matched | grid met (318 pre-registered cells + A4 / A8 / A15 additions); weight norm-matching replaced by a norm-matched random perturbation control per weight cell (owner decision) | §4.2 |
| §3.3.4 five audit layers per cell | met; ripple (A9), Anand-style arbitration (A12), relearning (A10) added by the third audit | §4.3–4.4 |
| §3.3.5 scorecard per cell with AuditKIT `compare` CIs | met (`scorecard.json` next to every cell) | audit/analyze.py |
| §3.3.6 supported / refuted / inconclusive per hypothesis with the strongest alternative | met | §4.3–4.5 |
| §3.3.7 public repo, HF artifacts, write-up, Space | built; publication waits for the owner (the Hub still holds harmful completions from the 2026-11-03 push and must be squashed) | §5 |
| §5 MUST chain CircuitKIT + AuditKIT + SafeTune | met (discovery / faithfulness / hooks / MEMIT / re-attribution → refusal-direction baseline + second judge → every score and paired comparison) | README |
| §5 signed ablation under zero and mean, never clamped | met, with the exact per-task floor of the ratio (r₀) and its bf16 resolution | audit-07 §B.1–B.2 |
| §11 uv, ruff, `just check`, manifests | partly met: data-stage manifests post hoc (`data/MANIFEST.json`), audit manifests carry no input or output hashes, and every manifest's git SHA is the skeleton commit (code never committed) | justfile, report §5 |
| §12.1 faithfulness curve 1–30 %, signed, both replacements, with area | met | circuits.md |
| §12.3 EAP-IG vs true patching ρ ≥ 0.7 | met for the four original tasks (0.92 / 0.81 / 0.95 / 0.73); not met for the corrected IOI bystander `ioi_fixed` (0.67 signed, 0.45 magnitudes; A18 chain), the remedy is queued as A19 | checks/results/eap_vs_patching.json |
| §12.4 sufficiency, necessity, minimality | met (necessity added by A5, re-referenced to r₀) | circuits.md |
| §12.5 steering directions vs full-component patching | met | checks/results/steer_direction.json |
| §12.6 ≥ 3 seeds, CIs, effect sizes | met; seed-cluster CIs and a Holm count added | hypotheses_*.md |
| §13.1 cross-circuit CAD | delivered, plus a within-bystander split (A17) | §4.3–4.4 |
| §13.2 faithfulness-gated locality (H3) | gate half delivered on 18 gated and 6 ungated cells (the three ungated LoRA cells show the predicted advantage); monotone half untestable with three pillar values | §4.3–4.4 |
| §13.3 Circuit-Interference Law across interventions and languages (H4) | delivered; "consistent in direction, not established"; A17 holds circuit identity fixed | §4.3–4.4 |
| §13.4 first circuit edits on an Aya model with cross-lingual collateral (H5) | delivered; refuted at MLP 0 against the causal-tracing and Taylor sites | §4.3 |

Interpretations where the brief is ambiguous (recorded in the pre-registration and its amendments):
"the circuit" for the cells is the pre-registered per-type set (top 10 % heads and top 10 % MLPs
separately, A5), and A15 adds CircuitKIT's joint-ranked set as a second, faithful target;
"size-matched" is per component type; "unrestricted" MEMIT is layers 0–28 because the last layers'
edit gradient is zero; "verified" means every layer measured with controls, not a proof of absence
(brief §3.4).

## 2. Technical improvements (by audit, with impact)

Second audit (2026-11-05): the per-type intervention set, the pillar set and the L4 masks were three
different node sets sharing the word "circuit"; A5 evaluated the pre-registered set through the same
CircuitKIT path with necessity and re-attributed every L4 cell with the edge matrices kept. Impact: the
fact circuits' intervention set restores 0.4–3 % of the logit difference (joint set 33–56 %), H3's gate
fires on B1 as well, the "faithful circuits show no advantage either" reading was withdrawn. A6 rebuilt
85 B1 steering cells with fold-disjoint sources (no reading changed).

Third audit (2026-11-06): EAP-IG scores scale as 1/batch_size with a batch-mean metric; the B1
discovery-seed-0 base edges had been cached at batch 4 while every cell ran at batch 2, so a third of
the B1 L4 data was inflated 2× (a no-op perturbation scored CAD 1.0). A13 rescored it offline from the
kept edges. Impact: no real B1 intervention's target CAD exceeds the resample floor, H3 18/18, H4 on B1
from "refuted" to "consistent in direction". A post-hoc criterion that A5 had slipped into the H3 gate
was exposed (both readings now printed). Seed-cluster CIs, Anand's arbitration probe, causal tracing,
ripple, relearning, the edge-level distance, the ROME smoke, the profiler and the k/v/gate/up
equivalence case were added.

Fourth audit (2026-11-07): exact per-task ratio floors r₀ = −c/(cl − c) and bf16 resolution of every
reported ratio; the IOI 1 %/2 % plateau explained (six added nodes net to less than one bf16 quantum);
the built-in IOI task found to read its metric at the answer token with a double BOS and misaligned
pairs (A16: `ioi_fixed`, re-attributed on every cell); a Holm step-down over the H2 tests; amendment
A15 (B1 LoRA and steering on the faithful joint set with count-matched random and inverse sets, full
audit); the L4 power check at 64 examples / 10 IG steps. This session added independent unit checks
of the cross-circuit CAD edge masks and of the statistics helpers, and amendment A17 (within-bystander
overlap split from the kept edges, 20 s per behaviour on CPU, model-free).

Fifth audit (2026-11-08, `docs/audit-2026-11-08.md`): an independent read of every deliverable against
the result files, cell tensors and model weights. Found: LoRA cells are applied by merging ΔW into the
bf16 weights, which rounds 31–43 % of the non-zero entries to zero (L2 error 25–37 %; efficacy
re-measured with the exact adapter is unchanged on 4 of 5 cells); `n_trainable` counts masked adapter
entries; k/v adapters cover whole GQA groups; the fact circuit was discovered on 2–3 office-holder
questions per 64 while 137 of 180 targets are office holders; the A15 locality advantage rests on three
seeds (paired p = 0.13) and fails H2's inverse condition; the power check's floors halve but the cells'
CADs do not; norm-matched noise reproduces MEMIT collateral only at MLP 0; and a set of stale numbers
in this file, the report and the README. Impact: text corrected in all three; no run re-done; no
H1–H5 verdict changes.

Amendment A18 (2026-11-08 14:13 – 2026-11-09 05:41 IST, `scripts/chains/13_a18.sh`): the fifth audit's defect
removed at the source. LoRA cells are applied as forward hooks adding the fp32 low-rank term to the
bf16 output with one rounding (PEFT's training-time forward) on both the HF and the TransformerLens
model (`interventions/cell.py` `lora_term`, `add_term`, `_lora_hooks_tl`); verified bit for bit against
PEFT's forward on all 96 adapted modules of a real cell (`checks/results/apply_equivalence.json`: 0 ulp; the
merged weights differed on 24 % of output entries), HF ↔ TL logit-delta cosine ≥ 0.997 on real cells.
New `audit/l4.py` attributes the target, the bystanders and `ioi_fixed` in one pass per cell (211 s
instead of three scripts), `Cell.peek` stops loading tensors to read a name, `cad_triple` replaces
five copies of the floor triple, `n_trained` (unmasked adapter entries) is recorded and reported next
to `n_trainable`, the A15 overlap column measures joint conditions against `circuit_joint`, every
seed-tagged test prints per-seed differences and a paired t, manifests record `git_dirty`,
`data/b2_build.py` uses `ast.literal_eval`, `just test` runs the three repos' tests. All 79 LoRA cells
re-audited on every layer; differences at bf16-noise size (mean |Δ| 0.005 efficacy, 0.003 L3, 0.002
bystander CAD); the A15 LoRA L3 advantage falls below the Holm line (p = 0.047; B1 7 of 20 survive).
Run defects fixed: MPS allocator growth across cells (OOM at the 16th B2 cell → `audit.cells_per_process`
cap with a shell loop), the mask check's per-type assumption for the A15 joint sets.

## 3. Mathematical validation

**Faithfulness ratio and its floor.** CircuitKIT reports r = (m − c)/(cl − c). A model whose logit
difference is driven to 0 scores r₀ = −c/(cl − c), which is 0.5 only when c = −cl; measured r₀ is 0.509
(b1_en), 0.47–0.49 (other fact circuits and language-ID), 0.155 (refusal, `checks/results/metric_span.json`;
the per-type evaluation in `circuits.md` gives 0.152, within one bf16 quantum), 0.024 (built-in IOI). The
zero-referenced ratio r′ = m/cl satisfies r′ = (r − r₀)/(1 − r₀) = r(1 − c/cl) + c/cl (both identities
checked numerically in `tests/test_units.py::test_stats_helpers`). Reading necessity in r′: ablating
the per-type fact set leaves −8 … +3 % of the clean logit difference, random per-type sets 7–30 %.
Pillar 1 means are accumulated in bf16 (8-bit significand), so a ratio is resolved to
2^(⌊log₂ m⌋ − 7)/(cl − c) = 0.002–0.006; differences of that size are not differences.

**EAP-IG scale.** `circuitkit/backends/eap/attribute.py` divides the accumulated scores by
`total_items` (and by `total_steps` for IG); with a batch-mean metric each example's gradient already
carries 1/B, so scores are mean attribution / B. Rankings are scale-free; magnitudes are not comparable
across batch sizes (the A13 bug, and the reason cross-task weighted Jaccard is also reported on
unit-mass scores).

**Cross-circuit CAD.** CAD(C) = Σ_{e∈E(C)} |s_e(θ) − s_e(θ′)| / Σ_{e∈E(C)} |s_e(θ)| (Sadhu et al. 2026
Eq. 6), with E(C) the edges whose source and destination are both in C (logits always a destination).
`neb.mechanistic.induced_edge_mask` was checked this session against a brute-force decode of
CircuitKIT's documented index layout (forward: input, then per layer heads and MLP; backward: per
layer q, k, v head slots then the MLP input; last slot logits) on a parallel-attention graph, including
the cohere2 structural fact that a layer's heads do not feed that layer's MLP; a model-free
`Graph.from_model` on the model's dimensions reproduces the stored CAD exactly (5354 edges, 0.13581937
for `b1_lora_circuit_s0` on b2_en; 5360 edges if parallel attention is wrongly switched off). The
independent recomputation of the third audit (max |diff| 2.4e−7 over 24 pairs) stands.

**What "below the resample floor" means.** The cell's CAD compares intervened and base edges on the
*same* 32 examples, so its measurement noise is the same-data repeat floor, which is exactly 0 on this
machine (deterministic EAP-IG on MPS for identical inputs). The resample floor is base-vs-base on
*another* subset: the between-subset variation of the base attribution. "CAD − floor < 0" therefore
means the intervention changes the edges of these examples by less than changing the examples changes
them; it is a scale statement about the attribution estimate, not a null-hypothesis test of "no
effect". It also says nothing about how a cell-vs-cell CAD difference varies with the example set,
which was never measured. Raw CAD still orders every real intervention above its norm-matched control
(report §4.3). The power check measured the budget dependence: at 64 examples / 10 IG steps the floors
fall to 0.54–0.64× while the cells' CADs fall to 0.38–0.75× on the target and only 0.71–1.08× on the
bystanders, so the excess moves toward 0; the check changes examples and IG steps together and cannot
say which drives it. On the per-type masks the strongest weight interventions come within 0.02 of the
floor, LoRA at the circuit and steering stay 0.12–0.24 below; on the joint masks unrestricted LoRA and
Taylor-layer MEMIT exceed it (§6).

**Partitioned CAD is magnitude-confounded (A17).** For a partition E = E₁ ∪ E₂ of a circuit's edges,
CAD(E₁) − floor(E₁) and CAD(E₂) − floor(E₂) share no denominator: CAD(Eᵢ) = Σ_{Eᵢ}|Δ| / Σ_{Eᵢ}|s|, so
the part holding the large-|s| edges reads lower for any Δ of a given size. Dividing each part's CAD by
its own resample floor cancels Σ_{Eᵢ}|s|: CAD(Eᵢ)/floor(Eᵢ) = Σ_{Eᵢ}|Δ_int| / Σ_{Eᵢ}|Δ_resample|, the
change in units of that part's own sampling noise, and Λ = log of the ratio of the two parts' ratios is
scale-free. Measured: the raw difference D takes either sign when the split is by the intervened
components (positive for 13 of 28 B1 and 8 of 18 B2 condition rows) and is negative for every row when
the split is by the pre-registered target set (B1 −0.006 to −0.256, B2 −0.099 to −0.235), which is the
confound at work. Λ for LoRA / steering at the circuit is +0.074 / +0.050 on B1 and +0.196 / −0.274 on
B2, not consistently different from the norm-matched perturbation controls (LoRA's control +0.238 on
B1, −0.008 on B2), and negative for MEMIT at MLP 0 (−0.236 B1, −0.210 B2).

**Steering norm match.** S = sqrt(Σ_n ‖coeff·v_n‖²) is the Frobenius norm of the stacked per-node
vectors; every B1 steering cell, A15 included, is matched to |Δ| 8.65 at the 0.5× main multiplier, so
the joint-set vectors (natural size 74.1 at coefficient 1) are injected at coefficient 0.117 against
0.5 for the per-type set (natural size 17.3). Equal injected energy, not equal per-node norm.

**Statistics.** Paired percentile bootstrap (2000 resamples) for every comparison, with a seed-cluster
bootstrap beside it (3 clusters, 10 distinct resamples: not a valid interval, and it can be narrower
than the pair CI, e.g. B1 LoRA L4 circuit − random pairs [−0.006, +0.031], clusters [+0.009, +0.025]);
two-sided bootstrap p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)) floored at 1/2000; Holm step-down at α = 0.05
over the H2 tests (B1: 12 pre-registered + 8 post-hoc A15; B2: 12; implementation checked on a hand
case in `test_stats_helpers`). Every p-value treats the pairs inside a seed (language × family for L3,
bystander for L4) as independent; the independent unit is the seed, of which there are three. H4's per-cell Spearman over
three bystanders takes values in {−1, −0.5, +0.5, +1}; its mean is a mean of such values and the CI is
on the fraction of cells with agreeing ranks. H1's ratio CI resamples (unrestricted, circuit) pairs and
takes the ratio of means.

**Verification status by claim type.** Formal identities: r′, the Holm rule, the magnitude confound.
Numerical verification: edge masks, CAD, HF↔TL equivalence (cosine ≥ 0.997 on every case incl.
k/v/gate/up), LoRA mask leakage 0.0, label-position loss equal to HF's, the memory-patch parity (node
Spearman 0.9993). Empirical observations: every result in report §4. Unverified hypotheses: the
mechanism behind the EAP-IG resample floor's size (measured: a 7× budget halves it while the cells'
CADs change by −62 % to +8 %; the mechanism is not derived);
whether a parameter-matched LoRA control would change the H1 LoRA reading (the A15 cells say parity
tracks the number of adapted MLP layers).

## 4. Bug and correctness report

Found and fixed across the audits (each with the artifact that re-measured it):
- L4 base-edge batch-size mismatch for B1 discovery seed 0 (A13, `audit/l4_rescore.py`).
- Three "circuit" node sets across stages (A5, `aya_circuits/cellset.py`, `audit/l4_cellset.py`).
- B1 steering sources overlapping the efficacy fold (A6, 85 cells rebuilt).
- A post-hoc gate criterion added by A5 while claiming none (report H3: both readings).
- L5 "arbitration" and "circuit distance" not the cited papers' metrics (A12, `l4_target_edge_l1`).
- Built-in IOI task: metric at the answer token, double BOS, misaligned pairs (A16, `data/ioi_fixed.py`).
- 45 of 207 refusal EAP pairs not length-matched (diagnostic task `b2m_en`, A16).
- Stale and false prose statements (third audit B7–B8), double rounding (D).
- Harmful completions published to the Hub (A7, `data/sanitize.py`, `just check` scan; the squashing republish awaits the owner).
- Back-translation prompt that made the model answer instead of translate; ROME smoke OOM on MPS.
- This session: the A15 chain built no steering controls (the grid exits after three cells per process
  and only the LoRA loop re-invoked it); `scripts/chains/11_a15c.sh` built and audited them. The A17 section's
  positivity guard rejected every row because the same-data repeat floor is exactly 0 (fixed); the
  first "shared with the intervened components" definition used the induced (both-endpoint) mask,
  which leaves one edge for a single-MLP MEMIT cell (replaced by a touching mask with its own
  brute-force test).

Tests: `just test` runs 98 tests across the three repos (`--import-mode=importlib`): 13 unit tests,
and `tests/test_entrypoints.py`, which imports every module and script and runs the four `--check`
self-tests (certification rule, A20 condition sets, A21 cap and noise, A21 statistics); the Holm
counts and the H4 statistic were recomputed with the corrected bystander and, after A18, the B2 LoRA
L3 difference (−0.008) passes Holm's last step at exactly p = 0.050, reported as borderline and
immaterial; `ruff check` and `ruff format --check` clean; `just check` (ruff, pytest, dry acceptance
check, leak scan, shellcheck and shfmt, every chain's stages printed under `DRY_RUN=1`) passes. GPU-dependent checks write
JSON under `checks/` (equivalence, EAP-vs-patching, LoRA masks, metric span, IOI position and plateau,
steering direction, ROME, profiler, MPS numerics, memory-patch parity, back-translation).

Fixed by A18: the bf16-merged LoRA application (every LoRA number now describes the trained adapter;
MEMIT deltas, merged in place, lose 0.2–4 % in L2, within the bf16 floor of the metrics). Remaining
concerns (documented, not fixed): the corrected IOI bystander's EAP-IG ranking agrees with exact
patching at ρ = 0.67 only (A19 queued). LoRA conditions are node-count
matched, not parameter matched, and `n_trainable` counts masked entries (trained entries: per-type
circuit 2.8–3.0 M vs random 3.2–3.5 M vs unrestricted 15.1 M; joint circuit 7.8–8.2 M vs random_joint
8.2–8.6 M; reported 3.5–3.7 / 4.1–4.6 / 15.1 and 8.4–8.8 / 9.0–9.4 M); k/v adapters are trained for the
whole GQA group of a selected head; the A15 table's "overlap with circuit" column counts overlap with
the per-type circuit; the fact circuit was discovered on few office-holder questions (3 / 3 / 2 of 64
per seed) while 137 of 180 targets are office holders; no artifact is tied to a code commit (every
manifest records the skeleton commit `416938a`, the sibling repos are not git repositories); the
first-word answer matcher admits wrong people (a stricter rule
lowers B1 efficacies by 0.02–0.11, H1 unchanged); 13–16 of 180 targets are miscorrections or contested;
the neighbourhood and never-edited sets are all `capital` facts; held-out harmful sets are 12 prompts
per language; MPS has no deterministic index_put kernel (steering cells reproduce bit-exactly, LoRA
cells are not checked bitwise); the 28 benign near-miss prompts are both the steering source and the
over-refusal evaluation set.

## 5. Performance report (one Apple-silicon machine, 30 GB MPS cap, bf16)

| stage | n | median wall | range | peak memory |
|---|---|---|---|---|
| EAP-IG discovery (64 ex, 5 IG steps, batch 4 / 2) | 24 | 169 s | 119–264 s | 25.4 GB |
| faithfulness pillars + size curve + stability | 34 | 864 s | 250–1556 s | 30.9 GB (one pause-free run at the cap) |
| per-type evaluation + necessity (`cellset`) | 16 | ~24 min | 13–32 min | under cap (langid at batch 2) |
| MEMIT cell (60 facts, fp32 statistics on CPU) | 43 | 694 s | 136 s – 2.4 h (29-layer) | 28.8 GB |
| LoRA cell (60 steps, 2048-token batches) | 45 | 147 s | 104–266 s | 26.7 GB (9.3 GB with the label-position loss alone) |
| steering cell | 40 | 6 s | 2–516 s (first cell builds the means) | 26.4 GB |
| HF audit per cell (L1/L2/L3/L5, 5 languages) | 335 | 254 s | 31 s (sweep cells) – 613 s | 29.8 GB |
| L4 re-attribution per cell (4 tasks, 32 ex, 3 IG steps, batch 2) | 185 | ~165 s | 143–194 s | under cap |
| A18 HF audit per LoRA cell, adapter as hooks (L1/L2/L3/L5) | 79 | ~300 s | 284–345 s | 28.8 GB (cap at 12 cells per process after an OOM) |
| A18 L4 single pass per LoRA cell (target + 3 bystanders + `ioi_fixed`, `audit/l4.py`) | 79 | 211 s | 202–246 s | under cap |
| A18 chain end to end (79 cells, ripple, arbitration, relearning, masks, power, checks, tables) | – | 15.5 h incl. one restart | – | – |
| L4 re-attribution at 64 examples / 10 IG steps (power check) | 4 cells + 12 base | 1080 s per cell, 640–1170 s per base attribution | – | under cap |
| A17 shared-edge split (CPU, model-free) | 2 | 20 s | – | < 2 GB |

Profiler (`checks/results/profile_lora.json`): 20 LoRA steps in 44 s, 462 tokens/s, peak 9.3 GB, no mask leak;
95 % of CPU time waits on MPS synchronisation, so the CPU profile cannot attribute GPU time; GPU-bound.
Optimisations that mattered, in order of saved memory: label-position loss (26.7 → 8.3 GB for LoRA),
CircuitKIT memory patch for MEMIT fp32 statistics (parity-checked), L4 batch 4 → 2, faithfulness batch
8 → 4 → 2 for language-ID, three LoRA cells per process (MPS allocator growth). Offline recomputation
from kept edge matrices (A5 joint check, A13 rescore, pillar masks, A17) replaced three full
re-attribution campaigns. A15 chain: 22 cells built in 32 min, HF audit 2.2 h, L4 (per-type and joint
masks) 46 min, ripple + arbitration 20 min, power check 2.1 h; the steering-control follow-up 2 h.

## 6. Experimental results

The research report (`docs/report.md` §4) holds every table; the verdicts as of this audit:
- **H1 efficacy parity.** MEMIT refuted (0.43; MLP 0 is a poor edit site, MLP 5 reaches parity); LoRA
  refuted for the per-type set (0.41; random head sets 0.64 with ~22 % more adapter entries, ~18 %
  more trained entries) and supported for the faithful joint set (1.000, A15), where count-matched
  random and inverse sets also reach parity: capacity, not membership (efficacy vs adapter size
  Spearman 0.85 over the 28 non-control B1 LoRA cells, 0.86 on trained entries); steering vacuous
  (≤ 0.13; joint set ≤ 0.07).
- **H2 locality advantage.** Refuted in the wrong direction for MEMIT at MLP 0 on L3 and L4 (|Δθ| 131
  vs 36–69 elsewhere; norm-matched noise at MLP 0 does more damage than the edit, while at every other
  site the edit does 4–12× the damage of its noise control, so "size, not content" holds at MLP 0 only);
  null for LoRA; mixed and at metric resolution for steering; B2: refuted or null for every
  intervention, with one borderline LoRA advantage, L3 circuit − random −0.008 (p = 0.050, passes
  Holm's last step at its threshold, one-tenth of the metric's scale) next to a LoRA L4 disadvantage
  (+0.008) and an inverse disadvantage (+0.04). On the faithful set (A15) LoRA shows the study's one
  locality advantage, L3 −0.008 [−0.017, −0.000] against count-matched random (uncorrected p = 0.047;
  after A18 it no longer survives Holm, B1 7 of 20 survive; with the bf16-merged adapters it had read
  −0.011, p = 0.002), per seed −0.013 / −0.007 / +0.000 (paired t over the three seeds p = 0.21), it
  is null against inverse (so the pre-registered H2 rule, circuit < random and < inverse, is not met)
  and null on L4; A15 steering is vacuous with differences ≤ 0.015.
- **H3 faithfulness gates locality.** Gate half: 18/18 gated cells show no L4 advantage (13/18 with no
  L3 advantage either), and of the six A15 cells that pass the gate the three LoRA cells show the
  predicted advantage on L4 (two of them on L3 too) while the three vacuous steering cells show ±0.01; both gate
  readings printed. Monotone half: six ungated cells and three pillar values give it no power.
- **H4 overlap predicts interference.** Inconclusive. With the corrected IOI bystander the raw
  pre-registered statistic holds by its letter (per-cell Spearman of overlap vs bystander CAD +0.71 B1,
  +0.68 B2) but the bystanders' own resample floors rank with overlap too, every value is below its floor,
  and the floor-adjusted ranking reverses (−0.25, −0.39); the v0.14–v0.15 "consistent in direction"
  reading (+0.59 / +0.54 floor-adjusted) rested on the mis-specified IOI task. The within-bystander split
  (A17) attributes where edges move to where parameters changed, with no consistent difference between
  trained changes and norm-matched noise.
- **H5 cross-lingual propagation.** Refuted at MLP 0 against the causal-tracing (MLP 3) and Taylor
  (MLP 16) sites (−0.22 and −0.16); `circuit_deep` propagates as well as the traced site.
- **Findings no single metric shows.** Single-layer MEMIT edits raise never-edited top-1 error from
  0.02 to 0.28–0.65 while WikiText moves ≤ 1 % and MMLU ≤ 6 points; every B2 LoRA condition buys
  0.20–0.37 of non-English refusal for +0.37–0.54 benign over-refusal; the SafeTune refusal direction is
  the strongest and least local steering cell; arbitration: the base model over-trusts injected
  passages and MEMIT at MLP 0 makes its confident decisions worse.
- **L4 power check.** At 64 examples / 10 IG steps the resample floors halve (b1_en 0.51 → 0.27,
  b2_en 0.39 → 0.22, IOI 0.48 → 0.31, langid 0.43 → 0.27) while the cells' CADs fall less (target
  0.38–0.75×, bystanders 0.71–1.08×); on the per-type masks the unrestricted LoRA and the Taylor-layer
  MEMIT come within 0.02 of the floor on the target and LoRA at the circuit and steering stay 0.12–0.24
  below; on the joint masks the unrestricted LoRA (+0.020 target, +0.044 langid) and the Taylor-layer
  MEMIT (+0.035 target) exceed it. The check used the built-in IOI task. Under-sampled, not blind; the
  full rerun (≈ 60 h) was not done. **IOI fixed (A16).** CircuitKIT's built-in IOI EAP path reads its metric at the answer token with
  a double BOS and misaligned pairs; the corrected task on three seeds is necessary, 2–5 % sufficient at
  10 % and 43–57 % at 30 %, with random sets at exactly 0.000 and stability 0.86 (built-in 0.66–0.72);
  re-attributed on all 213 audited cells it is the least disturbed bystander (raw CAD 0.07–0.08, floor
  0.33); L4 H2 differences move by ≤ 0.05, no H1/H2/H3/H5 verdict changes, H4 becomes inconclusive (above).
  **Length-matched refusal task (`b2m_en`).** Same circuit (Jaccard 0.80 with the original) and the same
  unfaithfulness (patching −0.004 / +0.003 / −0.003): the 45 misaligned pairs did not cause the refusal
  circuit's failure.

## 7. Reproducibility

Environment: Python 3.12 via `uv` (`uv.lock` committed in each of the three repos), `circuitkit==0.2.0`,
`auditkit==1.1.3`, `safetune==0.1.10`, `transformer-lens==3.8.0`, `transformers==5.22.1`,
`torch==2.15.0`, `peft==0.21.2`; `nix develop` provides uv / Python / just / bash from the pinned nixpkgs.
Layout: `~/Developer/circuit-steering/{circuit-steering, tiny-aya-circuits, nothing-else-broke}` side
by side (editable path dependencies). Model access needs `HF_TOKEN` with the Cohere licence accepted;
every chain sets `CIRCUITKIT_TL_MEMORY_PATCH=1` (`scripts/lib.sh`); the chains that
run after the model is cached also set `HF_HUB_OFFLINE=1`.

Commands (from `circuit-steering/`; every stage skips cached outputs and is resumable):
```
just check                      # ruff, pytest, dry acceptance check, leak scan, shell, chain plans
just chain full                 # data → circuits → grid → audit → tables → analysis (or: nix run .#full)
just chain cellset              # A5: per-type evaluation + necessity, L4 on the per-type masks
just chain a6                   # A6: fold-disjoint B1 steering cells
just chain gaps_circuits && just chain gaps_followup && just chain gaps_final   # third-audit closures
just chain a15 && just chain a15b && just chain a15c && just chain a16         # fourth-audit closures
uv run python audit/l4_shared.py grid.behaviour=b1   # A17 (CPU, 20 s)
just chain a18 && just chain a19 && just chain a20 && just chain a21           # fifth audit, A19–A21
just reproduce                  # every chain above in the order it was first run
just analyze                    # every table and analysis from the stored runs (CPU, minutes)
```
Configuration: `configs/base.yaml` (seeds [0, 1, 2], 5 random draws, top fraction 0.10, L4 at 32
examples / 3 IG steps / batch 2, 30 GB budget); overrides as `key.sub=value`. The circuit, grid and
audit stages write a manifest (git SHA, `uv.lock` sha256, model revision, seed, dtype, device, wall
time, peak memory; output hashes for the circuit and grid stages, input hashes only for the circuit
stages and the model filter, neither for the audit manifests, e.g. `manifest_hf.json`); the data
scripts write none and `data/MANIFEST.json` hashes every data file post hoc. The git SHA is the
skeleton commit `416938a` for every artifact, because the code was never committed. Determinism: seeds fixed
everywhere; EAP-IG and steering cells reproduce bit-exactly on this machine; LoRA cells are not
bitwise-checked because MPS lacks a deterministic `index_put_with_accumulate`.

## 8. Three-pass review (this session, on the fourth-audit state and the queued chains)

**Pass 1, mathematics.** Verified the r′ identity and bf16 resolution claims; verified the CAD edge
masks by brute force and the model-free reproduction of stored CAD; checked `weighted_jaccard`
(Σmin/Σmax), Taylor scores (Σ|θ ∂L/∂θ| with the GQA share split per group), the perturbation control
(per-module B·A Frobenius match under the real zero pattern), the AURC (discrete risk–coverage
average) and the spread (population sd) against their definitions; derived the magnitude confound of
partitioned CAD and replaced the raw statistic with the scale-free Λ; articulated what "below the
resample floor" does and does not mean. No defect found in the existing formulas.

**Pass 2, software and numerics.** Confirmed `circuit_scores.json` node scores equal the pillar-set
ranking (so A15's inverse and random joint sets rank on the right scores), `_coerce("null")` → None
(A16's `curve_tasks=null` gives every new task the size curve and stability), the overlap / table /
analysis stages enumerate tasks from the run directory, and `analyze.py` switches the IOI bystander to
`ioi_fixed` only when every L4 cell has it. Found and fixed the two A17 issues in §4. Regenerated the
B2 analysis: every pre-existing number unchanged, the A17 section appended. After the chains the B1
tables regenerated with the pre-registered sections' numbers unchanged and the A15 / A17 / Holm sections
appended; the H3 pooled gate lines read as before (18/18, 13/18).

**Pass 3, research.** The reviewer objections of brief §9 are each answered in the artifacts, and the
remaining weaknesses are stated rather than softened (§9 below). The A15 cells are the experiment the
first three audits left undone; their reading follows the decision rules pre-committed in the amendment
text, and the one positive result (LoRA L3) is reported with its size (0.011 on a 0.03 scale), its
seed-cluster interval, its Holm status, its driver (non-English known-fact accuracy) and the control
that does not confirm it (L4).

## 9. Publication-readiness assessment

Contributions that stand: the five-layer Nothing-Else-Broke audit with controls on every row; cross-
circuit CAD with same-settings floors and the finding that the mechanistic layer at 32 / 3 resolves only
layer-0 MEMIT; the demonstration that at the layer-0 site a change of the size MEMIT needs is damaging
whatever it writes (norm-matched noise there does more damage than the edit; at every other site the
edit does 4–12× the damage of its noise control, so elsewhere the content matters); the first circuit-targeted
steering / LoRA / editing results on an Aya model with cross-lingual collateral; five audits that found
and fixed real bugs in the authors' own numbers, with every definition traced to its primary paper.

What a mechanistic-interpretability reviewer will still hold against it: one model, one discovery
method, node-level circuits, three tied seeds; the pre-registered MEMIT site is MLP 0 by construction of
a logit-difference EAP-IG ranking, so the MEMIT verdicts are about layer-0 editing; the pre-registered
intervention set was unfaithful (the A15 cells test the faithful one, post hoc); the refusal circuit is
not a sparse node circuit at 10 %; L4 at the study's budget cannot see LoRA or steering effects on
bystanders (measured, not assumed); the one positive locality result rests on three seeds (paired
p = 0.21 after A18) and one layer, no longer survives the Holm correction, and does not meet H2's
inverse condition; LoRA conditions are node-count, not parameter, matched and the reported parameter
counts include masked entries (trained entries are reported beside them); the corrected IOI
bystander's attribution agrees with exact patching at ρ = 0.67 only; k/v adapters span whole GQA groups; the fact circuit was found on
city → country and capital recall while most edits are office holders; no artifact is tied to a code
commit; small held-out safety sets; the Hub artifacts still need the sanitising republish. Honest verdict: a well-controlled, fully audited study
whose pre-registered results are negative or null and whose one positive result (LoRA on a faithful
circuit is as effective as unrestricted and about half as damaging to known non-English facts as a
count-matched random adapter) is small, post hoc and single-layer; strongest as a methods-and-audit
contribution with a clear statement of what a circuit buys and does not buy on this model; since A18
every LoRA number describes the adapter that was trained. The
corrected IOI bystander (A16) changed H4 from "consistent in direction" to inconclusive and nothing
else, which is itself a lesson about how fragile cross-circuit rank statistics are to one bystander.
