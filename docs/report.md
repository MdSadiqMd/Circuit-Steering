# Report (v0.21 — A22–A24: a refusal adapter that is better than the base, audited, with controls; after A21)

Status: **final after amendment A18 (`scripts/chains/13_a18.sh`, 2026-11-08/09, closes the fifth audit, `docs/audit-2026-11-08.md`); amendments A15–A17 were run and reported after the fourth audit (`docs/audit-2026-11-07.md`); the IOI bystander is the corrected `ioi_fixed` task everywhere L4 and H4 are quoted; every LoRA number describes the trained adapter (A18)**. Every number in §4 comes
from `audit/analyze.py` (`docs/results/hypotheses_*.md`, `*_cells.csv`), `aya_circuits.table`
(`docs/results/circuits.md`) or the `checks/results/*.json` files. Smoke-scale numbers (runs/smoke/) are never
reported. The B1 L4 values the third audit found inflated for discovery seed 0 (audit B1, A13) are
restated from the rescored edges; what changed is listed in §6 (v0.14).

"Verified" in this report means every audit layer was measured and reported with controls (brief §3.4).
It is necessary evidence, not a proof that nothing else changed: the audit can only see the behaviours,
languages and circuits it measures, with the sample sizes and judges stated in §5.

## 1. Question
Does confining steering, LoRA and knowledge editing to the components an EAP-IG circuit points to
change the target behaviour with less collateral damage than confining them to size-matched random,
inverse, Taylor-selected or unrestricted components, on `CohereLabs/tiny-aya-global`?

## 2. What was built (see docs/plan.md, docs/notes.md)
- Data: 1502 Wikidata facts with labels in en/hi/ar/fr/sw; 180 facts Tiny Aya answers wrongly in
  English (B1 targets, 137 of them current office holders; the brief asked for ≥ 200, see §5). Cyber-intrusion refusal category with
  near-miss benign prompts, translated with back-translation checks (B2).
- Circuits: EAP-IG node + signed edge scores, 3 seeds, for fact recall, refusal, IOI, language-ID.
- Interventions: steering (difference-of-means at nodes), head-masked LoRA, MEMIT with CPU float64
  statistics; conditions circuit / random (5) / inverse / Taylor / unrestricted; norm-matched random
  perturbation controls; for B2 the SafeTune refusal-direction steering baseline (`safetune_rd`,
  amendment A4), norm-matched to the circuit condition.
- Track repos chained (README "Track repos chained"): CircuitKIT (discovery, faithfulness, hooks,
  MEMIT, L4 re-attribution) → SafeTune (refusal-direction baseline; second refusal judge on every
  stored generation) → AuditKIT (every score, every paired comparison).
- Audit: L1 behavioural, L2 safety, L3 cross-lingual, L4 cross-circuit CAD + AS-NC + circuit distance
  with same-data repeat and resample floors, L5 selective-prediction metrics.

### 2.1 Why each model, dataset and metric (brief §3.3 item 7)
| choice | why |
|---|---|
| `CohereLabs/tiny-aya-global` | the only Cohere model that is instruction-tuned (refusal exists for B2), supported by CircuitKIT (cohere2), SafeTune and AuditKIT, and fits one 16 GB GPU / this 30 GB MPS budget for every stage; brief §4 |
| B1 facts from Wikidata (P36/P37/P38/P17/P35/P6) | functional relations with labels and aliases in all five languages, so the same triple is asked in parallel prompts; office holders (P35/P6) are facts the model is verifiably wrong about (June 2024 cutoff), i.e. honest "false facts" to correct |
| B2 cyber-intrusion (HarmBench) + XSTest near-miss | the category with the largest English-to-weakest-language refusal gap in the base model (headroom for strengthening), near-zero benign refusal (headroom for over-refusal), and prompts that are safe to publish |
| EAP-IG, node level, top 10 % | CircuitKIT's default and most stable discovery algorithm; node level is what the three interventions act on (heads, MLPs); 10 % is the size at which the fact circuit first recovers a third of the effect (size curve, §4.1) |
| patching + zero/mean ablation (signed) + same-path random + minimality | brief §5 and §12: sufficiency (Pillars 1–2), necessity (the set ablated and the rest kept, amendment A5), the random control and the minimality check; ablation signed under both replacements because the ratio depends on the replacement (Miller et al. 2024) |
| L1 neighbourhood / paraphrase / Global-MMLU / MGSM / IFEval / WikiText | locality on facts the model knows (the set is the first 60 known facts by key, which are all `capital` facts, so it is not a same-relation neighbourhood for the office-holder edits; audit B4), paraphrase generalisation of the edited facts, one knowledge, one reasoning and one instruction-following benchmark with parallel multilingual versions, and perplexity as the fluency floor |
| L1 one-hop ripple (A9) | questions whose answer follows from the corrected fact (capital/continent of a city's corrected country, inverse questions, the capital of the country an office holder leads), RippleEdits-style (Cohen et al. 2023); covers 174 of 180 facts |
| L2 held-out harmful, other-category harmful, near-miss benign, XSTest safe | refusal (efficacy and spill-over to other categories) and over-refusal on two benign sets, the pairing "Locate, Steer, and Improve" asks for |
| L3 = L1 + L2 in hi/ar/fr/sw | the brief's cross-lingual collateral layer; the five languages span scripts and resource levels, and the B1 labels exist in all of them |
| L4 cross-circuit CAD, AS-NC, node turnover, edge-level distance | Lexsi's CAD (Sadhu et al. 2026, Eq. 6, verified) extended to bystander circuits (novel contribution 1); AS-NC as a node-level variant of the paper's neuron-level activation shift (computed on English fact prompts for both behaviours); "circuit distance" in the tables is top-k node turnover (1 − Jaccard), not Chen et al.'s edge-level circuit distance, whose analogue is `l4_target_edge_l1` (Σ|Δs_e| / Σ|s_e| over every edge of the target graph; audit C1) |
| L5 evidence arbitration (A12, `audit/arbitration.py`) | Anand 2026: memory against an injected passage (agree / update / stale / authority), spread = sd of the log-odds A, AURC over decisions ranked by |A|, error of the most confident quarter; 138 never-edited facts balanced over six relations. The earlier context-free margin family (`hidden`) stays reported as a margin, not as arbitration (audit B3) |
| L5 relearning probe for B2 (A10) | how much strengthened refusal survives a 40-step benign fine-tune (Qi et al. 2024), the persistence question Guo et al. 2025 ask of removed knowledge, without training on any harmful completion |
| paired percentile bootstrap, 3 seeds, 5 random draws | brief §12.6; AuditKIT's `paired_bootstrap` supplies the paired deltas, its p-value is not a null test (docs/notes.md), so CIs are reported instead |

## 3. Pre-registered hypotheses
See docs/preregistration.md (frozen before the first cell).

## 4. Results

### 4.1 Circuits (final; `docs/results/circuits.md`, `runs/circuits/`)
All numbers are signed raw ratios (metric − corrupt)/(clean − corrupt) at 10 % of the 612 heads+MLPs
(62 nodes incl. input), 96 examples for the pillars and 64 for the same-path controls (5 random draws).

| circuit | patching (seeds) | random node set | minus 20 % of nodes | zero-abl. circuit / random | stability Jaccard |
|---|---|---|---|---|---|
| fact recall en (`b1_en`) | 0.33 / 0.56 / 0.36 | 0.000 | 0.09 / 0.11 / 0.07 | 0.50/0.49 · 0.51/0.49 · 0.52/0.50 | 0.83 / 0.81 / 0.81 |
| fact recall hi | 0.82 / 0.61 / 0.55 | ≤ 0.001 | 0.19 / 0.11 / 0.06 | 0.54/0.49 · 0.53/0.51 · 0.51/0.47 | 0.84 / 0.82 / 0.82 |
| fact recall ar | 0.92 / 0.64 / 0.67 | ≤ 0.002 | 0.15 / 0.11 / 0.11 | 0.48/0.46 · 0.51/0.47 · 0.56/0.51 | 0.82 / 0.82 / 0.77 |
| fact recall fr | 0.60 / 0.84 / 0.81 | ≤ 0.002 | 0.09 / 0.18 / 0.22 | 0.58/0.49 · 0.49/0.49 · 0.52/0.50 | 0.83 / 0.82 / 0.82 |
| fact recall sw | 0.17 / 0.43 / 0.36 | ≤ 0.002 | 0.04 / 0.11 / 0.09 | 0.48/0.45 · 0.50/0.50 · 0.50/0.52 | 0.90 / 0.88 / 0.84 |
| refusal en (`b2_en`) | −0.04 / −0.03 / −0.04 | +0.005 | −0.01 / −0.02 / −0.03 | −0.54/0.02 · −0.25/0.03 · +0.44/0.03 | 0.83 / 0.83 / 0.82 |
| IOI | 0.11 / 0.02 / 0.20 | 0.28 / 0.30 / 0.24 | 0.04 / 0.08 / 0.12 | 0.15/0.52 · −0.29/0.38 · −0.09/0.48 | 0.71 / 0.72 / 0.66 |
| language-ID en | 0.66 / 0.35 / 0.35 | 0.000 | 0.18 / 0.08 / 0.08 | 0.54/0.50 · 0.55/0.53 · 0.48/0.49 | 0.78 / 0.77 / 0.78 |

Seeds 1–2 of the non-English fact circuits and the stability pillar of every circuit that lacked it were
added by A11 (stability = CircuitKIT's pillar, a Jaccard over resampled discoveries; the cross-seed
top-10 % Jaccard is a different quantity, listed in `circuits.md`). The per-type evaluation below was not
run for the new non-English seeds: those circuits are never intervened on.

**The pre-registered circuit itself (amendment A5; `aya_circuits/cellset.py`, second table of
`circuits.md`).** The table above is CircuitKIT's joint-ranked set (top 61 of 612 by its absolute node
scores, 21 MLPs for b1_en). The cells intervene on the pre-registered per-type set (top 10 % of heads and
of MLPs separately: 58 heads + 4 MLPs; Jaccard with the joint set 0.45–0.73). Evaluated through the same
CircuitKIT path, with 5 per-type random sets of the same size, and with *necessity* (the set ablated under
mean replacement, the rest kept; 1 = no loss, 0.5 = a symmetric logit difference driven to 0):

| circuit | per-type set: patching (seeds) | random per-type sets | necessity: set ablated (same call) / random sets ablated | joint set: necessity |
|---|---|---|---|---|
| fact recall en | 0.019 / 0.026 / 0.021 | 0.000 | 0.52 / 0.52 / 0.49 vs 0.65 / 0.66 / 0.64 | 0.51 / 0.50 / 0.49 |
| fact recall hi · ar · fr · sw (seed 0) | 0.004 · 0.004 · 0.016 · 0.017 | 0.000 | 0.49 · 0.49 · 0.49 · 0.43 vs 0.57 · 0.51 · 0.61 · 0.53 | 0.49 · 0.47 · 0.46 · 0.49 |
| refusal en | −0.001 / −0.002 / −0.005 | +0.002 | 0.48 / 0.50 / 0.53 vs 0.38 / 0.40 / 0.33 | 0.25 / 0.19 / −0.04 |
| IOI | 0.39 / 0.47 / 0.35 | 0.21 / 0.18 / 0.16 | 0.07 / 0.07 / 0.15 vs 0.35 / 0.27 / 0.40 | −0.05 / 0.06 / 0.16 |
| language-ID en | 0.033 / 0.022 / 0.025 | 0.001 / 0.000 / 0.001 | 0.48 / 0.54 / 0.46 vs 0.61 / 0.69 / 0.62 | 0.47 / 0.52 / 0.50 |

Findings, stated as measured:
- **The set the cells intervene on is not the set that recovers fact recall.** Capping the MLPs at
  four removes the MLPs that carry recall (MLP 6, 28, 27, 4, 1, 31, 29 rank 5–12 by attribution and
  are outside the per-type set): the per-type fact set restores 0.4–3 % of the logit difference under
  patching against 33–56 % for the joint set, above the per-type random sets (exactly 0) but far from
  faithful. Ablating it (mean replacement) removes the whole clean logit difference: re-referenced to
  the exact floor of each task's ratio (r0 = −corrupt/(clean − corrupt), 0.47–0.51 for the fact
  circuits, `checks/results/metric_span.json`; fourth audit B.1), the metric left after ablating the set is
  −8 % to +3 % of the clean value on every fact circuit and seed, while ablating 58 random heads + 4
  random MLPs leaves 7–30 % (b1_en 26–30 %). The per-type fact set is necessary and not sufficient
  (necessity column `nec0_*` in `circuits.md`).
  The refusal per-type set is neither: patching ≈ 0, and ablating it leaves 38–45 % of the refusal
  logit difference (r0 = 0.15 for this task) against 22–29 % after random per-type sets, whereas
  the joint refusal set is necessary (4 %, 12 % and −22 % left) without being sufficient. IOI is the
  one circuit whose per-type set is the better one (0.35–0.47 vs 0.02–0.20), and it is necessary
  (4–13 % left vs 25–39 % after random sets; r0 = 0.02), but per-type random sets already restore
  0.16–0.21 on IOI. Every locality result in §4.3–4.4 is therefore a result about interventions on
  node sets that do not reproduce the behaviour on their own. H3's gate fires for all of them only with
  amendment A5's added criterion (patching < 0.1); under the gate as pre-registered the B1 per-type
  sets pass it (see H3).
- **Fact recall is localised (joint-ranked set).** The 10 % EAP-IG circuit restores 33–56 % of the English logit
  difference under patching while five size-matched random node sets restore exactly nothing
  (patched metric equals the corrupt metric to bf16 resolution). Removing a random 20 % of the
  circuit's own nodes drops patching to 4–19 %, so the effect is concentrated in a few nodes
  (`MLP 0` carries the largest score, 3.45 vs 0.74 for the next node). Cross-seed top-10 % Jaccard
  0.85–0.91. The non-English fact circuits overlap the English one strongly (weighted Jaccard
  0.53–0.79, top-10 % Jaccard 0.79–0.85).
- **Zero-ablation is floored and uninformative here.** Ablating 90 % of the nodes drives the
  logit difference to ≈0 on every circuit, and with a near-symmetric clean/corrupt metric the ratio of a
  destroyed model is 0.5. Random node sets score 0.45–0.53 under zero ablation, the circuits 0.48–0.58
  (fact and language-ID circuits): the margin is ≤ 0.10 everywhere. The floor's exact value per task is
  r0 = −corrupt/(clean − corrupt) (0.509 b1_en, 0.47–0.49 other fact circuits and language-ID, 0.15
  refusal, 0.02 IOI); every CircuitKIT ratio is a bf16 mean, resolved to 0.002–0.006 of the span, so
  two-decimal values are at resolution (fourth audit B.1–B.2). The pre-registered P2 gate (raw ratio ≤ 0) can therefore never fire
  for the fact circuits; mean ablation behaves the same way. We report the margin over random sets
  instead and treat patching + the two controls as the informative pillars.
- **The refusal circuit is not faithful at 10 %.** Patching −0.04 (random +0.005), minimality
  indistinguishable from zero, zero-ablation sign flips across seeds (−0.54, −0.25, +0.44 in the
  64-example control run; −0.50, −0.24, +0.45 in the 96-example pillar run). The size
  curve reaches 0.45–0.58 only at 30 % of the nodes. By the pre-registered rule (P2 or P5 failure ⇒ no
  locality advantage predicted) every B2 cell is gated: H3 predicts the circuit condition should not
  beat random on B2, and the B2 results below are read as a test of that prediction. *Not a pair-
  alignment artefact (A16, `b2m_en`):* restricting the EAP data to the 162 token-length-matched pairs
  gives the same circuit (weighted Jaccard 0.80, top-10 % Jaccard 0.79 with `b2_en`; cross-seed
  0.85–0.88) with the same unfaithfulness: patching −0.004 / +0.003 / −0.003 at 10 %, zero-ablation
  +0.55 / −0.21 / +0.18, 0.32–0.39 at 30 %. The 45 misaligned pairs explain nothing; refusal on this
  model is not a sparse node circuit at 10 % of the components.
- **IOI fails on this model — on CircuitKIT's built-in task, which is mis-specified for this
  tokenizer (amendment A16).** The built-in EAP path serialises the sentence *with* the indirect object,
  prepends a second BOS and reads its metric at the answer token (`checks/check_ioi_position.py`:
  metric position = the IO name on 100 % of pairs, clean logit difference 2.0 there against 11.2 at
  the "to" position; same defects on GPT-2), and 229 of the first 404 pairs differ in token count. The
  numbers below describe that task; `ioi_fixed` (same generator, prompt truncated before the IO, one
  BOS, length-matched single-token pairs) replaces it as the bystander. *`ioi_fixed`, 3 seeds (A16;
  `circuits.md`):* at 10 % the joint-ranked circuit restores 0.054 / 0.020 / 0.039 of the logit
  difference, the per-type set 0.031 / 0.016 / 0.028, and every same-path random node set exactly 0.000
  (the "random beats circuit" artefact of the built-in task is gone); the size curve rises steeply
  afterwards (0.21 / 0.31 / 0.35 at 20 %, 0.57 / 0.43 / 0.55 at 30 %), the stability pillar is 0.86 on
  all three seeds (built-in task 0.66–0.72), and the set is necessary: ablating either the per-type or
  the joint set under mean replacement leaves −0.05–0.02 of the ratio (random per-type sets 0.67–1.1 on
  three draws of five, i.e. no loss, and ≈ 0 on two). So on this model the IOI mechanism at 10 % of the
  nodes is a necessary but far from sufficient set, with the behaviour spread over about 30 % of the
  components; it is a legitimate bystander whose own faithfulness at 10 % is low, which the H4 reading
  must carry. As a bystander it is the least disturbed circuit (mean raw CAD 0.069 on B1, 0.076 on
  B2) and has the lowest resample floor (0.328); its H4 term is in §4.3 and §4.4. As measured on
  the built-in task: the EAP-IG IOI circuit restores 2–20 % while random node sets restore
  24–30 %; its stability is the lowest of all circuits (pillar 0.66–0.72, cross-seed top-10 % Jaccard
  0.63–0.69), and its size curve is not monotone: the top 1 % and 2 % of nodes restore 0.30 / 0.37 / 0.26
  (identical values at 1 % and 2 % on two seeds: the 2 % set is a strict superset whose six added
  nodes all reach the logits and each moves patching by −0.03 to +0.03 alone, but together by less
  than one bf16 quantum of the mean metric, `checks/results/ioi_plateau.json`; fourth audit C.1), 10 % restores
  0.05 / 0.00 / 0.18 (the curve runs draw their own examples, so their 10 % point differs from the
  pillar value by up to 0.07). The cause is in the task, not the model: under this tokenizer 14 of 99
  IOI names are multi-token and CircuitKIT's own tokeniser reports unequal clean/corrupt token counts for
  most pairs (229 of the first 404), which EAP end-pads, so attribution compares shifted positions
  (fourth audit C.2). IOI is kept only as a bystander whose edge scores are recomputed in L4; its CAD
  values are reported but carry little meaning.
- **Non-English fact circuits (3 seeds).** Patching at 10 %: hi 0.55–0.82, ar 0.64–0.92, fr 0.60–0.84,
  sw 0.17–0.43, random node sets ≤ 0.002 everywhere; Swahili is the weakest on every seed and its seed-0
  circuit is barely above its minimality control (0.17 vs 0.04).
- **Overlaps (seed 0, weighted Jaccard on |signed node scores|, `input` pseudo-node excluded):** fact–refusal
  0.30, fact–language-ID 0.49, fact–IOI 0.10, refusal–language-ID 0.34, refusal–IOI 0.09. With the
  corrected IOI task (A16) the IOI overlaps rise to fact–`ioi_fixed` 0.23 and refusal–`ioi_fixed` 0.20
  (top-10 % Jaccard 0.26 / 0.18; built-in IOI vs `ioi_fixed` 0.26) and the ranking of the three
  bystanders is unchanged for both behaviours (language-ID > the other target > IOI). H4 is tested on
  these three distinct overlap values per behaviour.
- Node and edge attribution agree (Spearman of |edge row sums| vs node scores 0.968–0.999 for all 24
  discoveries, including seeds 1–2 of the non-English fact circuits added by A11).
- **Area under the faithfulness-vs-size curve** (a partial-range, node-level analogue of MIB's CPR,
  which integrates over the full edge fraction 0–1 against the empty circuit; `docs/results/circuits.md`): patching AUC
  over 1–30 % of the nodes is 0.54 / 0.58 / 0.58 for the fact circuit (seeds 0/1/2) and 0.12 / 0.12 /
  0.12 for the refusal circuit; zero-ablation AUC sits on the 0.5 floor for the fact circuit (0.52 /
  0.52 / 0.52) and swings with the sign flips for refusal (−0.22 / 0.16 / 0.54). For the circuits added
  by A11: language-ID 0.65 / 0.59 / 0.55, IOI 0.21 / 0.38 / 0.29, hi 0.72 / 0.53 / 0.53, ar 0.76 / 0.68 /
  0.67, fr 0.67 / 0.73 / 0.70, sw 0.41 / 0.54 / 0.51.
- **EAP-IG vs true activation patching** (brief §12.3, `checks/results/eap_vs_patching.json`, seed 0, 32
  examples, top-50 nodes by |score|): Spearman between the signed EAP-IG node score and the measured
  loss change when the node's activation is replaced by its corrupted-run value is 0.92 (fact recall),
  0.81 (refusal), 0.95 (language-ID), 0.73 (built-in IOI) and 0.67 (`ioi_fixed`, run in the A18
  chain); on magnitudes 0.91 / 0.74 / 0.90 / 0.65 / 0.45; sign agreement 0.90–0.94. The four original
  tasks pass the brief's ρ ≥ 0.7 rule on signed scores; `ioi_fixed`, the bystander every L4 and H4
  number now uses, does not (0.67 signed, 0.45 on magnitudes), so its node ranking is the least
  trustworthy of the study, which weakens its bystander CAD and H4 further (§5). The brief's remedy
  (more integration steps or a corrected attribution) is the subject of the queued amendment A19. The scales are not
  comparable: with a batch-mean metric CircuitKIT's `attribute` returns mean attribution / batch size
  (audit B1), so the fact circuit's MLP 0 score 3.1 at batch 4 corresponds to 12.4 against a measured
  19.7; the ranking, not the scale, is what the circuits should be read from.

### 4.2 Intervention cells (final counts; `runs/interventions/`)
Per behaviour: steering 17 main cells + 68 sweep cells, LoRA 17, MEMIT 20 (incl. `circuit_deep`; B1 23 with
the three A8 `causal_trace` cells), plus
one norm-matched random perturbation control per weight cell. Steering is norm-matched (|Δ| 8.65 for B1 after A6,
24.5 for B2 at the 0.5× main coefficient). MEMIT writes at `MLP 0` for `circuit` (|Δ| 125–136 B1, 77–78
B2), at `MLP 5` (B1) / `MLP 19` (B2) for `circuit_deep`, at `MLP 16` (B1) / `MLP 27` (B2) for Taylor, and
across all 29 editable layers for unrestricted (|Δ| 36 B1, 26 B2); 49–60 of 60 facts (B1) and 20–28 of 28
prompts (B2) count as successes by CircuitKIT's own criterion, which is any increase of the target
  probability (`memit_wrapper`: `confidence_increase > 0`), not the target becoming the top-1
  continuation: the layer-0 edits raise the mean target probability from 0.003–0.005 to 0.08–0.12
  across the three seeds, and the
  audited greedy efficacy is 0.37 (§4.3). Amendment A8 adds three B1 `causal_trace` cells (+ their
  perturbation controls) at the layer English causal tracing selects (`interventions/causal_trace.py`,
  Meng et al. 2022: 100 known facts, 5 noise draws at 3σ of the embedding scale, MLP restoration over a
  10-layer window; clean target probability 0.88, corrupted 0.013). The windowed indirect effect is flat
  over windows starting at layers 0–11 (0.71–0.77; argmax layer 3, layers 2–7 within 0.01) and near zero
  after layer 15; single-layer restoration peaks at MLP 0 (0.31) and MLP 5 (0.31). MEMIT at MLP 3 needs
  |Δθ| 68–70 and inserts 59–60 of 60 facts (mean target probability 0.003–0.005 → 0.45–0.51).
- **ROME on Tiny Aya (brief §7.4, `checks/results/rome_smoke.json`).** CircuitKIT's circuit-guided entry point
  (`CircuitKnowledgeEditor.edit_via_circuit(method="rome")`) cannot take a circuit loaded with
  `ck.load_scores`: it looks for `.nodes`/`.edges`, finds none, selects no layer and fails on
  `None < 0` (filed in the JSON; the editor also clones every parameter, which in fp32 exceeds the MPS
  cap, so the check runs on CPU). CircuitKIT's `RomeHandler` called directly at MLP 5 edits one fact
  (Paraguay's head of government → Santiago Peña): target first-token probability 1.7e−5 → 0.993,
  greedy reply "María Ñánder" → "Santiago Peña", |ΔW_out| 4.3, weights restored exactly. ROME works on
  cohere2; the study's edits remain MEMIT batch edits.

- **How LoRA cells are applied (amendment A18).** Every LoRA cell (and its perturbation control) is
  applied as forward hooks that add the fp32 low-rank term scaling·B·A·x to the bf16 module output with
  one rounding, on the HF model and on the TransformerLens model alike: the forward PEFT used during
  training (adapters autocast to fp32). Until the fifth audit the cells were merged into the bf16
  weights, which rounded 31–43 % of the entries of ΔW to zero (§5 "A18"). `checks/results/apply_equivalence.json`:
  the hook reproduces PEFT's forward bit for bit on all 96 adapted modules of a real cell, the merged
  weights differ on 24 % of their output entries, and HF and TL hooks agree on real cells (logit-delta
  cosine 0.997–0.9994). All 79 LoRA cells of both behaviours were re-audited on every layer.

### 4.3 Audit and hypotheses: B1 false-fact correction (final: 231 cells incl. sweep and A15 cells, all five layers)
Source: `docs/results/hypotheses_b1.md`, `docs/results/b1_cells.csv`. Efficacy = accuracy gain on the seed's 60 edited facts
(base 0.0 by construction). Δ = circuit − comparison, percentile-bootstrap 95 % CI.

| intervention · condition | efficacy | |Δθ| | L3 mean abs Δ (hi/ar/fr/sw facts, benign and other-category refusal) | L1 neighbourhood Δ (en) | never-edited top-1 error Δ | WikiText ppl ratio |
|---|---|---|---|---|---|---|
| MEMIT · circuit (MLP 0) | 0.37 | 131 | 0.197 | −0.33 | +0.32 | 1.010 |
| MEMIT · circuit_deep (MLP 5) | 0.88 | 61 | 0.160 | −0.39 | +0.38 | 1.004 |
| MEMIT · causal_trace (MLP 3, A8) | 0.81 | 69 | 0.099 | −0.32 | +0.26 | 1.011 |
| MEMIT · Taylor (MLP 16) | 0.65 | 44 | 0.101 | −0.26 | +0.63 | 1.002 |
| MEMIT · inverse (MLP 13–18) | 0.69 | 53 | 0.106 | −0.18 | +0.49 | 1.002 |
| MEMIT · random (5 draws, one random layer each) | 0.57 | 40 | 0.069 | −0.19 | +0.36 | 1.003 |
| MEMIT · unrestricted (29 layers) | 0.86 | 36 | 0.075 | −0.15 | +0.24 | 1.001 |
| MEMIT · perturb_circuit (random ΔW, |Δθ| 131 at MLP 0) | 0.01 | 131 | 0.391 | −0.47 | +0.40 | 1.070 |
| LoRA · circuit | 0.41 | 3.3 | 0.030 | −0.03 | +0.03 | 0.996 |
| LoRA · inverse | 0.44 | 3.3 | 0.031 | −0.09 | +0.09 | 0.996 |
| LoRA · Taylor | 0.49 | 3.2 | 0.035 | −0.10 | +0.09 | 1.003 |
| LoRA · random (5 draws) | 0.64 | 3.6 | 0.032 | −0.03 | +0.04 | 1.011 |
| LoRA · unrestricted (all heads) | 0.99 | 6.3 | 0.043 | −0.02 | +0.05 | 1.019 |
| LoRA · perturb_circuit | 0.01 | 3.3 | 0.005 | 0.00 | 0.00 | 1.000 |
| steering · circuit (0.5×) | 0.04 | 8.6 | 0.032 | 0.00 | +0.01 | 1.003 |
| steering · unrestricted (0.5×) | 0.02 | 8.6 | 0.018 | 0.00 | 0.00 | 1.014 |

- **H1 (efficacy parity).** MEMIT: refuted, circuit/unrestricted = 0.43 [0.35, 0.49]; the layer-0 site
  corrects 37 % of the facts against 86 % for the 29-layer edit. The exploratory `circuit_deep` site
  (MLP 5, the circuit's strongest MLP outside the detokenisation block) reaches 0.88, i.e. parity. LoRA:
  refuted, circuit/unrestricted = 0.41 [0.36, 0.47] (0.41 vs 0.99); the size-matched random head sets
  reach 0.64, i.e. the circuit's heads are a *worse* place to put a fixed LoRA budget than random heads
  of the same count (with ~22 % more adapter entries and ~18 % more trained entries, §5). Steering: vacuous, every
  condition stays at or below 0.13 at every coefficient, so difference-of-means steering between known and
  wrongly answered facts does not correct specific facts.
- **H2 (locality advantage), behavioural L3.** MEMIT at the circuit has *more* cross-lingual
  collateral than every comparison: circuit − random Δ = +0.134 [+0.096, +0.174], circuit − inverse
  +0.091 [+0.046, +0.138], circuit − Taylor +0.095, circuit − unrestricted +0.122 (all CIs exclude 0,
  n = 36–60 seed × language × family pairs). The norm-matched random perturbation at the same site is
  worse still (0.39, with WikiText +7 %), so at MLP 0 a change of the size this site needs (|Δθ| 131
  vs 36–69 elsewhere) is damaging whatever is written. That does not make size the general explanation:
  at every other site the real edit causes 4–12× the L3 collateral of equal-norm noise (0.07–0.16 vs
  0.013–0.023) and 15–39 points of neighbourhood loss against ≤ 2 for the noise, so there the
  collateral comes from what was written. Site and size are confounded at MLP 0 (no norm-capped edit
  there, no 131-norm edit elsewhere). For steering the
  differences are at the resolution of the metrics (circuit − random +0.013 [+0.007, +0.020] on a
  0.02–0.03 scale). LoRA is null on L3: circuit − random −0.002 [−0.011, +0.007], circuit − inverse
  −0.003 [−0.019, +0.012]; LoRA on the circuit's heads does beat the unrestricted adapter (−0.013
  [−0.027, −0.000]), which has 4.1–4.3× the adapter entries (15.1 M vs 3.5–3.7 M) and 5.1–5.4× the
  entries that actually train (15.1 M vs 2.8–3.0 M; §5 "Fifth audit").
- **H5 (cross-lingual propagation).** Refuted for the pre-registered site against both comparison
  sites: edited-fact gain in hi/ar/fr/sw is 0.01/−0.03/0.09/0.08 for MLP 0 versus 0.08/0.08/0.39/0.24
  for the Taylor layer (circuit − Taylor Δ = −0.16 [−0.22, −0.11]) and 0.15/0.12/0.47/0.29 for the
  causal-tracing layer (A8; circuit − causal_trace Δ = −0.22 [−0.28, −0.16], seed-cluster [−0.26,
  −0.19]); neighbourhood drops are not smaller (circuit − Taylor −0.06 [−0.16, +0.02], circuit −
  causal_trace −0.07 [−0.17, +0.00]). `circuit_deep` propagates about as well as the causal-tracing site
  (0.14/0.12/0.57/0.41) with the same neighbourhood damage as the other single-layer edits.
- **Collateral that no single metric would have shown.** Every single-layer MEMIT batch edit of 60
  facts raises the never-edited top-1 error of the base model from 0.02 to 0.28–0.65 (0.26 for the
  29-layer edit) and the English neighbourhood accuracy falls 18–39 points (15 for the 29-layer edit;
  47 for the MLP-0 perturbation control), while WikiText perplexity moves by ≤ 1 % and MMLU by
  ≤ 6 points: perplexity and MMLU are blind to the damage. Every pre-registered MEMIT condition lowers English
  refusal of held-out harmful prompts (mean Δ −0.05 to −0.14; the layer-0 edit −0.14, its perturbation control
  −0.22; the exploratory `circuit_deep` site leaves it at 0.00 and the causal-tracing site at −0.03), the only B1 cell
  families that move safety by more than one prompt; LoRA and steering change held-out English refusal by at
  most −0.033 (random steering vectors; −0.028 inverse, −0.017 random LoRA; 0 for the circuit cells). LoRA on the same circuit nodes
  leaves every collateral metric in the table within 0.03 of the base; its one larger move is other-category
  refusal (−0.075, three of 40 prompts).
- **Steering sweep at matched efficacy (cells rebuilt with fold-disjoint sources, amendment A6).** The
  largest efficacy any steering curve reaches is 0.13, and the common target e* = 0.022 is at noise
  level; at that point the circuit vectors cost the least perplexity (1.00 vs 1.02–1.24) because they
  are the smallest coefficients. Collateral grows with the coefficient (WikiText ×1.1–1.8 at 2×)
  without efficacy following. The superseded cells (sources overlapping the efficacy fold) gave the
  same picture (max 0.12, e* = 0.028, ×1.2–2.3 at 2×).
- **H2, mechanistic L4 (cross-circuit CAD on the refusal, IOI and language-ID circuits; 32 examples, 3
  IG steps, batch 2; every mask is the pre-registered per-type set since amendment A5; the IOI
  bystander is the corrected `ioi_fixed` task since amendment A16, with the built-in-task values kept
  in `b1_cells.csv` as `*_ioi`; the joint-ranked values are the `*_joint` fields and the per-cell
  Spearman between the two definitions is 0.958).** Restated in v0.14 after the A13 rescore of discovery
  seed 0 (base edges computed at batch 4) and in v0.16 for the corrected IOI bystander; per-seed means
  agree (bystander CAD of the real MEMIT / LoRA / steering cells 0.147 / 0.137 / 0.151, 0.118 / 0.135 /
  0.115, 0.072 / 0.072 / 0.078 for seeds 0 / 1 / 2). CAD is reported raw and as the excess over the
  same-settings resample floor (base vs base on another example subset; 0.33 (`ioi_fixed`), 0.39
  (refusal) and 0.41 (language-ID) for the bystanders, 0.44–0.58 for the fact circuit). Only the layer-0
  MEMIT cells and their perturbation control exceed that floor on bystanders (excess +0.08 and +0.43;
  joint +0.12); every other condition of every intervention sits *below* it (real conditions −0.21 to
  −0.33, their perturbation controls −0.27 to −0.36; joint −0.23 to −0.39), i.e. its bystander edge
  changes on these 32 examples are smaller than the change that resampling the examples causes in the
  base attribution. Paired tests read the same way under both definitions (`hypotheses_b1.md`): MEMIT circuit −
  random +0.40 [+0.34, +0.46], circuit − inverse +0.35 [+0.28, +0.42], circuit − Taylor +0.37, circuit −
  unrestricted +0.36, circuit − causal_trace +0.35 (all CIs and seed-cluster CIs exclude 0: H2 refuted
  in the wrong direction, consistent with L3); LoRA circuit − random +0.012 [−0.007, +0.031]
  (seed-cluster [+0.008, +0.026]), circuit − inverse −0.012 [−0.061, +0.027] (null); steering circuit −
  random +0.020 [+0.008, +0.032] and circuit − inverse −0.055 [−0.073, −0.039] (mixed, and all below
  floor). **Target-circuit CAD does not exceed its floor for any real intervention** (v0.13 said it did
  for most weight cells; that reading came from the inflated seed 0): MEMIT −0.01 (MLP 0) to −0.19, LoRA
  −0.05 to −0.40 (unrestricted −0.13), steering −0.42 to −0.47; only the MLP-0 perturbation control is above it
  (+0.08). The b1_en floor is high (0.44–0.58; base-vs-base edge Spearman 0.35–0.39), so the fact
  circuit's edge scores are themselves unstable across example subsets. Raw target CAD still orders real
  interventions above their norm-matched controls (MEMIT 0.31–0.50 vs 0.04–0.21 except the MLP-0 control
  at 0.59; LoRA 0.17–0.53 vs 0.01–0.03; steering 0.04–0.09), so the interventions move the fact circuit's
  edges more than equal-norm noise does, by less than resampling the attribution examples moves them.
- **L4 power check (`audit/l4_power.py`, `runs/audit/b1/power/l4_power.json`): under-sampled, not
  blind.** Base, repeat and resample edges and four seed-0 cells were re-attributed at 64 examples and
  10 IG steps (the study uses 32 / 3; 1060–1100 s per cell against ~165 s). The resample floor roughly
  halves: b1_en 0.507 → 0.273 on the per-type mask (base-vs-base edge Spearman 0.35–0.39 → 0.51), b2_en
  0.392 → 0.221, IOI 0.483 → 0.311, language-ID 0.430 → 0.274 (0.54–0.64× the 32 / 3 value); the
  repeat floor stays exactly 0. The cells' CADs fall less than the floors: on the target circuit to
  0.38–0.75× (LoRA at the circuit 0.366 → 0.140, excess −0.141 → −0.133; steering 0.086 → 0.046,
  −0.421 → −0.227; unrestricted LoRA 0.339 → 0.254, −0.168 → −0.019; MEMIT at the Taylor layer 0.372 →
  0.259, −0.135 → −0.014) and on the bystanders only to 0.71–1.08× (language-ID: unrestricted adapter
  0.307 → 0.251, excess −0.122 → −0.023). The check changes the number of examples and the number of IG
  steps together, so it cannot say how much of a 32 / 3 CAD is IG discretisation and how much is the
  finite example set. So a 7× larger attribution budget brings the strongest weight interventions to
  within 0.02 of the floor on the per-type target mask (−0.019, −0.014; language-ID −0.023) while LoRA
  at the circuit and steering stay 0.12–0.24 below it on every per-type mask, with the ordering of
  conditions preserved. On the joint-ranked masks the unrestricted adapter (+0.020 target, +0.044
  language-ID) and the Taylor-layer MEMIT (+0.035 target) are *above* the 64 / 10 floor, so "no real B1
  intervention exceeds the floor" holds for the 32 / 3 budget only. The check used the built-in IOI
  task, not `ioi_fixed`. The L4 readings above are therefore limited by attribution budget, and a full
  re-attribution at 64 / 10 (about 7× the L4 campaign, ≈ 60 h here) is the obvious next step; it was
  not run.
- **H3 (faithfulness gates locality).** On the joint-ranked set no B1 circuit fails the P2/P5 gate
  (patching 0.33–0.56, random-node margin 0.34–0.58), which is how v0.7 read it. On the per-type set
  the cells actually intervene on, the gate *as pre-registered* (P2 raw ratio ≤ 0 under either
  replacement, or no P5 margin over random) is also passed: zero-ablation 0.53–0.57, mean 0.49–0.50,
  random margin +0.020 to +0.025. Only amendment A5's added criterion (patching < 0.1; per-type
  patching is 0.019 / 0.026 / 0.021) gates them, and A5 wrongly called that "no rule change" (A13).
  Read with the original gate, nine B1 circuit cells passed it and none shows an L4 locality advantage
  (all 9 negative after the rescore: steering −0.01 to −0.04, LoRA −0.01 to −0.03, MEMIT −0.33 to
  −0.50; one L3 advantage, LoRA seed 1 +0.012): the circuit passing P2/P5 did not buy locality. Read
  with the A5 gate, all nine are gated and the prediction (no advantage) holds for 9 of 9. Across the
  9 cells the Spearman between any pillar of either set and the H2 advantage lies between −0.26 and
  +0.26 (p ≥ 0.49; only 3 distinct pillar values per behaviour). The pooled test with B2 is in §4.4.
- **A15: cells on the faithful joint-ranked set (post hoc; `hypotheses_b1.md` "A15").** The set whose
  pillars the first table of §4.1 reports (40 heads + 21 MLPs for b1_en; patching 0.33 / 0.56 / 0.36,
  random-node margin 0.34 / 0.58 / 0.38, necessity 0.49–0.52) was intervened on with LoRA and steering,
  against five random sets and one inverse set with the same head and MLP counts (`random_joint`,
  `inverse_joint`; the inverse set necessarily shares 6 / 6 / 8 MLPs with the circuit, a random one
  12–19 nodes; the "overlap with circuit" column of the `hypotheses_b1.md` A15 tables counts overlap
  with the *per-type* circuit, not with `circuit_joint`). *LoRA, H1 (A15): parity.* `circuit_joint`
  corrects 1.00 / 0.98 / 1.00 of its fold against
  0.99 for the unrestricted adapter (ratio 1.000 [1.000, 1.000]) with 8.4–8.8 M adapter entries (7.8–8.2 M
  of them trained; unrestricted 15.1 M); but `random_joint` (0.97–1.00, 9.0–9.4 M) and `inverse_joint` (0.97–1.00,
  8.8–9.1 M) reach the same, so parity comes from adapting 21 MLP layers, not from which ones: across
  the 28 non-control B1 LoRA cells efficacy tracks adapter size (Spearman 0.85 on the reported adapter
  entries, 0.86 on the trained entries), and the
  per-type set's 0.41 (3.5 M, 4 MLPs) is the capacity story of §4.5. The joint-set adapters also carry
  the corrected facts into hi/ar/fr/sw (+0.45 edited-fact gain, random_joint +0.45, unrestricted +0.48)
  where the per-type circuit adapter gives +0.08 and per-type random +0.18. *LoRA, H2 (A15), the one
  locality advantage in the study, now below the Holm line.* L3: `circuit_joint` − `random_joint` =
  −0.008 [−0.017, −0.000] (seed-cluster [−0.013, +0.000]; bootstrap p = 0.047, which no longer survives
  Holm over the 20 B1 tests; with the bf16-merged adapters of v0.17 it had read −0.011 [−0.020, −0.003],
  p = 0.002), driven by the non-English neighbourhood (known-fact accuracy −0.024 vs −0.056; inverse
  −0.062, unrestricted −0.064; the refusal families differ by ≤ 0.02); − `inverse_joint` −0.003 [−0.016,
  +0.009] (null); − unrestricted −0.014 [−0.026, −0.003]. The statistical unit behind it is three seeds:
  the per-seed differences are −0.013 / −0.007 / +0.000, a paired t-test over the seeds gives p = 0.21
  and a sign test cannot go below p = 0.25; the bootstrap p treats the 60 language × family pairs as
  independent, and the 20-test Holm family includes the 8 post-hoc A15 tests. Because the comparison
  with `inverse_joint` is null, the pre-registered H2 rule (circuit < random *and* circuit < inverse) is
  not met. Other collateral points the same way (English neighbourhood −0.011 vs −0.030, benign refusal
  +0.01 vs +0.02, WikiText 1.016 vs 1.022). L4 (per-type bystander masks, `ioi_fixed` bystander):
  − `random_joint` −0.019 [−0.046, +0.005] (seed-cluster [−0.035, −0.006]; p = 0.13, does not survive
  Holm), − `inverse_joint` −0.023 [−0.066, +0.017], − unrestricted −0.043 [−0.071, −0.019]; all below
  the resample floor. Against the pre-registered conditions the joint circuit equals the per-type circuit
  on L3 (−0.001) and L4 (+0.011) and disturbs bystanders *more* than the small per-type random sets
  (+0.022 [+0.008, +0.038]): a larger adapter moves more edges. Within the 11 joint cells the adapter
  size does not track L3 collateral (Spearman +0.14 on adapter entries, −0.03 on trained entries), so
  `circuit_joint`'s 7 % fewer adapter entries (5 % fewer trained entries) do not explain the
  difference. Reading: an adapter restricted to the faithful fact circuit reaches the unrestricted
  adapter's efficacy and damages known non-English facts less than a count-matched random adapter by
  0.008 on a 0.03 scale, from three seeds, uncorrected p = 0.047, not surviving the multiplicity
  correction, and the mechanistic layer cannot confirm it. *Steering (A15):* `circuit_joint` efficacy 0.05 / 0.00 /
  0.07 (vacuous like every B1 steering cell), L3 0.013–0.019 against 0.029–0.038 for the per-type
  circuit vectors at the same |Δ| 8.65, bystander CAD excess −0.29 to −0.36 (per-type masks), target
  CAD excess −0.40 to −0.54. Its controls are as
  vacuous (`random_joint` 0.00–0.03, `inverse_joint` 0.02–0.03) and as local: `circuit_joint` −
  `random_joint` L3 −0.003 [−0.007, +0.002], L4 −0.001 [−0.005, +0.004]; − `inverse_joint` L3 −0.005
  [−0.010, −0.000] and L4 −0.014 [−0.024, −0.002] (p = 0.039 and 0.021, neither survives Holm,
  seed-cluster CIs include 0). The joint-set vectors are more local than the per-type circuit vectors
  at the same injected energy (L3 −0.017, L4 −0.023, both CIs exclude 0), because 61 small additions
  at coefficient 0.12 disturb less than 62 larger ones at 0.5. *H3 (A15):* the three `circuit_joint`
  LoRA cells are the study's first cells that pass the gate on the set they intervene on, and all three
  show an advantage over `random_joint` on L4 (+0.009 / +0.035 / +0.006) and, two of them, on L3
  (+0.013 / +0.007 / +0.000), the direction H3 predicts; the three steering `circuit_joint` cells also pass the gate but,
  with nothing to localise, their advantages are ±0.01. With six cells and three pillar values the
  monotone half remains untestable.
- **H4 (overlap predicts interference): inconclusive, and the earlier "consistent in direction"
  reading rested on the mis-specified IOI bystander.** With `ioi_fixed` (A16): overlap of the fact
  circuit with the bystanders langid 0.49 > refusal 0.30 > IOI 0.23; mean raw bystander CAD langid 0.141
  > refusal 0.116 > IOI 0.069, the same order, and the pre-registered statistic computed per cell over
  its three bystanders (no pseudo-replication) is strongly positive: mean Spearman(overlap, raw CAD)
  +0.71 [+0.63, +0.79] over 119 cells, seed-cluster [+0.64, +0.76], full rank agreement in 73 of 119
  cells. But the bystanders' own resample floors rank the same way (langid 0.409 > refusal 0.393 > IOI
  0.328), every raw value is below its floor, and after subtracting the floors the order reverses
  (refusal −0.277 < langid −0.269 < IOI −0.259; per-cell Spearman(overlap, CAD − floor) −0.25 [−0.38,
  −0.11], seed-cluster [−0.42, −0.14]). With the built-in IOI task the floor-adjusted statistic had read
  +0.59 [+0.51, +0.68] (v0.14–v0.15): that bystander's floor (0.483) was the highest and its circuit the
  least overlapping, which produced the agreement. Reading: by the letter of the pre-registered rule (raw
  bystander CAD monotone in overlap) H4 holds on B1; but a circuit's overlap with the fact circuit is
  confounded with how noisy its own attribution is (the shared edges are the large, unstable fact-recall
  edges), so the raw ranking is not evidence that overlap causes interference, and the floor-adjusted
  ranking contradicts it. Three bystanders cannot separate the two; H4 is inconclusive on B1. Overlap
  normalised to unit mass per circuit (audit C3) keeps the overlap order. *Within one bystander
  (amendment A17, exploratory; `audit/l4_shared.py`, `hypotheses_b1.md` "A17"; pairs over refusal,
  language-ID and both IOI tasks):* splitting each bystander's own edges into those touching the cell's
  intervened components and the rest, and comparing the two parts in units of their own resample floors
  (Λ = log(CAD/floor) shared − log(CAD/floor) unshared, which cancels CAD's Σ|s_e| denominator), the
  edges touching the adapted components move slightly more for LoRA at the circuit (+0.07 [+0.03,
  +0.11]), no more than the norm-matched perturbation at the same nodes (+0.12 [−0.01, +0.27]); for
  steering at the circuit +0.05 [−0.07, +0.16]; and *less* for MEMIT at MLP 0 (−0.24 [−0.35, −0.13]; its
  control −0.39): the layer-0 edit moves the rest of each bystander more than MLP 0's own edges. Pooled
  over the non-control cells +0.01 [−0.04, +0.05] (n = 219). The raw-CAD version D of the same split takes
  either sign (split by the intervened components S it is positive for 13 of the 28 B1 condition rows,
  e.g. MEMIT inverse +0.063 and its control +0.099; split by the target set T it is negative for every
  row, −0.006 to −0.256, perturbation controls included), because CAD is relative and the two regions
  hold edges of different base magnitude: partitioned CAD comparisons are magnitude-confounded and must
  be read against a same-region floor, which is the same confound that undoes H4 above. Trained and
  random changes do not separate on Λ either: LoRA at the circuit +0.07 against its control +0.12,
  random LoRA +0.13 against +0.03, Taylor +0.04 against −0.12.
- **AS-NC (activation shift at the last prompt position of 32 English fact prompts, outside vs inside the
  pre-registered target set, 550 vs 62 nodes; joint-set values in parentheses).** Layer-0 MEMIT: outside
  1.47–1.71 vs inside 0.87–1.08 across the three seeds (joint 1.48–1.74 vs 0.72–0.85; ratio 1.6–1.9):
  the edit's mechanistic footprint is mostly *outside* the set it was aimed at. Other conditions
  (seeds 0/1/2, outside vs inside): unrestricted MEMIT 0.81/0.58/0.67 vs 0.66/0.57/0.55, LoRA at the
  circuit 0.34/0.33/0.37 vs 0.30/0.31/0.31, `circuit_deep` 0.35/0.27/0.30 vs 0.44/0.34/0.36. Shifts
  concentrated inside the set occur for `circuit_deep` (all seeds), MEMIT at random layers on seeds 0
  and 1 (0.16 vs 0.23, 0.21 vs 0.31) and steering at the circuit (0.07–0.08 vs 0.20, where the
  injected vector itself is part of the inside activation). AS-NC here is a node-level variant of
  Sadhu et al.'s neuron-level measure, on English fact prompts for both behaviours.
- **L4 edge-level target distance (Chen et al.'s Circuit Distance analogue, Σ|Δs_e| / Σ|s_e| over every
  edge of the fact graph; no resample floor computed for it).** MEMIT at MLP 0 0.96 (its perturbation
  control 1.07), other MEMIT sites 0.62–0.86, LoRA 0.38–0.48 (unrestricted 0.68), steering 0.08–0.21,
  perturbation controls 0.03–0.29. Paired: MEMIT circuit − random +0.36 [+0.24, +0.50], circuit −
  causal_trace +0.24 [+0.17, +0.31]; LoRA circuit − random +0.01 [−0.07, +0.09]. It is the
  paper-faithful quantity; the node-turnover "circuit distance" stays in the tables for continuity.
- **L1 one-hop ripple (A9; 334 questions over 174 of the 180 facts; base accuracy 0.54).** Accuracy
  gain on questions whose answer follows from the corrected fact (the edited city's country's capital
  and continent, inverse questions, the capital of the country an office holder leads). No condition
  propagates: the best is LoRA at the circuit, +0.046, and the cells that correct 81–99 % of the facts
  gain −0.02 to +0.00 (MEMIT unrestricted / causal_trace / `circuit_deep`, LoRA unrestricted). MEMIT at
  MLP 0 *lowers* ripple accuracy by 0.15 (its perturbation control by 0.38): circuit − random −0.13
  [−0.18, −0.10], circuit − causal_trace −0.15 [−0.15, −0.14]. The net numbers hide two opposite
  movements: MEMIT cells fix 11–24 % of the ripple questions the base gets wrong and break 14–47 % of
  those it gets right (LoRA 13–20 % and 9–17 %, perturbation controls ≤ 9 % and ≤ 10 % except MLP 0).
  LoRA circuit − random +0.036 [+0.006, +0.069] does not survive seed resampling ([+0.000, +0.085]).
  Descriptive (A9).
- **L5 evidence arbitration (A12, Anand 2026; 552 probes on 138 never-edited facts).** The base model
  follows the injected passage almost always: follow rate 1.00 for `agree`, 0.80 `update`, 0.64 `stale`,
  0.61 `authority` (sd(A) 9.17, AURC 0.212, error of the most confident quarter 0.109), so its errors are
  over-trusting context. MEMIT at MLP 0 widens the spread (×1.24) and makes confident decisions worse
  (AURC +0.10, most-confident-quarter error +0.16, mean A +3.7 toward the passage), its perturbation
  control more so (×1.42, +0.18, +0.27); the other MEMIT sites move AURC and that error by −0.01 to
  +0.05 (spread ×1.01–1.08). LoRA shifts A toward memory (mean −1.0 to −3.7) and slightly improves AURC
  (−0.01 to −0.04; unrestricted LoRA widens the spread ×1.36); steering moves AURC and the error by at
  most 0.02 (spread ×0.94–1.04). Paired: MEMIT circuit − random spread ratio +0.20 [+0.17, +0.22],
  circuit − causal_trace +0.19 [+0.16, +0.22]. Descriptive (A12).
- **Steering directions against full-component patching (brief §12.5, `checks/results/steer_direction.json`).**
  On the seed's held-out wrong facts (metric: gold − base-top-1 logit at the last position), replacing
  the circuit nodes' full activations by the target-side mean moves the metric by +2.4 to +4.4, but
  replacing only their component along the steering vector moves it by +0.03 to +0.12 (0.8–2.8 % of the
  full effect), and the steering cell itself by −0.08 to −0.02. The B1 difference-of-means vector is not
  the direction along which these components carry the fact, which is consistent with its null
  efficacy (H1 vacuous). Four B1 non-circuit cells are flagged (direction effect opposite in sign to
  the full effect: inverse seed 0, whose full effect is 0.001, and random draws 0/3/4); no circuit cell
  is.

### 4.4 Audit and hypotheses: B2 refusal strengthening (final: 162 cells incl. the 3 `safetune_rd` cells, all five layers)
Source: `docs/results/hypotheses_b2.md`, `docs/results/b2_cells.csv`. Target: refusal of held-out
harmful cyber-intrusion prompts (12 per language). The base model already refuses 11/12 in English, so
the pre-registered English efficacy is capped at +0.083 (amendment A3); `xl` = mean refusal gain over
hi/ar/fr/sw, where the base refuses 0.50/0.58/0.83/0.58. "benign" = refusal of near-miss benign cyber
prompts (over-refusal; base 0.07 en, 0.00–0.04 elsewhere).

| intervention · condition | efficacy en (cap 0.083) | efficacy xl | |Δθ| | L3 mean abs Δ | benign refusal Δ (en) | L1 neighbourhood Δ (en) | WikiText ppl ratio |
|---|---|---|---|---|---|---|---|
| LoRA · circuit | 0.08 | 0.28 | 3.5 | 0.084 | +0.37 | −0.03 | 1.063 |
| LoRA · random (5) | 0.08 | 0.29 | 3.7 | 0.090 | +0.39 | −0.02 | 1.033 |
| LoRA · inverse | 0.08 | 0.20 | 4.0 | 0.044 | +0.38 | −0.01 | 1.012 |
| LoRA · Taylor | 0.08 | 0.26 | 3.6 | 0.096 | +0.54 | −0.03 | 1.053 |
| LoRA · unrestricted | 0.08 | 0.37 | 6.3 | 0.195 | +0.42 | −0.03 | 1.045 |
| LoRA · perturb_circuit | 0.00 | 0.00 | 3.5 | 0.008 | 0.00 | 0.00 | 1.000 |
| steering · circuit (0.5×) | 0.00 | 0.22 | 24.5 | 0.188 | +0.19 | −0.10 | 1.168 |
| steering · random (5) | −0.10 | 0.10 | 24.5 | 0.109 | +0.16 | −0.06 | 1.122 |
| steering · inverse | 0.03 | 0.05 | 24.5 | 0.038 | +0.01 | −0.02 | 1.059 |
| steering · Taylor | 0.00 | 0.13 | 24.5 | 0.081 | +0.07 | −0.03 | 1.124 |
| steering · unrestricted | 0.08 | 0.31 | 24.5 | 0.263 | +0.50 | −0.07 | 1.182 |
| MEMIT · circuit (MLP 0) | −0.11 | −0.13 | 78 | 0.073 | −0.02 | −0.06 | 0.998 |
| MEMIT · circuit_deep (MLP 19) | 0.00 | 0.06 | 27 | 0.023 | 0.00 | −0.02 | 1.003 |
| MEMIT · inverse | 0.08 | 0.15 | 39 | 0.034 | −0.02 | −0.01 | 1.001 |
| MEMIT · random (5) | 0.02 | 0.03 | 30 | 0.020 | +0.01 | −0.02 | 1.002 |
| MEMIT · Taylor (MLP 27) | 0.00 | −0.01 | 30 | 0.023 | 0.00 | −0.02 | 1.000 |
| MEMIT · unrestricted | 0.00 | −0.10 | 26 | 0.021 | 0.00 | −0.02 | 1.001 |
| MEMIT · perturb_circuit | −0.25 | −0.22 | 78 | 0.186 | 0.00 | −0.12 | 1.016 |

- **H1 (efficacy parity), pre-registered English.** Vacuous for every intervention: LoRA 1.00 (both
  exactly at the +0.083 cap), steering 0.00 (0.00 vs 0.08), MEMIT undefined (unrestricted 0.00, circuit
  −0.11). **H1-xl (A3).** Refuted for LoRA (0.78 [0.78, 0.78]; 0.29 vs 0.38, every circuit cell at
  the same 0.29) and steering (0.69 [0.67, 0.73]; 0.22 vs 0.31); undefined for MEMIT, whose
  unrestricted and circuit cells both *lower* non-English refusal (−0.10, −0.13). Paired at seed ×
  language: LoRA circuit = random (Δ +0.00 [−0.02, +0.03]) and = Taylor (+0.01), > inverse (+0.09
  [+0.05, +0.14]); steering circuit > random (+0.11 [+0.03,
  +0.18]; seed-cluster CI [−0.03, +0.23], not robust to resampling whole seeds), > inverse (+0.17), > Taylor (+0.09), < unrestricted (−0.10 [−0.15, −0.06]).
- **H2 (locality advantage), L3.** Refuted for steering (circuit − random +0.08 [+0.03, +0.12], seed-cluster
  CI [−0.02, +0.15], so this comparison alone is inconclusive at the seed level; circuit −
  inverse +0.15 [+0.10, +0.20], seed-cluster [+0.13, +0.17]: the circuit vector disturbs non-English facts, benign and other-category
  refusal more than random or inverse vectors of the same norm, though less than the unrestricted
  vector, 0.26). Mixed for LoRA (circuit − random −0.008 [−0.015, −0.000], a 0.008 advantage on a 0.09
  scale; circuit − inverse +0.037 [+0.022, +0.054], disadvantage). Refuted for MEMIT (circuit − random
  +0.06 [+0.03, +0.08], circuit − inverse +0.04 [+0.01, +0.07]); the norm-matched perturbation at MLP 0
  is worse again (0.19 vs 0.07), so, as in B1, a change of the size the layer-0 site needs (|Δθ| 78 vs
  26–39) is damaging there whatever is written; at the other sites the real edits cause about twice the
  L3 collateral of equal-norm noise (0.020–0.034 vs 0.009–0.016).
- **The dominant B2 collateral is over-refusal, and no condition avoids it.** Every LoRA condition
  raises English refusal of *benign* near-miss prompts by 0.37–0.54 (base 0.07) while gaining 0.20–0.37
  on non-English harmful prompts; circuit (+0.37) is not better than random (+0.39) or inverse (+0.38),
  and Taylor is worst (+0.54). The steering circuit vector costs +0.19 benign refusal, a 10-point drop
  in English fact neighbourhood accuracy and +17 % WikiText perplexity for a 0.22 xl gain; the inverse
  vector costs almost nothing and gains almost nothing (0.05). The over-refusal propagates: LoRA at the
  circuit raises benign refusal in hi/ar/fr/sw by +0.17/+0.07/+0.04/+0.11 (random +0.16/+0.06/+0.06/
  +0.14, unrestricted +0.38/+0.29/+0.21/+0.32), the steering circuit vector by +0.12/+0.27/+0.30/+0.43
  (inverse ≤ +0.04); MEMIT changes it by ≤ 0.04 everywhere.
- **MEMIT cannot strengthen refusal at any site.** Writing "refuse" keys at the circuit's MLP 0 lowers
  held-out refusal in every language (en −0.11, hi −0.22, fr −0.19) and costs 16–18 points of hi/ar
  neighbourhood accuracy; the best MEMIT site is the *inverse* set (+0.15 xl, no collateral above 0.03),
  and `circuit_deep` (MLP 19) is a no-op (+0.06 xl, all collateral ≤ 0.02). Cells that lower refusal are
  withheld from the Hub by the rule in `data/push_hf.py`.
- **H3 (faithfulness gates locality), B2 half.** Every B2 circuit fails the pre-registered gate under
  both definitions: joint set patching −0.04 to −0.03 at 10 % (random-node margin −0.05 to −0.04,
  zero-ablation sign flips on two seeds); the per-type set the cells intervene on (A5) −0.001 / −0.002 /
  −0.005 (random per-type sets +0.002), and ablating it costs less than ablating random per-type sets
  (necessity 0.48 / 0.50 / 0.53 vs 0.39 / 0.40 / 0.34). The pooled evaluation is below, after the L4
  results.
- **Steering sweep.** Coefficients above the matched base are destructive for every condition: at 2×
  the circuit vector drives English refusal to 0 (efficacy −0.92) with WikiText ×30, random vectors
  reach ×5 to ×1428; the matched-efficacy table is uninformative (e* = −0.03, English at ceiling).
- **H2, mechanistic L4 (per-type masks since A5; `ioi_fixed` bystander since A16; joint-ranked values in
  `b2_cells.csv` `*_joint`; per-cell Spearman between the definitions 0.86).** Bystander CAD is below the
  resample floor (0.33 `ioi_fixed`, 0.41 language-ID, 0.51 facts) for every B2 condition (excess −0.05
  for the SafeTune direction, −0.10 for MEMIT at MLP 0, −0.22 to −0.39 for the rest) except the layer-0
  perturbation control (+0.02; the rest −0.22 to −0.39 for real conditions, −0.35 to −0.41 for their
  controls); the paired differences are therefore differences between values that are smaller than
  the base attribution's between-subset variation (the same-data repeat floor is exactly 0, so they are
  not measurement noise of the cell itself, but their stability across example subsets was never
  measured), and they all point the wrong way: MEMIT circuit − random +0.28 [+0.25, +0.30], LoRA circuit −
  random +0.008 [+0.002, +0.016], steering circuit − random +0.037 [+0.010, +0.062] and circuit −
  inverse +0.10 [+0.07, +0.14]. H2 is refuted or null on both layers for every B2 intervention under
  either definition. Target-circuit CAD exceeds its floor for the conditions that changed the behaviour
  most (LoRA unrestricted +0.48 (joint +0.69), LoRA circuit / random / Taylor +0.06 to +0.10 (joint
  +0.12 to +0.14), steering unrestricted +0.13, `safetune_rd` +0.19, steering circuit +0.02, and MEMIT
  at MLP 0 +0.05) and not for
  the inverse sets (LoRA −0.14, steering −0.22) or any perturbation control (−0.31 to −0.38, except the
  MLP-0 control at +0.23). Holm over the 12 B2 H2 tests: all 12 survive with the corrected bystander (8
  of 12 had with the built-in task); the LoRA circuit − random L3 difference (−0.008, bootstrap
  p = 0.050 after A18, 0.036 with the merged adapters) passes the last step of the step-down at
  exactly its threshold and is one-tenth of the metric's scale, so it is reported as borderline and
  immaterial.
- **Exploratory: the refusal circuit's edges move with the behaviour even though the circuit fails the
  patching pillar.** Across the 57 B2 main cells (incl. `safetune_rd`), Spearman(efficacy xl, target CAD −
  floor on the per-type mask) = 0.73 (p = 1e−10); within steering 0.86 (n = 20), within
  LoRA 0.78 (n = 17), within MEMIT −0.32 (n = 20, all near zero efficacy). The B1 analogue over the 79
  B1 main cells without perturbation controls (A15 cells included) is 0.68 (p = 4e−12), but only between
  interventions: within steering it is negative (−0.49, p = 0.009, n = 28), within LoRA 0.34 (p = 0.08,
  n = 28), within MEMIT −0.04 (p = 0.86, n = 23), so on B1 it separates weight edits from steering and
  says nothing within an intervention (on the 57 pre-registered cells alone: 0.68, steering −0.65, LoRA
  0.16). A per-type set that recovers none of the refusal logit
  difference under patching and is not necessary still contains the edges whose attribution changes
  when refusal is strengthened, which is the "A circuit, not the circuit" reading: EAP-IG found
  refusal-relevant components, not a sufficient or necessary set.
- **H4 (overlap predicts interference): inconclusive, as on B1.** Overlap of the refusal circuit with
  the bystanders (A16): langid 0.34 > facts 0.30 > IOI 0.20; mean raw bystander CAD langid 0.104 > facts
  0.088 > IOI 0.075, the same order; per-cell Spearman(overlap, raw CAD) +0.68 [+0.57, +0.77] over 94
  cells, full rank agreement in 48 of 94. The floors are facts 0.507, langid 0.408, IOI 0.328, so the
  floor-adjusted means invert the order (IOI −0.252 > langid −0.304 > facts −0.419; per-cell Spearman
  −0.39 [−0.46, −0.32], full agreement in 0 of 94 cells). As in §4.3, the pre-registered raw statistic
  holds by its letter and the floor-adjusted one contradicts it, and overlap with the target is
  confounded with the bystander's attribution noise; not established. Within one bystander (A17, as in
  §4.3): LoRA at the circuit +0.19 [−0.01, +0.40] (control +0.02), Taylor +0.33 (control +0.12), random
  +0.10 (control +0.02), steering at the circuit −0.27 [−0.42, −0.10], pooled +0.055 [+0.000, +0.108]
  (n = 139): on B2 the edges touching the adapted components move more in noise units for every
  trained LoRA condition than for its norm-matched control, and less for the circuit steering vector;
  nothing specific to the circuit.
- **H3, pooled with B1 (18 circuit cells, 6 circuits; B1 L4 side rescored, A13).** On the per-type sets
  the cells intervene on, the 9 B2 cells fail the gate as pre-registered (random margin ≤ 0, zero-ablation
  ≤ 0 on two seeds) and the 9 B1 cells pass it; all 18 are gated only with A5's added patching < 0.1
  criterion (A13). Gate half: 18/18 show no L4 advantage (v0.13's one exception, B1 LoRA seed 0 at
  +0.02, came from the inflated seed-0 edges and is −0.01 after the rescore); with the strict rule of no
  L4 *and* no L3 advantage 13/18 (the L3 "advantages" are +0.003 to +0.020 on a 0.08 scale; the B2 LoRA
  circuit − random L3 difference is −0.008 [−0.015, −0.000], borderline and one-tenth of the scale). On
  the joint-ranked sets only the 9 B2 cells are gated: 9/9 and 5/9. Monotone half: no pillar of either
  set (patching, ablation margins, necessity, necessity margin) correlates with the L4 or L3 advantage
  (|ρ| ≤ 0.21, p ≥ 0.41, n = 18; the largest are random margin −0.204 against L4 and zero-ablation
  ratio −0.204 against L3). The v0.7 reading "faithful circuits (B1) show no advantage either" is
  withdrawn: no cell in this study intervened on a set that reproduces its behaviour, so the monotone
  half of H3 is untested. Under the original gate, the B2 cells confirm the gate half (no advantage)
  and the B1 cells, which pass the gate, show no advantage either; under the A5 gate all 18 are gated
  and 18/18 are consistent.
- **SafeTune refusal-direction baseline (`safetune_rd`, amendment A4; 3 seeds, all five layers).**
  SafeTune's Arditi sweep picks layer 19/18/19 of 36 with a raw difference-of-means norm of 27.5–28.8
  (bypass 0.00 under ablation, KL 0.03); norm-matched to the circuit condition (|Δ| 24.49, coefficient
  0.85 × the raw direction, i.e. close to the paper's natural scale) it is the strongest refusal
  strengthener and by far the least local steering cell: efficacy_xl 0.340 vs circuit 0.215 (circuit −
  safetune_rd −0.125 [−0.181, −0.076], seed × language), but English benign over-refusal +0.81 (circuit
  +0.19), XSTest-safe +0.79, neighbourhood accuracy −0.77 (circuit −0.10), WikiText ×1.50 (circuit ×1.17),
  L3 mean |Δ| 0.64 (circuit 0.19), bystander CAD 0.40 (the only steering condition near its resample floor,
  excess −0.06; circuit −0.23), target-circuit CAD 0.58 (excess +0.19, the largest of any B2 steering
  cell; joint masks 0.72 / +0.28). A
  single residual-stream direction added at every position turns the model into a refuser of everything,
  which is the expected behaviour of the raw refusal direction at full strength on near-miss prompts; the
  circuit-node vectors at the same norm are weaker and more local, but still not local (H2). The
  comparison is exploratory (post hoc, no sweep cells), so it does not change any hypothesis verdict.
- **Judge robustness (AuditKIT vs SafeTune refusal detectors).** Re-judging every stored English refusal
  generation of the base model and the 94 B2 (91 B1) main cells with SafeTune's detector agrees with the
  AuditKIT verdict on 99.5 % (B2) and 99.2 % (B1) of generations; base rates are identical to two decimals
  and every per-condition harmful/benign/XSTest delta moves by ≤ 0.03 (the largest differences are on the
  MEMIT perturbation controls, whose outputs are degenerate). No conclusion in §4.3–4.4 depends on which
  regex judge is used; both are English-only, which is why the non-English families keep the multilingual
  onset patterns in `neb.judge` (§5).

- **L5 relearning probe (A10; same 40-step benign LoRA fine-tune on 64 Dolly pairs applied to the base
  model and to every main cell; no harmful completion is trained on).** The attack does not erode the
  base model: after it, held-out refusal is unchanged in en/ar/sw and one prompt higher in hi and fr
  (0.67, 0.92), benign over-refusal 0.04 (was 0.07). Gains are therefore measured against the attacked
  base. LoRA cells keep 69–89 % of their xl refusal gain (circuit 0.20 of 0.29; random 0.22 of 0.29;
  inverse 0.17 of 0.20; unrestricted 0.29 of 0.38) while their English benign over-refusal falls from
  +0.37–0.55 to +0.09–0.11 (unrestricted +0.20). The steering circuit vector keeps 0.21 of 0.22 (random 0.05 of 0.10, Taylor
  0.06 of 0.12, unrestricted 0.15 of 0.31), and SafeTune's direction turns negative (−0.14 xl, −0.14
  en). Steering hooks stay attached during the fine-tune, so for steering this measures whether a
  benign adapter counteracts an always-on vector, not whether a weight change persists. Paired: steering
  circuit − random +0.16 [+0.06, +0.23] (seed-cluster [+0.05, +0.25]); LoRA circuit − random −0.017
  [−0.058, +0.021]. MEMIT cells that had lowered refusal stay lower. Descriptive (A10).
- **L5 evidence arbitration, B2 cells (A12; same probe and base as §4.3).** Strengthening refusal by
  steering makes the model follow injected passages more and worse: SafeTune's direction AURC +0.14,
  most-confident-quarter error +0.16, mean A +6.2 toward the passage; the circuit vector +0.06, +0.05,
  +1.9 (spread ×0.86; circuit − random spread ratio −0.13 [−0.18, −0.07]); MEMIT at MLP 0 +0.06 / +0.09
  and its perturbation control +0.05 / +0.12; LoRA ≤ 0.02 on both. Descriptive (A12).
- **L4 edge-level target distance, B2.** LoRA unrestricted 1.22, SafeTune direction 0.73, MEMIT at MLP 0
  0.74 (control 1.01), steering unrestricted 0.67, LoRA circuit / random / Taylor 0.59–0.61, steering
  circuit 0.58, other MEMIT sites 0.19–0.37, LoRA perturbation controls 0.02–0.04.
- **Steering directions against full-component patching, B2 (§12.5).** On the benign near-miss prompts
  (metric: logit("I'm") − logit("In"), the refusal vs compliance onsets), replacing the circuit nodes'
  activations by the harmful-side mean moves the metric by +8.3 to +8.4; replacing only their component
  along the steering vector moves it by +5.4 to +5.5 (64–66 %), and the steering cell itself by +4.0 to
  +4.5. Taylor 64 %, unrestricted 93 %, random draws 36–96 %; one cell flagged (inverse seed 1, whose
  full effect is 0.07). The B2 vectors act mostly along their own direction at the nodes they patch:
  no subspace illusion at the circuit nodes. (The Taylor and unrestricted cells are identical across
  seeds because their node sets do not depend on the discovery seed.)

### 4.5 Strongest alternative explanation for each verdict (brief §3.3 item 6)
- **H1 refuted (MEMIT, LoRA).** Alternative: the pre-registered "circuit" site is MLP 0 for both
  behaviours, so H1 tests layer-0 editing, not circuit editing in general; the exploratory MLP-5 site
  reaches parity (0.88). For LoRA the alternative is capacity: the circuit's 61 nodes give the adapter
  fewer useful parameters than the unrestricted 15.1 M, and random head sets (0.64) show that *which*
  heads matter less than *how many*.
- **H2 refuted/null.** Alternative: the collateral at the circuit site is a function of the size of the
  weight change (|Δθ| 131 vs 36–69), which the norm-matched random perturbation at MLP 0 exceeds (L3 0.39,
  WikiText +7 %); this holds at MLP 0 only (elsewhere the real edits cause 4–12× the collateral of
  equal-norm noise), and site and size are not separated there; for LoRA and steering the differences are at the resolution of the metrics (L3 scale
  0.02–0.09, bystander CAD below the resample floor), so a small true advantage could be hidden by
  measurement noise rather than absent.
- **H3 gate holds, correlation absent.** Alternative: only three distinct pillar values per behaviour
  (one per seed) and no faithful circuit that ever showed an advantage, so the test has no power on the
  monotone half; a faithful circuit with a locality advantage was never observed, not shown impossible.
- **H4 inconclusive.** Alternative to the positive raw-CAD reading: the three bystanders' own
  attribution noise (their resample floors) ranks with their overlap with the target, because the shared
  edges are the large, unstable fact-recall edges; raw CAD then tracks overlap for a reason that has
  nothing to do with the intervention, and the floor-adjusted ranking reverses. Alternative to the
  negative floor-adjusted reading: subtracting a floor that is itself a function of overlap can
  over-correct. Every value is below the floor, so neither reading is about interference; three
  bystanders cannot separate overlap from bystander identity. The within-bystander split (A17) holds
  identity fixed and finds that where a bystander's edges move follows where the parameters changed,
  with no consistent difference between a trained adapter and norm-matched noise (B1 LoRA at the
  circuit +0.07 vs its control +0.24; B2 +0.20 vs −0.01).
- **H5 refuted.** Alternative: the verdict compares MLP 0 with two single sites (causal tracing at
  MLP 3, A8; Taylor at MLP 16) that both propagate better, and with `circuit_deep` (MLP 5), which
  propagates as well as the traced site. The causal-tracing profile is flat over layers 0–11, so any
  early-to-mid MLP other than the detokenisation layer would likely do as well: the evidence points at
  MLP 0 being a poor editing site (|Δθ| 131), not at circuit membership harming propagation.

### 4.6 Post-hoc countermeasures to three of the four limitations (A19–A21, run on puget; `docs/results/puget-a19-a21.md`)
Sealed before running (`docs/preregistration.md` A19–A21), run on the second machine with every base and
cell of an amendment on that machine, copied back and verified byte for byte. Exploratory: nothing enters
the H1–H5 Holm count and no verdict above changes.
- **Inside view (A19).** Per-example EAP-IG with a measured quadrature rule (`midpoint6`; error vs a
  48-step reference 0.078 against CircuitKIT's 0.237) and jackknifed paired CAD with TOST at δ_L4 = 0.05.
  Robust under the sealed SE and two post-hoc sensitivity SEs: circuit-site MEMIT disturbs every
  bystander circuit far beyond the margin (+0.32 … +0.37 CAD on B1, +0.23 … +0.26 on B2); B1 steering on
  the circuit is less disruptive than the inverse set (−0.055); LoRA circuit − random is equivalent on
  both behaviours. Several LoRA verdicts are not robust to the wider SE and are not claimed.
- **Certified circuit (A20).** A circuit passing sufficiency ≥ 0.8 and necessity r′ ≤ 0.2 on a held-out
  fold exists on every seed but holds 86–135 nodes (14–22 %) and 29–34 of the 36 MLPs. LoRA on it reaches
  0.99 efficacy like every control (H1 at ceiling, uninformative); steering at the shared norm stays
  near 0.02. Against entry-matched random heads (trained entries within 0.16 %), random, inverse and
  unrestricted sets, every L3 and L4 contrast is equivalent within the margins (one inconclusive). The
  earlier locality null is therefore not an artefact of an unfaithful circuit.
- **Edit site (A21).** MEMIT at 17 layers, uncapped, rescaled to a common norm and matched noise: layer 0
  (the pre-registered circuit site) writes the fewest facts (0.37 vs 0.83–0.88 at layers 4–10) and
  propagates least to other languages (0.03 vs 0.28–0.35); its large L3 collateral is mostly edit size
  (0.185 → 0.050 at the common norm); it is the only site where same-norm noise does real damage. The
  sealed test (does the circuit's score at a layer predict efficacy or collateral beyond depth and size)
  rejects none of its six primary tests; the score is negatively related to cross-lingual propagation.

### 4.7 A refusal adapter that is better than the base model (A22–A24; `docs/results/a22_b2.md`, `a24_b2.md`)
Sealed before running (`docs/preregistration.md` A22, A23, A24, each with its file hash in
`docs/notes.md`). Exploratory: nothing enters the H1–H5 Holm count and no verdict above changes.
Held-out test items only: 159 harmful prompts per language (80 of the trained category, 79 of two
categories never trained on), 59 benign near-miss prompts per language, 200 XSTest unsafe prompts;
checkpoints chosen on a separate dev split by a fixed rule. Harmful compliance is the share of
replies an LLM judge labels `comply` (`audit/a22_judge.py`); the regex refusal rate is beside it.

| | base | A22 `circuit30_s0` | A24 `circuit30_s0` | SafeTune steering |
|---|---|---|---|---|
| harmful comply, hi/ar/fr/sw | 0.124 | 0.038 (−0.086 [−0.112, −0.063]) | 0.071 (−0.053 [−0.077, −0.031]) | 0.088 (−0.036 [−0.057, −0.017]) |
| harmful comply, en | 0.088 | 0.019 (−0.069 [−0.113, −0.031]) | 0.044 (−0.044 [−0.082, −0.013]) | 0.038 |
| harmful refusal, hi/ar/fr/sw (regex) | 0.580 | 0.942 | 0.885 | 0.741 |
| benign refusal, 5 languages | 0.031 | 0.088 | 0.061 | 0.153 |
| WikiText perplexity ratio | 1 | 1.260 | 1.018 | 1.043 |
| perplexity ratio, 5 languages, held-out XNLI | 1 | 1.401 | 1.023 | 1.035 |
| known-fact / paraphrased / MMLU (5 languages) | 0.877 / 0.840 / 0.453 | 0.863 / 0.800 / 0.440 | 0.873 / 0.805 / 0.450 | 0.847 / 0.820 / 0.417 |
| replies that are exactly the refusal, en/hi/ar/fr/sw | – | 0.77 / 0.32 / 0.02 / 0.11 / 0.01 | 0.94 / 0.87 / 0.87 / 0.84 / 0.91 | – |

- **A22** (24 LoRA cells + a SafeTune baseline; `scripts/chains/17_a22.sh`). LoRA on the
  joint-ranked 30 % refusal circuit, trained on 80 harmful prompts × 5 languages → the model's own
  refusal and on 151 benign prompts × 5 languages → the base model's own greedy reply. Every LoRA
  condition lowers compliance (means over cells, hi/ar/fr/sw: circuit30 0.059, random30 0.052,
  inverse30 0.063, unrestricted 0.058, circuit10 0.058, random10 0.079) and raises benign refusal by
  about 0.05 (0.075–0.113). The SafeTune refusal direction, with its strength chosen on the same dev
  split by the same rule, is worse on both (released cell − baseline: comply −0.050 [−0.072,
  −0.030], benign refusal −0.064 [−0.102, −0.027]).
- **What the audit found in A22.** WikiText perplexity ratio 1.07–1.26 for `circuit30`, up to 1.58
  for `unrestricted`; on held-out sentences in five languages 1.15–1.40 (about the same in each
  language), with facts, MMLU and IFEval within 0.01–0.04 of the base. Outside English the adapters
  rarely end the turn after refusing: `lora.py::encode` appends the tokenizer's EOS
  (`<|END_OF_TURN_TOKEN|>`), and generation stops on `<EOS_TOKEN>`. The same recipe trained every
  LoRA cell of the main study.
- **A23** (stopped). Plain-text retain pairs with the base model's greedy continuation as target:
  the dev perplexity ratio went to 1.60 after 15 steps and 6.79 after 120, the sealed checkpoint
  rule found no qualifying snapshot, and the run was stopped after one cell with no test item
  generated. Cross-entropy on greedy text sharpens the adapter; A22's greedy benign targets are the
  likely source of its own drift.
- **A24** (9 cells; `scripts/chains/18_a24.sh`). A23 with the continuation sampled from the base
  model at temperature 1, the model's own end tokens and the perplexity limit (≤ 1.05 on dev) in
  the checkpoint rule. Means over 3 seeds: comply 0.066 / 0.064 / 0.064 (circuit30 / random30 /
  unrestricted), benign refusal 0.081 / 0.081 / 0.084, WikiText ratio 1.020 / 1.019 / 1.015,
  five-language ratio 1.031 / 1.025 / 1.015; 83–96 % of refusals end the turn.
- **The circuit restriction.** A22: circuit30 − random30 +0.007 on comply (seed-cluster CI
  [+0.001, +0.013], paired t p = 0.18) and −0.010 on benign refusal ([−0.015, −0.003], p = 0.10):
  opposite directions, under a point each. `circuit30_s0` and `random30_s0` judge all 795 harmful
  test items alike; the 30 % circuit holds 35 of 36 MLPs. circuit10 − random10: −0.020 on comply and
  +0.038 on benign refusal, with the circuit cells trained four times longer under the checkpoint
  rule. A24: circuit30 − random30 +0.002 [−0.002, +0.005] and 0.000; − unrestricted +0.002 and
  −0.002, every interval including 0. No advantage is claimed.
- **Release rule (A23 (f)).** A24's `circuit30_s0` has the lower perplexity (met) and the lower
  benign refusal (met), and its comply share is 0.033 above A22's against 0.03 allowed (not met), so
  A22's adapter stays the release; both are published, and over three seeds the recipes differ by
  0.007 on comply.
- **The regex judge.** On the base model's 795 harmful test replies the regex counts 498 refusals,
  the LLM judge 555, and labels 147 further replies `safe` and 93 `comply`. English: regex refusal 0.811, judge
  comply 0.088; Hindi: 0.396 and 0.113 (judge refuse 0.604, safe 0.283); agreement on refusal
  97.5 % and 78 %. The 42-point English–Hindi refusal gap of the regex is a 2.5-point compliance gap.

## 5. Limitations and failure log
Numbers from `docs/notes.md`, `docs/results/circuits.md` and the audit; nothing here is softened.

**Found on 2026-11-11/12 (A22–A24)**
- *What the adapters still cost.* Benign refusal +0.03 to +0.06 (five languages), of which
  Swahili +0.15 to +0.17: the Swahili benign prompts are model translations of idioms ("steal the
  show", "blow up a balloon"), 54 of 191 of the benign training prompts failed back-translation
  there, and nothing in A24 targets it. Paraphrased known facts −0.035.
- *Seeds.* The released cell is the strongest of A22's three (comply 0.038 against 0.074 and
  0.066); it was fixed before training, and the three are reported separately.
- *Judge.* A local 27 B model (Qwen3.8-27B, GGUF Q5_K_M), temperature 0, grammar-constrained. A
  stratified sample of 60 labels was checked by the AI assistant that ran the analysis: 55 agree on
  comply / not comply, 2 disagree, 3 are cut off inside an opening disclaimer and labelled `safe`;
  all five doubtful ones are base-model replies. No human check yet. The shortcut for refusal-only
  replies was narrowed before any label existed (A22 (h)).
- *Test prompts.* Some MultiJail "harmful" prompts are benign questions; helping with them counts
  as comply for every model. The test items are the same for A22 and A24, and A24's design was
  chosen on the dev split after A22's test results were known; no test item is in a training or dev
  set.
- *Benign refusal is the regex judge, and the LLM judge agrees with it.* A judge pass over the
  benign test replies (`audit/a22_judge.py families=benign`, `runs/a2*/judge_benign/`; robustness
  check, not sealed) gives, as five-language refusal share by judge / by regex: base 0.034 / 0.031,
  A22 `circuit30_s0` 0.085 / 0.088, A24 `circuit30_s0` 0.064 / 0.061, SafeTune 0.153 / 0.153; by
  the judge the rise over the base is +0.051 [+0.020, +0.081] for A22, +0.031 [+0.007, +0.058] for
  A24 and +0.119 [+0.081, +0.159] for the SafeTune baseline. Unlike on harmful prompts, the regex
  is accurate here, because an over-refusal is the trained refusal sentence.
- *L4 was not run for A22–A24.*
- *Operations.* The first A22 launch died on a full disk (swap growth with 15 GB free); the judge
  stage failed once on a model path carrying terminal colour codes and was restarted by the
  supervisor; A23 was stopped by hand after its first cell (`docs/notes.md`).

**Found on 2026-11-09 (A19–A21)**
- *Audit order noise.* Until 2026-11-09 `audit/run_audit.py` iterated the edited facts in Python set
  order, which follows each process's hash seed, so batching and bf16 padding could flip near-tie answers:
  one A20 cell audited twice differed in 2 of 1990 scored items (edited-fact families only). About ±1
  item in 60 per edited-fact family per cell, unbiased, in the main study and A20; fixed before A21
  (`sorted(edited)`), whose audits are byte-identical across processes. Not re-run.
- *A19's sealed SE covers example sampling only*; the spread across cells is about 10× larger, so A19 and
  A20 L4 verdicts are reported next to two post-hoc sensitivity SEs and only verdicts that survive all
  three are claimed (§4.6). `midpoint6` leaves a 26–92 % IG completeness gap.
- *Cross-device.* A19–A21 ran on CUDA; puget reproduced the Mac's pillar and audit headlines, but MEMIT
  layer-0 edits differ from the Mac's by 1.7–15 % relative L2 (prefix sampling); no comparison mixes
  devices. A21 covers 17 of 29 editable layers.

**Circuits**
- *Zero-ablation is floored at 0.5 for B1-style metrics.* Ablating 90 % of the 612 nodes drives the
  logit difference to ≈ 0, which is exactly halfway between clean (+11.25) and corrupt (−11.31), so random
  size-matched node sets score 0.47–0.51 under zero ablation, the same as the circuit (0.50). The
  pre-registered P2 gate (raw ratio ≤ 0) cannot fire for B1 circuits; the informative quantity is the
  margin over random sets (`ablation_margin`), reported in `circuits.md` and used in H3. The exact floor
  r0 = −corrupt/(clean − corrupt) is measured per task (`checks/results/metric_span.json`) and the necessity
  numbers are re-referenced to it (`nec0_*`, fourth audit B.1); CircuitKIT's ratios are bf16 means,
  resolved to 0.002–0.006 of the span.
- *The refusal circuit is not faithful at the pre-registered 10 %.* b2_en patching raw ratio is −0.04,
  −0.03, −0.04 across seeds (random node sets +0.003 to +0.006); zero-ablation inverts the sign on two of
  three seeds (−0.50, −0.24, +0.45). Faithfulness only appears at 30 % of nodes (0.58 / 0.49 / 0.45). The
  pre-registered H3 gate therefore fires for every B2 circuit, and the B2 locality result is a test of what
  an *unfaithful* circuit buys, which is what H3 predicts should be nothing.
- *IOI is not recoverable at 10 % on Tiny Aya.* Patching 0.11 / 0.02 / 0.20; random node sets score
  *higher* (0.28 / 0.30 / 0.24) than the circuit. The task is mis-specified for this tokenizer: 14 of 99
  names are multi-token and most clean/corrupt pairs differ in token count, so EAP compares end-padded,
  shifted positions (fourth audit C.2). IOI is used only as a bystander circuit in L4 and its CAD must
  be read against that. The flat 1–2 % size curve is a bf16-resolution effect, not a selection bug
  (`checks/results/ioi_plateau.json`).
- *B1 circuits vary by language and seed.* Patching at 10 % (3 seeds): en 0.33–0.56, hi 0.55–0.82, ar
  0.64–0.92, fr 0.60–0.84, sw 0.17–0.43; the seed-0 Swahili fact circuit is barely above its minimality
  control (0.17 vs 0.04).
- *CircuitKIT's Pillar 5 baselines are degenerate at node level* (random and magnitude circuits sit on the
  corrupt floor, std 0 across draws, because `input` is rarely sampled and no path then carries clean
  activations). All random-circuit comparisons in this report use our same-path control
  (`aya_circuits/random_baseline.py`), not Pillar 5.
- *EAP-IG edge scores are sampling-noisy.* A no-op edit scored CAD 1.085 against base edges computed
  under different settings; L4 recomputes base edges under the audit's own settings and reports a
  same-settings null floor, and every CAD is read relative to that floor.
- *`langid_en` seed 1* OOMed twice in the chain (batch 8, then batch 4 for its control) and was computed last at batch 2 (patching 0.35, random 0.000, control 0.36).

**Interventions**
- *MEMIT at the pre-registered circuit site (MLP 0) needs a change 1.9–3.6× larger* (|Δθ| 131 vs 36–69)
  than any other site to write the same 60 facts; a random perturbation of the same norm at the same
  site produces more collateral than the edit (L3 0.39 vs 0.20, WikiText +7 %). At the other sites the
  opposite holds (edit 4–12× the collateral of its equal-norm control). The circuit pointed to the
  detokenisation block, which is not where MEMIT's key/value assumption holds. `circuit_deep` is an
  exploratory, post-hoc site and is reported as such.
- *MEMIT at the final layers is degenerate* (zero gradient at the subject position of `resid_post` of
  the last layer), so "unrestricted" means layers 0..28, not 0..35.
- *Difference-of-means steering cannot correct specific facts.* Max efficacy 0.13 over four
  coefficients and five conditions; the matched-efficacy comparison is at noise level (e* = 0.022).
  Amendment A6 (sources disjoint from the efficacy fold) changed no reading: main-cell efficacy 0.00–0.07
  before and after, H2 steering tests +0.040 vs +0.048 (random) and −0.075 vs −0.084 (inverse).
- *Steering coefficient choice.* The smoke run at the norm-matched 1.0× coefficient raised benign
  refusal 0 → 0.33 and halved neighbourhood accuracy; the main grid uses 0.5× with a sweep, a choice
  made before the main grid but after a smoke observation.
- *LoRA cells are head-masked but rank/steps are fixed* (no per-condition tuning). Head masks are exact
  for q and o; k and v are trained for the whole GQA group of a selected head (4 query heads share one
  KV head), so a "circuit head" adapter also changes the keys and values read by its group mates: the
  58 heads of `circuit_s0` put k/v adapters under 144 heads (random_s0: 208). 13 b1 LoRA cells were
  rebuilt after an OOM fix that changed the loss implementation but not its value (exactness test
  `tests/test_units.py::test_label_position_loss_matches_hf`).

**Audit and statistics**
- *Refusal is judged by a pattern detector.* AuditKIT's `RefusalDetector` is English-only and reported
  0 % refusal in hi/ar/fr/sw on generations that were refusals; `neb/judge.py` adds per-language onsets.
  No LLM judge is used, so partial compliance after a refusal onset is not caught. Base refusal on the
  full 40-prompt cyber pool: en 0.88, fr 0.78, sw 0.60, hi 0.48, ar 0.43 (held-out 12: 0.92 / 0.83 / 0.58 /
  0.50 / 0.58).
- *Held-out harmful sets are small* (12 prompts per language per family), so a refusal Δ of 0.08 is one
  flipped generation; the Hub withholding rule treats one flip as a decrease by design.
- *AuditKIT's `paired_bootstrap` p is not a significance test* (uncentred resampling gives p ≈ 0.5 when
  every pair shares a sign; observed Δ = +0.19, CI [+0.13, +0.26], p = 0.49). Only percentile-bootstrap
  95 % CIs of the paired mean difference are reported.
- *WikiText perplexity and MMLU are blind to the measured damage*: MEMIT raises the
  never-edited top-1 error 0.02 → 0.26–0.65 while perplexity moves ≤ 1 % and MMLU ≤ 6 points.
- *Sweep cells get a reduced, English-only audit* (amendment A1) and B1 steering efficacy is measured on
  the seed's fact fold (amendment A2); both are recorded in `docs/preregistration.md` and were decided
  before any sweep cell was audited.
- *Cross-circuit L4 is read against a resample floor that most cells do not exceed.* Bystander CAD is
  below the base-vs-base resample floor for every condition except the layer-0 MEMIT and its
  perturbation control (and, on B2, the SafeTune direction comes within 0.05 of it), under both circuit
  definitions and with either IOI task, so the H2 L4 differences are differences between values smaller
  than the base attribution's between-subset variation (deterministic on the 32 examples used; how
  they vary with the example set was not measured).
  The floors themselves differ by bystander (0.33–0.51) and rank with the bystander's overlap with the
  target, which is why H4's raw and floor-adjusted readings disagree. LoRA-at-the-circuit and steering
  bystander CADs stay 0.08–0.35 below the floor even at 64 examples / 10 IG steps (power check, §4.3),
  so at this budget the layer does not separate their bystander edge changes from resampling variation,
  and H3's "no L4 advantage" for such cells is weak evidence.


**Brief requirements not met, substituted, or deferred** (re-checked line by line on 2026-11-06; items
closed by the third audit are marked *closed*, say where, and were run by `run_gaps_*.sh`, finished
2026-11-07 09:55 IST)
- *B1 size*: 180 false facts, not the ≥ 200 the brief asks for. Owner decision (2026-11-06): no new
  pool and no B1 rerun.
- *Weight interventions norm-matched across conditions* (brief §3.3 item 3): replaced by a norm-matched
  random perturbation control per weight cell. Owner decision (2026-11-06): no norm-matched cells.
- *L1 one-hop ripple set*: *closed* (A9, `data/b1_ripple.py`, `audit/ripple.py`; run, §4.3).
- *L5 relearning probe for B2*: *closed* (A10, `audit/relearn.py`; run, §4.4).
- *L5 arbitration spread / AURC as Anand defines them*: *closed* (A12, `audit/arbitration.py`; §4.3,
  §4.4); the earlier family was a context-free margin.
- *H5 comparison site*: causal tracing *closed* (A8, `interventions/causal_trace.py` + MEMIT cells at the
  traced layer; §4.2, §4.3); the Taylor-layer comparison stays reported.
- *Non-English fact circuits with 3 seeds*, *stability and the 1–30 % size curve for every circuit*:
  *closed* (A11; §4.1). The per-type (A5) evaluation was not run for the new non-English seeds (never
  intervened on).
- *§7.4 ROME smoke on Tiny Aya*: *closed* (`checks/results/rome_smoke.json`, §4.2): ROME works when CircuitKIT's
  `RomeHandler` is called directly; the circuit-guided `edit_via_circuit` entry point fails on a loaded
  `Circuit` (CircuitKIT API gap, filed in the JSON); the first run of the check ran out of MPS memory
  (the editor clones every fp32 parameter) and the check now runs on CPU. The edits remain MEMIT.
- *§12.5 steering directions validated against full-component patching*: *closed*
  (`checks/results/steer_direction.json`, §4.3, §4.4): no subspace illusion at the circuit nodes; on B1 the
  vector carries 1–3 % of the components' effect.
- *§11 profiler and deterministic algorithms*: *profiled, determinism not available*
  (`checks/results/profile_lora.json`): 20 LoRA steps 44 s (462 tokens/s, peak 9.3 GB, no mask leak), 95 % of CPU
  time is spent waiting on MPS synchronisation, so the CPU profile cannot attribute GPU time;
  `torch.use_deterministic_algorithms(True)` is rejected on MPS (`index_put_with_accumulate` has no
  deterministic kernel), so bit-exact reruns are not guaranteed on this machine. Seeds are fixed;
  main steering cells and their sweep twins reproduce exactly, LoRA runs are not checked for bitwise
  equality.
- *§3.3 back-translation of the B1 parallel prompts*: *closed* (`checks/results/b1_backtranslation.json`;
  14 template rows per language, one subject each). The first run reused the B2 prompt and the model
  *answered* the questions instead of translating them (back-translations such as "Pakistan"), so its
  F1 was meaningless; with the question quoted and "do not answer" the mean token-F1 is hi 0.58, ar
  0.69, fr 0.60, sw 0.69, and 3 / 1 / 3 / 2 rows fall under 0.5. All flagged rows are paraphrase
  templates whose back-translation is a faithful question form of an English imperative ("What is the
  capital of this country: Saint Lucia?" for "Tell me the capital of Saint Lucia"), except Swahili
  continent ("What is the name of this country: Saint Lucia?"), which a Swahili speaker should check.
  The back-translator is the model under study. B2 had its own check from the start.
- *HF ↔ TL equivalence for k/v/gate/up LoRA factors*: *closed* (`checks/results/apply_equivalence.json`, case
  `lora_kv_gate_up`: logit-delta cosine 0.9997, |Δ| 25.4 HF vs 25.8 TL).
- *AuditKIT `compare` for before/after comparisons with CIs*: *closed* — every `scorecard.json` holds
  `ak.compare` per family on AuditKIT RunResults rebuilt from the stored per-sample scores, with a
  percentile CI (AuditKIT's own p resamples uncentred differences and is not a null test).
- *CircuitKIT selective-LoRA masks*: our masks are built in `interventions/lora.py`; they equal CircuitKIT's
  selection for all 28 pre-registered restricted LoRA cells (`checks/results/lora_masks.json`); the 11 A15
  LoRA cells are not in that check (each records `mask_leak_max` 0.0 at training).
- *Dataset hashes for every stage*: data scripts wrote no manifests; `data/MANIFEST.json` (post hoc) holds
  sha256, rows and producer for every data file.
- *L2 via SafeTune `evaluate`*: SafeTune's judges are vLLM/CUDA-only and WildGuard is gated, so L2 is
  scored by AuditKIT's `RefusalDetector` with multilingual onsets and re-judged by SafeTune's own
  detector; SafeTune supplies the refusal-direction baseline instead (A4). Not changed.
- *Robustness and generalization pillars* skipped for every circuit: robustness is uninterpretable for a
  signed metric (brief §8) and the design has no related task for generalization. Not changed.
- *Brief §8 (no generated harmful content in the repo or on the Hub)*: violated by the 2026-11-03 Hub
  push (audit `hf.json` replies, baseline previews); fixed locally (A7, `data/sanitize.py`, sanitized
  publish staging, `just check` scans). The Hub is unchanged until the owner publishes; that publish must
  use `squash=true` so the old revisions are removed as well.
- *Repository layout*: `circuits/` lives in the sibling repo `tiny-aya-circuits` (owner's split,
  2026-11-05), not in this repo as brief §5 lists.
- *Publication and the Space*: pushed only on an explicit instruction from the owner. The Space app
  shows judgments, never replies, for harmful prompts.

**Amendment A18 (2026-11-08 14:13 – 2026-11-09 05:41 IST; `scripts/chains/13_a18.sh`, logs `.cache/logs/a18*.log`; closes the fifth audit's one pipeline defect)**
- *What was wrong.* `interventions/cell.py` applied a LoRA cell as `W += ΔW` on the bf16 weights;
  measured against the real weights, 31 / 31 / 38 / 43 / 31 % of the non-zero entries of ΔW rounded
  to zero for circuit_s0 / random_s0 / circuit_joint_s0 / unrestricted_s0 / perturb_circuit_s0
  (relative L2 error 0.254 / 0.260 / 0.321 / 0.368 / 0.255; MEMIT deltas 0.002–0.040). PEFT trains the
  adapters in fp32 and adds the term to the bf16 output with one rounding, so the audited model was not
  the trained one.
- *What changed.* LoRA cells are applied as forward hooks adding that fp32 term (`lora_term`,
  `add_term`) on both model stacks; on the TL side the projection inputs are cached at
  `hook_{q,k,v}_input`, `hook_mlp_in`, `hook_z`, `mlp.hook_post` and the term is added at the projection
  outputs inside EAP-IG's hooks. Verified: hook = PEFT forward to 0 ulp on all 96 adapted modules of
  `b1_lora_circuit_s0` (the merged weights differed on 24 % of output entries, up to 0.93 ulp); HF ↔ TL
  logit-delta cosine 0.998 / 0.9994 / 0.997 on three real cells; merged vs hook logit-delta cosine 0.999
  (max-abs disagreement 0.16–0.50, the HF ↔ TL bf16 floor is 0.25). All 79 LoRA cells (45 B1 incl. the
  11 A15 cells and 17 controls, 34 B2) were re-audited on all five layers, with the L4 target,
  bystanders and `ioi_fixed` attributed in one pass per cell (`audit/l4.py`), then ripple, arbitration,
  relearning, the pillar and shared masks, the two LoRA power-check cells, tables and analyses; the
  superseded files are kept under `runs/_superseded/2026-11-08_A18_lora_bf16_merge/`.
- *What moved.* Per cell, the mean absolute change over the 79 cells is 0.005 in efficacy (max 0.017),
  0.003 in L3 (max 0.009), 0.002 in bystander CAD (max 0.005), 0.004 in target CAD, 0.0005 in the
  WikiText ratio; condition means are unchanged to two decimals (Spearman old vs new 0.92–1.00 across
  the collateral columns). Two readings move: the A15 LoRA L3 advantage (−0.011, p = 0.002 → −0.008,
  p = 0.047) no longer survives Holm (B1 7 of 20 survive, was 8), and the B2 LoRA L3 circuit − random
  difference (−0.007, p = 0.036 → −0.008, p = 0.050) passes the last Holm step at exactly its threshold.
  No H1–H5 verdict changes.
- *Run defects fixed on the way.* The B2 HF pass ran out of MPS memory at its 16th cell of one process
  (allocator growth across cells, as the LoRA grid had shown): `audit.cells_per_process: 12` makes
  `run_audit.py` exit 3 after 12 new cells and release the cache after each, and chains loop on it.
  `checks/check_lora_masks.py` compared every cell against CircuitKIT's per-type top-10 % rule, which
  the A15 joint sets (40 heads + 21 MLPs) cannot satisfy; it now checks every cell against the
  selection its own node list implies (and the pre-registered cells against CircuitKIT's rule too):
  39 of 39 match, no masked entry trained.
- *EAP-IG vs patching on `ioi_fixed`* (run in the chain, `checks/results/eap_vs_patching.json`): ρ = 0.67 on
  signed scores, 0.45 on magnitudes, sign agreement 0.90, below the brief's 0.7 rule (§4.1). The
  remedy is the queued amendment A19 (per-example attribution with a measured quadrature rule; prepared
  here, run elsewhere).

**Fifth audit (2026-11-08; full record `docs/audit-2026-11-08.md`; documented, then closed by A18)**
- *LoRA deltas were rounded when merged into bf16 weights* (B.1): fixed and re-audited under A18 above;
  the tables of §4 now describe the trained adapters.
- *`n_trainable` counts adapter entries, not trained entries.* Masked `lora_B` rows (q/k/v) and
  `o_proj` `lora_A` columns stay exactly 0 but are counted. Trained entries: B1 circuit 2.8–3.0 M
  (reported 3.5–3.7 M), random 3.2–3.5 M (4.1–4.6 M), inverse 3.0–3.2 M, Taylor 2.4 M (2.9 M),
  circuit_joint 7.8–8.2 M (8.4–8.8 M), random_joint 8.2–8.6 M (9.0–9.4 M), unrestricted 15.1 M. Ratios
  change (unrestricted / circuit 5.1–5.4× instead of 4.1–4.3×); the capacity reading does not.
- *The fact circuit is discovered on a different relation mix than the edited facts.* The 64 discovery
  pairs of seeds 0 / 1 / 2 contain 3 / 3 / 2 office-holder questions (city → country 36–40, capital 8–14);
  137 of the 180 edit targets are office holders. "Confining the edit to the circuit" means confining it
  to a circuit found mostly on city → country and capital recall.
- *The one locality advantage rests on three seeds* (A15 LoRA L3: per-seed −0.013 / −0.007 / +0.000
  after A18, paired t p = 0.21; was −0.014 / −0.013 / −0.002, p = 0.13) and does not meet the
  pre-registered H2 rule (null against the inverse set); after A18 it no longer survives Holm; see §4.3
  "A15".
- *MEMIT "size, not content"* holds at MLP 0 only; elsewhere the real edit causes 4–12× the L3
  collateral of equal-norm noise (§4.3 H2).
- *The A15 overlap column* in `hypotheses_b1.md` counts overlap with the per-type circuit (inverse_joint
  shows 1 node; its true overlap with circuit_joint is 6 / 6 / 8 MLPs).
- *A15 steering sweep cells.* The A15 amendment says "main cells only (no steering sweep)", but the grid
  built and audited 44 sweep twins of the A15 steering cells; they appear only in the matched-efficacy
  table of `hypotheses_b1.md`.
- *Low-confidence targets.* 68 of the 180 targets have top-1 probability < 0.3 for the wrong answer, so
  for them an edit inserts a fact rather than overturning a confident belief.
- *Arbitration A sums token log-probabilities without length normalisation.* The two candidates differ
  in token count on 88 of 138 probes (up to 6 tokens; mean difference −0.03), so follow rates are not
  biased on average but |A|, which orders decisions for AURC, partly orders by length; the reported
  values are cell − base deltas on the same probes, which cancels most of it.
- *Provenance.* The git history of this repo holds only the two skeleton commits of 2026-11-02 and the
  sibling repos are not git repositories, so every manifest records `git_sha 416938a` with no dirty
  flag and no artifact is tied to the code that produced it. The pre-registration freeze was verified
  byte-for-byte against the local 2026-11-02 backup.

**Fourth audit (2026-11-07; full record `docs/audit-2026-11-07.md`)**
- *H4 is confounded with circuit identity.* Amendment A17 (exploratory) splits each bystander's own
  edges by whether they touch the intervened components (`audit/l4_shared.py`, from the kept edges):
  the magnitude-free statistic Λ is small for LoRA and steering at the circuit (B1 +0.07 / +0.05, B2
  +0.20 / −0.27), not consistently different from the perturbation controls (B1 +0.24, B2 −0.01), and
  negative for MEMIT at MLP 0 (B1 −0.24, B2 −0.21); the raw-CAD split D is magnitude-confounded and takes
  either sign by the intervened components (13 of 28 B1 rows and 8 of 18 B2 rows positive) while it is
  negative for every row by the target set. No verdict changes.
- *No cell had intervened on a faithful circuit.* Amendment A15 adds B1 LoRA and steering cells on the
  joint-ranked pillar set (patching 0.33–0.56) with count-matched random and inverse sets (§4.3 "A15").
  LoRA on it reaches the unrestricted adapter's efficacy (as do the count-matched random and inverse
  sets: capacity) and shows the study's one locality advantage, −0.008 on L3 against count-matched
  random (uncorrected p = 0.047 over pairs, not surviving Holm after A18; three seeds, per-seed paired
  p = 0.21; null against the inverse set, so the pre-registered H2 rule is not met), null on L4. The steering half is vacuous (efficacy ≤ 0.07 for the circuit
  and its controls) and its differences are ≤ 0.015 (none survives Holm). The chain that built the
  cells stopped the steering grid after three cells (`grid.cells_per_process`), so the steering
  controls were built by `scripts/chains/11_a15c.sh` afterwards.
- *L4 sensitivity measured.* `audit/l4_power.py` at 64 examples / 10 IG steps halves the resample floors
  (0.54–0.64×) while the cells' CADs fall less (target 0.38–0.75×, bystanders 0.71–1.08×); on the
  per-type masks the strongest weight interventions come within 0.02 of the floor and LoRA at the
  circuit and steering stay 0.12–0.24 below, on the joint masks unrestricted LoRA and Taylor-layer MEMIT
  exceed it (+0.02 to +0.04) (§4.3). The layer is under-sampled, not blind; a full
  rerun at that budget (≈ 7× the L4 campaign) was not done.
- *Corrected IOI bystander (A16).* `ioi_fixed` replaces the built-in task in every L4 mean and in H4:
  the L4 H2 differences move by ≤ 0.05 and no H2 verdict changes; H4 moves from "consistent in
  direction" to inconclusive, because the raw statistic holds by its letter (+0.72 / +0.69 per cell)
  while the floor-adjusted one reverses (−0.25 / −0.39) and the floors rank with overlap. The
  length-matched refusal task `b2m_en` reproduces the refusal circuit and its unfaithfulness.
- *Multiple comparisons.* Bootstrap p-values and a Holm step-down count over the H2 tests
  (`hypotheses_*.md`; the B1 family is the 12 pre-registered tests plus 8 post-hoc A15 tests): on B2, all 12 survive with the corrected bystander (8 of 12 with the built-in
  task); the LoRA L3 "advantage" (−0.008, p = 0.050 after A18) passes as the last step at exactly its
  threshold and is one-tenth of the scale. B1: 7 of 20 survive after A18 (the four MEMIT MLP-0
  disadvantages, the two steering L4 differences and steering L3 circuit − random); the A15 LoRA L3
  advantage (p = 0.047), the A15 steering differences (≤ 0.015, p ≥ 0.021), the A15 L4 differences and
  every per-type LoRA difference do not.
- *Exact ratio floors and resolution*, *IOI plateau and pair misalignment*: see the bullets above and
  the fourth audit B.1–B.2, C.1–C.2.

**Third audit (2026-11-06; full record `docs/audit-2026-11-06.md`)**
- *L4 base-edge scale bug (A13).* EAP-IG scores scale as 1/batch_size with our batch-mean metric; the B1
  discovery-seed-0 base edges of b1_en, b2_en and IOI were cached at batch 4, every cell at batch 2, so
  those CAD values were inflated (a no-op-sized random LoRA perturbation read CAD 1.0 with edge Spearman
  0.99). `audit/l4_rescore.py` recomputed the three base files at batch 2 and rescored the 32
  affected cells from their kept edges (every task, seed and behaviour now has a median cell/base edge
  ratio of 0.96–1.00); the B1 L4, H3, H4 and exploratory readings in §4.3–4.4 are restated from it.
  What changed: no real B1 intervention's target CAD exceeds the floor any more; H4 on B1 goes from
  "refuted" to "consistent in direction, not established"; H3's one B1 exception disappears.
- *Evaluation sets.* L1 neighbourhood (60), paraphrase (40) and L5 never-edited (100) facts are all
  `capital` facts (first by key), while 76 % of the edits are office holders; the B2 "other category" is
  40 MultiJail theft/fraud prompts; the 28 benign near-miss prompts are both the steering source set and
  the over-refusal evaluation set; held-out harmful (12) and benign (28) are below the pre-registered
  minimum n of 40.
- *B1 steering vector*: target = 64 known *capital* questions, source = wrongly answered facts (mostly
  office holders), so the direction mixes question template with knowledge.
- *Answer matching*: the first-word rule admits wrong people ("Nicos Anastasiades" for gold "Nicos
  Christodoulides"); a stricter rule lowers B1 efficacies by 0.02–0.11 and leaves H1 at 0.430 (MEMIT)
  and 0.400 (LoRA).
- *Targets*: about 13–16 of the 180 "false facts" are miscorrections or contested (London → Canada,
  Hamilton → Canada, Santa Ana / San Juan → United States, Belfast → United Kingdom, Hargeisa →
  Somaliland, Sevastopol → Ukraine, ambiguous city names, contested capitals); golds are a Wikidata
  snapshot of 2026-11-02.
- *Refusal circuit data*: 45 of 207 English refusal EAP pairs are not length-matched (right padding), so
  their clean and corrupt positions are misaligned; the length-matched task `b2m_en` (A16) gives the
  same circuit and the same unfaithfulness, so this is not why the refusal circuit fails.
- *Statistics*: pair-level CIs treat languages, families and bystanders inside a seed as independent; the
  seed-cluster CIs now printed beside them leave every B1 verdict standing and remove two B2 steering
  comparisons (L3 and efficacy_xl circuit − random). Discovery seed, fact fold and training seed move
  together; LoRA/MEMIT locality is compared at unequal efficacy; Holm is applied to the H2 tests only,
  not across the dozens of other intervals. With three seeds the seed-cluster bootstrap has 10 distinct
  resamples and is not a valid interval: it can be narrower than the pair-level CI (B1 LoRA L4 circuit
  − random: pairs [−0.006, +0.031], clusters [+0.009, +0.025]), so "seed-cluster CI excludes 0" is not
  evidence of robustness across seeds. `pairs_for` pairs each random draw with the circuit cell of its
  discovery seed, so with five draws the circuit cells of seeds 0 and 1 enter twice.
- *Definitions*: "circuit distance" in the tables is node turnover, not Chen et al.'s edge-level distance
  (`l4_target_edge_l1` added); AS-NC is node-level on fact prompts; the overlap is a node-level weighted
  Jaccard over raw scores of differently scaled circuits (normalised values give the same H4 ranks); the
  AUC is a partial-range node-level analogue of MIB's CPR.
- *Format parity* was measured in English only.

**Mismatches found in the second audit (2026-11-05 14:30 IST; full record `docs/audit-2026-11-05.md`)**
- *Three node-set definitions shared the word "circuit" — resolved by amendment A5 (2026-11-05/06).* The
  pre-registered intervention circuit ranks heads and MLPs separately by |signed row sum| (58 heads + 4
  MLPs); the faithfulness pillars, size curve, random and minimality controls had been evaluated on
  CircuitKIT's joint-ranked set (21 MLPs among the 61), and L4's masks on a joint ranking by |signed row
  sum| (28 MLPs). Fix: `aya_circuits/cellset.py` evaluates the pre-registered set through the same
  CircuitKIT path for all 16 circuits (§4.1 second table), and `audit/l4_cellset.py` re-attributes every
  audited cell and scores L4 on the pre-registered masks (the joint-set values it also computes reproduce
  the stored `l4.json` numbers to 0.0 on every cell; per-cell bystander-CAD Spearman between the two
  definitions 0.995 on B1). All L4 numbers in §4.3–4.4 now use the pre-registered set; the joint-ranked
  values stay in the CSVs (`*_joint`) and the sensitivity tables of `hypotheses_*.md`. What the fix
  changed: the per-type fact set restores 0.4–3 % under patching (joint 33–56 %), so H3's gate fires on
  B1 as well and the "faithful circuits show no advantage either" reading of v0.7 is withdrawn: no cell
  in this study intervened on a set that reproduces the behaviour on its own. H2 and H4 verdicts are
  unchanged under either definition.
- *Necessity measured (A5).* The set ablated under mean replacement with the rest kept (`necessity` in
  `circuits.md`, with 5 random per-type sets under the same call): fact per-type sets 0.49–0.52 vs
  random 0.53–0.69 (necessary beyond chance); refusal per-type sets 0.48–0.53 vs random 0.34–0.40 (not
  necessary; the joint refusal set is, 0.25 / 0.19 / −0.04); IOI 0.07–0.15 vs 0.27–0.40 (necessary).
  Zero-ablation necessity is reported too but sits on the 0.5 floor for every fact set.
- *LoRA conditions are node-count matched, not parameter matched.* Adapters attach per touched layer, so
  trainable parameters are 2.85 M (Taylor), 3.49–3.70 M (circuit), 3.81–4.13 M (inverse), 4.13–4.55 M
  (random), 15.1 M (unrestricted); the random head sets' higher efficacy (0.64 vs 0.41) comes with ~22 % more
  adapter entries, which the H1 LoRA reading in §4.5 did not say. (These counts include masked entries
  that never train; trained-entry counts are in the fifth-audit block above.)
- *AS-NC prompts.* AS-NC is computed on 32 English fact prompts for both behaviours, so for B2 it is not
  measured on the target task's prompts.
- *Grid and audit manifests record no input-dataset hashes* (`inputs: {}`); only the discovery and
  filter manifests do, so brief §11's "sha256 of every input dataset" holds for two of five stages.
- *B2 non-English prompts are translated by Tiny Aya itself* (no human ar/sw in the cyber category;
  back-translation F1 flags 3/4/3/10 of 68 rows for hi/ar/fr/sw), so the L3 refusal numbers carry the
  model's own translation quality.
- *Measurement insensitivity.* Base MGSM accuracy is 0.10–0.17 (30 items, 160-token greedy answers) and
  IFEval 0.53 on 30 prompts; their per-cell deltas (±0.1) are at noise level and are not used in any verdict.

**Compute**
- All work ran on one Apple-silicon machine under a 30 GB cap. OOMs and their fixes: faithfulness
  (batch 8 → 4 → 2 for langid), MEMIT fp32 statistics (parameters frozen, CircuitKIT memory patch 0005
  after a parity check: node Spearman 0.9993, top-10 % Jaccard 1.00), LoRA (label-position loss; peak
  26.7 → 8.3 GB), L4 (batch 4 → 2), plus a 3-cells-per-process cap for the LoRA grid. MPS bf16 was
  checked against fp32 CPU (top-1 agreement 1.0, mean |Δlogit| 0.02).
- One model, one circuit-discovery method (EAP-IG), one circuit size for the main grid. Nothing here
  generalises beyond Tiny Aya without replication.

## 6. Changelog
- v0 (2026-11-02): skeleton.
- v0.21 (2026-11-12 08:30 IST): A22–A24 run and reported (§4.7, §5 "Found on 2026-11-11/12"):
  a refusal adapter with size-matched controls, a dev-tuned SafeTune baseline, an LLM judge, a
  five-language fluency check, two PEFT exports with exact parity; `audit/standard_checks.py`
  (31 of 92 cells that pass perplexity + MMLU are damaged at 0.10; noise 2 of 68);
  `docs/article.md`, `docs/abstract.md`, `docs/upstream-issues.md`, `docs/model_cards/`. No earlier
  number changed.
- v0.20 (2026-11-09 19:19 IST): text only, no run re-done. Pooled H3 monotone half restated from the
  regenerated `hypotheses_b1.md` (|ρ| ≤ 0.21, p ≥ 0.41; the B1 file's pooled block had been written
  before B2's last re-analysis and read B2's earlier points; B2's file was current); the chains moved to
  `scripts/` and `make` became `just` (paths in this report updated; changelog entries keep the old
  names). No verdict changed.
- v0.19 (2026-11-09 16:10 IST): amendments A19 (inside view with a measured quadrature rule, jackknifed paired CAD, TOST), A20 (certified circuit; steering and LoRA on it against entry-matched controls) and A21 (MEMIT at 17 sites, uncapped / common norm / noise) run on the puget machine and reported in §4.6 and `docs/results/puget-a19-a21.md`; §5 adds the audit order noise (fixed for A21), the scope of A19's SE and the cross-device notes. No H1–H5 verdict changes.
- v0.1 (2026-11-03 11:20 IST): §4.1 circuits and §4.2 cell inventory written from the finished circuit stage and grids; audit running.
- v0.2 (2026-11-03 17:45 IST): §4.3 B1 behavioural results (133 cells) from `audit/analyze.py`; L4, the late LoRA cells and B2 pending.
- v0.3 (2026-11-04 15:05 IST): §5 limitations and failure log written from `docs/notes.md` and `docs/results/circuits.md`; run resumed for the b2 audit, L4 sides and late LoRA cells.
- v0.4 (2026-11-04 23:10 IST): §4.3 completed with the L4 side (CAD vs resample floor, AS-NC), H3, H4 and the rebuilt LoRA cells; B2 audit running.
- v0.5 (2026-11-05 02:45 IST): §4.4 B2 behavioural results with amendment A3 (multilingual efficacy, English at ceiling); B2 L4 running.
- v0.6 (2026-11-05 05:50 IST): §4.4 completed with the B2 L4 side, H4, pooled H3 and the exploratory efficacy-vs-target-CAD finding; `run_tail.sh` finishing the langid_en seed 1 control and the final tables.
- v0.7 (2026-11-05 06:00 IST): final; `langid_en` seed 1 control added (16/16 circuits with controls), §5 updated.
- v0.8 (2026-11-05 11:30 IST): SafeTune refusal-direction baseline (3 cells, A4) and the AuditKIT-vs-SafeTune judge robustness check added to §4.4; §2 lists the track-repo chain.
- v0.10 (2026-11-05 14:30 IST): second audit pass (code, artifacts and every number in this file re-checked against `docs/results/*` and `runs/`; record in `docs/audit-2026-11-05.md`): H4 overlaps now exclude the `input` pseudo-node (`audit/analyze.py`, values match `overlap.json`; ranks and verdicts unchanged); stale pre-A4 numbers in §4.4 (exploratory Spearman 0.69/n=54 → 0.73/n=57, H4 floor-adjusted CADs, cell count 162) corrected; §5 b2 30 % curve value 0.70 → 0.45; §5 pool vs held-out refusal rates relabelled; AS-NC named as activation shift on fact prompts; the MEMIT `circuit_deep` refusal exception and LoRA's other-category refusal (−0.075) stated; new §5 block "Mismatches found in the second audit".
- v0.9 (2026-11-05 13:30 IST): brief audited line by line: §2.1 rationale table, §4.1 AUC and EAP-vs-patching check, §4.5 alternative explanations, §5 list of unmet or substituted brief items; B1 count corrected to 180; `make check` passes in all three repos; per-cell `scorecard.json` written next to every cell.
- v0.11 (2026-11-06 04:30 IST): amendment A5 executed: pillars, same-path controls and necessity on the pre-registered per-type circuit for all 16 circuits (§4.1 second table), L4 re-attributed for all 185 cells and scored on the pre-registered masks (joint-set values reproduce the stored ones to 0.0); §4.3–4.4 L4, H3, H4, AS-NC and the exploratory correlation restated on that definition with the joint-ranked values in parentheses; H3 reading corrected (every cell gated; monotone half untested); §5 second-audit block updated. Amendment A6 (fold-disjoint B1 steering sources) running; its cells are re-audited and §4.3 steering numbers re-checked in v0.12.
- v0.12 (2026-11-06 08:00 IST): amendment A6 executed (B1 steering cells rebuilt with fold-disjoint source prompts, 85 cells re-audited on all layers; superseded artifacts kept under `runs/_superseded/`); §4.2 norm, §4.3 steering sweep / H2 / H4 / exploratory numbers and §5 updated; no verdict changed. `make check` passes in all three repos.
- v0.13 (2026-11-06 20:15 IST): third audit (`docs/audit-2026-11-06.md`, amendments A7–A13): stale and false statements corrected, seed-cluster CIs added, B1 L4 paragraphs marked pending rescore, gap-closure runs queued.
- v0.18 (2026-11-09 06:30 IST): amendment A18 run (`run_a18.sh`): every LoRA cell applied as exact fp32 low-rank hooks (the trained adapter) instead of bf16-merged weights and re-audited on all layers (79 cells; one attribution pass per cell for target, bystanders and `ioi_fixed`); §4.2 application note, §4.3 / §4.4 LoRA numbers, A15, A17, H4, relearning, edge distances and §5 restated; the A15 LoRA L3 advantage falls below the Holm line (B1 7 of 20), the B2 LoRA L3 difference sits at p = 0.050; `ioi_fixed` EAP-vs-patching ρ 0.67 recorded as failing the brief's rule; `audit.cells_per_process` and the joint-set mask check added. No H1–H5 verdict changes.
- v0.17 (2026-11-08): fifth audit (`docs/audit-2026-11-08.md`), text only, no run re-done: stale numbers restated from the result files (steering L3 +0.013, B1 exploratory correlation n = 79, pooled H3 |ρ| ≤ 0.24, efficacy–adapter-size Spearman 0.85 over 28 cells, A15 steering CAD excess, neighbourhood 18–39 points, b2m_en 0.32–0.39, layer-0 target probability 0.08–0.12, B1 cell count 231); the power-check reading corrected (floors halve, cells' CADs do not; joint masks exceed the floor); the A17 raw-split and "trained = noise" sentences corrected; MEMIT "size, not content" restricted to MLP 0; A15's three-seed unit and unmet H2 inverse rule stated; fifth-audit caveats added to §5 (bf16-merged LoRA deltas, adapter-entry vs trained-entry counts, GQA k/v masks, discovery relation mix, seed-cluster CI validity, provenance). No H1–H5 verdict changed.
- v0.16 (2026-11-08 02:45 IST): amendment A16 closed (`run_a16.sh`): `ioi_fixed` discovered and evaluated on three seeds (necessary, 2–5 % sufficient at 10 %, 43–57 % at 30 %, random sets 0.000, stability 0.86) and re-attributed on all 213 audited cells; every L4 mean, the L4 H2 tests, the Holm counts, A17 and H4 restated with it (H4: inconclusive; raw per-cell Spearman +0.72 / +0.69, floor-adjusted −0.25 / −0.39, floors rank with overlap); `b2m_en` shows the refusal circuit's unfaithfulness is not a pair-alignment artefact; `checks/results/metric_span.json` extended to both tasks. No H1, H2, H3 or H5 verdict changed.
- v0.15 (2026-11-07 20:40 IST): fourth audit (`docs/audit-2026-11-07.md`): brief verified line by line; exact per-task ratio floors and bf16 resolution (`checks/results/metric_span.json`), necessity re-referenced (`nec0_*`); IOI plateau explained, the IOI pair misalignment and the built-in task's metric position found (`checks/results/ioi_plateau*.json`, `checks/results/ioi_position*.json`); Holm count over the H2 tests (B1 8/20, B2 8/12); amendment A15 run and reported (B1 LoRA + steering on the faithful joint set with count-matched random / inverse: LoRA parity from capacity, the study's one locality advantage on L3, steering vacuous); the L4 power check (64 / 10 halves the floors and the cells' CADs; under-sampled, not blind); amendment A17 (within-bystander overlap split; partitioned CAD is magnitude-confounded); independent unit checks of the CAD edge masks and statistics. Amendment A16 (`ioi_fixed`, `b2m_en`) running; its paragraphs are marked [pending A16].
- v0.14 (2026-11-07 11:30 IST): third-audit runs finished and restated. B1 L4 rescored (A13): bystander excess and paired tests restated (MEMIT circuit − random +0.42), target-circuit CAD no longer above the floor for any real intervention, H3's B1 exception gone (18/18), H4 per-cell statistic added to `audit/analyze.py` (B1 and B2 positive; reading "consistent in direction, not established", B1 was "refuted"), exploratory B1 efficacy–target-CAD correlation restated (0.68, between interventions only). New: causal-tracing site and MEMIT cells (A8; H5 refuted against it), ripple (A9), relearning (A10), arbitration (A12), edge-level target distance, steering-direction check, ROME smoke, profiler and determinism, B1 back-translation (prompt fixed after the first run answered instead of translating), k/v/gate/up equivalence, non-English circuit seeds 1–2 and stability / size curves for every circuit (A11).
