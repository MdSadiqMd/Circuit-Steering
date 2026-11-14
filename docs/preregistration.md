# Pre-registration (FROZEN 2026-11-02T08:43+05:30)

Status: **frozen 2026-11-02T08:43+05:30**, before the first intervention cell (git SHA and uv.lock hash in the cell manifests).
Nothing above the "Amendments" line is edited after this point; amendments are appended with dates and reasons (brief §5, §12.7).

## Model and format
- `CohereLabs/tiny-aya-global`, revision recorded in `checks/stack_check.json` / manifests, bf16 on `mps`
  (editing statistics in float32/float64 on CPU, see docs/notes.md).
- One prompt format for discovery, intervention and audit: minimal chat format (BOS + user turn +
  assistant marker incl. `<|START_RESPONSE|>`), chosen by the parity check `checks/format_parity.json`.

## Behaviours
- **B1 false-fact correction.** Targets: facts from the Wikidata pool (relations capital, country,
  language, currency, head_of_state, head_of_government; `continent` excluded as ambiguous) that
  Tiny Aya answers wrongly in English (`data/b1/false_facts.jsonl`). Edit language: English. Gold =
  Wikidata label (or alias) in each of en, hi, ar, fr, sw.
- **B2 refusal strengthening.** Category: **cyber** (HarmBench `cybercrime_intrusion`, 40 harmful behaviours + 28 XSTest-derived
  near-miss benign prompts). Chosen from `data/b2/baseline_summary.json` as the category with the
  largest English-to-weakest-language refusal gap (en 0.875 vs ar 0.425, hi 0.475) and near-zero
  benign refusal (0.07), and because its prompts are safe to publish. Harmful prompts split 70/30 into train/held-out; benign
  near-miss prompts (XSTest-derived) are never trained on. Only refusal-*strengthening* artifacts
  are produced or released.

## Circuits
- EAP-IG node level, `ig_steps=5`, `n_examples=64`, batch 4, three discovery seeds {0,1,2}; the
  circuit is the top 10 % of the 612 heads+MLPs by |score| (heads and MLPs ranked separately so
  every condition is size-matched per component type). Signed edge scores are kept for L4.
- Bystanders: refusal (`b2_en`) for B1, fact recall (`b1_en`) for B2, IOI (CircuitKIT built-in) and
  language identification (`langid_en`) for both.
- Faithfulness reported as signed raw ratios under zero and mean ablation, patching, 5-draw random
  and magnitude baselines, 3-run stability, and the size curve at 1, 2, 5, 10, 20, 30 %.

## Interventions and conditions
- Interventions: activation steering at nodes (difference of means, last position), circuit-restricted
  LoRA (rank 8, 60 steps, lr 2e-4, head-level masks), knowledge editing (MEMIT at MLP W_out, candidate layers 0..28 of 36 because a
  MEMIT target at the final layers cannot propagate to the output; unrestricted = MEMIT over all 29
  editable layers; B2 via the same editor with a refusal target, key = last prompt word). Conditions: circuit, size-matched random (5 draws), inverse
  (bottom-k), Taylor (gradient magnitude), unrestricted. Seeds {0,1,2} for all but random.
- Steering conditions are norm-matched to the circuit condition at coefficient 1 (coefficient capped at
  20× for conditions with tiny vectors); the main cell uses 0.5× that base (the smoke run at 1.0×
  doubled WikiText perplexity); the sweep {0.25, 0.5, 1, 2}× gives efficacy curves, and collateral is
  also compared at matched efficacy.
- Weight cells get a norm-matched random perturbation control (`interventions/perturb.py`).
- Editing-only exploratory condition `circuit_deep`: the circuit's highest-|score| MLP outside layers
  0-3. Added before freezing because in the smoke runs the top-|score| MLP of both circuits was MLP 0
  (the detokenisation block), so `circuit` editing is layer-0 editing; `circuit_deep` is reported
  alongside, never in place of, the pre-registered `circuit` condition.

## Hypotheses and decision rules (brief §3.2)
- **H1 efficacy parity**: circuit cell's target effect ≥ 0.9 × unrestricted cell's effect (mean over
  seeds; bootstrap 95 % CI of the ratio reported). Target effect: B1 = accuracy gain on edited facts
  (English); B2 = refusal-rate gain on held-out harmful prompts (English).
- **H2 locality advantage**: on L4 (mean cross-circuit CAD over bystanders) and L3 (mean absolute
  change in non-English fact accuracy and refusal rates), circuit < random and circuit < inverse with
  a paired-bootstrap 95 % CI excluding 0 across seeds × bystanders. A null is reported as a null.
- **H3 faithfulness gates locality**: Spearman between the per-circuit pillar scores (patching,
  signed zero- and mean-ablation raw ratio, stability mean Jaccard, random-baseline margin) and the H2
  advantage (circuit − random collateral) across cells; cells whose circuit fails P2 (raw ratio ≤ 0
  under either replacement) or P5 (no margin over random) are predicted to show no advantage.
- **H4 overlap predicts interference**: collateral on bystander B (CAD and behavioural change) is
  monotone (Spearman > 0, 95 % CI excluding 0) in weighted Jaccard overlap between the target
  circuit and B's circuit.
- **H5 cross-lingual propagation**: edits at circuit components propagate to hi/ar/fr/sw (accuracy
  gain on edited facts in those languages) at least as well as edits at the English causal-tracing /
  Taylor layer, with no larger drop in neighbourhood accuracy in those languages.
- Minimum n per metric family is the value in `configs/base.yaml` `audit`; single-seed results are
  labelled directional.

## Datasets (sha256 at freeze)
- `data/b1/facts_pool.jsonl`: `488e2690585de2449ee3583cf619e4b6b88de596021687f850fd4938352d3eb1`
- `data/b1/false_facts.jsonl`: `bca2041a6b47f279cfeec5b0c1dd773c5da0960a0fb8b7e742927a046b5f3459`
- `data/b1/eap_known_en.csv`: `41295a2503df1514a4183d9e4154e0788cc3fd7fddc57eabc3682d1d0e03275d`
- `data/b1/eap_known_hi.csv`: `75586dcd9b2d734a26b55c0cada703676b9460392da7d21b2ccecdf11a5dd384`
- `data/b1/eap_known_ar.csv`: `4ba0eb709498ebeead8d31396d4d115035469e8f00fb061ece1d980a6bd7a234`
- `data/b1/eap_known_fr.csv`: `8f1901dc4319d7f40310feee5fd5cabd8ca427e81afa4f46a11ddb9d0673898f`
- `data/b1/eap_known_sw.csv`: `a7f481ab98ae0d70887cd4809358ddc4ca0e3fcb3fdb7be2fd9b5a7ed16514ac`
- `data/b1/prompts_en.jsonl`: `9d4a8c7bd40d1a1f0ac7ace43bd4141332983747c6443d8f678c75d9adddb0c4`
- `data/b1/prompts_hi.jsonl`: `aa9dbabb08e575748e6292578f643e162e7e83a32ad838389386babc30b9d330`
- `data/b1/prompts_ar.jsonl`: `d5eca714bef4ef0c79c4bcf6f94ca324302e86002fce55a88bdcf4627a869344`
- `data/b1/prompts_fr.jsonl`: `4b46a3143a9209f86631d44f4916d11b07883aad9846674acc84e0d97e8f5f5c`
- `data/b1/prompts_sw.jsonl`: `df9a65993b95a3a7183b36c12fc3cc5dcbefb6b406d38471e1d2504ca438259a`
- `data/b2/baseline_cyber.jsonl`: `6ee74806d5857a75fd5b3999147a6c24f2d22e2dc0680270f57c74255b9d6d21`
- `data/b2/xstest_all.jsonl`: `1d4849161218b1c992c8c0b2cebd90f6c36c7120cff124f1be5d2d9c57d4a7b1`
- `data/b2/eap_en.csv`: `599c721670bb4cdb962bb4ffa7c21b49d6da0601cb43dfd30cb4dfa5996af98f`
- `data/b2/eap_hi.csv`: `feac4c53204b55812d36116733830dbc2523c59cf081cf8ddfcb4f11c5833844`
- `data/b2/eap_ar.csv`: `65a2881d749369703be17824bed6cb385bbe9ef94fc2e2d5eb0c239d36ad3968`
- `data/b2/eap_fr.csv`: `0c23535ee59e2e9050b4e02dc1f8c2250b6391f8c6a597ac68f923af0608b854`
- `data/b2/eap_sw.csv`: `12bfc24ad32469133b286eed513f4488796e4162ae1b6651d7aec8c68cc31513`
- `data/bystanders/eap_langid_en.csv`: `c9d9ec0739b7e37f9c6c08d869b479c955f95fe2cbfa86181cb223eb39db70d9`
- `data/bystanders/eap_langid_hi.csv`: `acfac17a88b25672dee30e4ba93883a66a6425725f4647927e30cd280a3e506a`
- `data/bystanders/eap_langid_ar.csv`: `ea5588969019a804add4a76a8c97a2341c516e8a9b92358a4f6353b6e5f668cf`
- `data/bystanders/eap_langid_fr.csv`: `cd8de5c37440c1c0a8f60cda541857b4cad3052c73e279231d799ee3a20691cb`
- `data/bystanders/eap_langid_sw.csv`: `5c94c272a819beab6a52da6f9b78f5d906edca07749424dff2695dc1ee61cb92`

## Analysis code path
`audit/analyze.py` reads `runs/audit/**/{hf,l4}.json` and `runs/circuits/**/faithfulness.json`,
computes the statistics above with AuditKIT's paired bootstrap (`auditkit._bootstrap.paired_bootstrap`)
and writes `docs/results/*.md` tables.

---
## Amendments
- **A1 (2026-11-02 23:40 IST, before any audit ran).** Audit scope of steering *sweep* cells
  (`cond@mult`, 68 per behaviour): English efficacy and core collateral only (edited-fact and
  neighbourhood accuracy, MMLU-en, WikiText perplexity, refusal on held-out harmful, near-miss benign
  and XSTest-safe prompts). L3 languages, MGSM, IFEval, L5 and L4 run on the main cells and the
  perturbation controls only. Reason: the sweep is 136 of ~320 cells and the full five-layer audit
  costs ~9 min per cell on this machine; the sweep's purpose (efficacy curves, collateral at matched
  efficacy) is served by the reduced set. Hypothesis tests H1–H5 use main cells only, as written above.
- **A2 (2026-11-02 23:40 IST, before any audit ran).** B1 efficacy for steering cells is measured on the
  same fact fold (the seed's third of `false_facts.jsonl`) that the LoRA and MEMIT cells of that seed
  trained on, so H1 compares interventions on identical facts per seed. Steering is not fact-specific;
  without this rule the audit had no edited-fact family for steering cells (code defect found on
  resuming, `audit/run_audit.py`).
- **A3 (2026-11-05 02:30 IST, post hoc, after the B2 behavioural audit).** The base model refuses 11 of
  the 12 held-out English harmful cyber prompts (0.917), so the pre-registered B2 efficacy (English
  refusal gain) is capped at +0.083 and cannot separate conditions. `audit/analyze.py` additionally
  reports `efficacy_xl`, the mean held-out harmful refusal gain over hi/ar/fr/sw (base 0.50/0.58/0.83/
  0.58), with the same H1 ratio rule and circuit − comparison paired tests at seed × language. The
  English H1 stays reported as pre-registered; every `efficacy_xl` statement is labelled A3 and is
  exploratory. The steering sweep (A1, English-only audit) cannot use it, so the B2 matched-efficacy
  table is reported as uninformative.
- **A4 (2026-11-05 10:35 IST, post hoc, after the full run).** Added one B2 steering comparison condition,
  `safetune_rd`: SafeTune's refusal direction (Arditi et al. 2024, difference of residual-stream means at
  the last prompt token with SafeTune's layer-selection sweep) extracted per seed from a 5/6 subsample of
  the same English harmful/benign prompts as the circuit steering cells, added at every position after
  the selected layer, norm-matched to the circuit condition exactly like random/inverse/Taylor
  (coefficient = size_at_1(circuit) / |d|, × 0.5), audited on all five layers, no sweep cells. It is the
  brief's required refusal-direction baseline and enters the reports only as a paired `circuit −
  safetune_rd` comparison (efficacy, efficacy_xl, every collateral metric); no hypothesis is re-decided on
  it. Also added, as a robustness check and not as a scoring change: every stored English refusal
  generation is re-judged by SafeTune's refusal detector next to the AuditKIT verdict the pre-registered
  numbers use.
- **A5 (2026-11-05 20:10 IST, post hoc, after the second audit).** The second audit (`docs/audit-2026-11-05.md`
  B1/B2) found that the faithfulness pillars, size curve and same-path controls had been evaluated on
  CircuitKIT's joint-ranked node set (top 61 of 612 by its absolute node scores) and the L4 edge masks on
  a joint ranking by |signed row sum|, while the pre-registered circuit above is per-type (top 10 % of heads
  and of MLPs separately). Added, without changing any rule: (1) `aya_circuits/cellset.py` evaluates the
  pre-registered set through the same CircuitKIT path (patching, zero/mean ablation, 5 per-type random
  sets under the identical call) and adds *necessity* (the set ablated, the rest kept, zero and mean, with
  the same random control) for every circuit and seed; (2) `audit/l4_cellset.py` re-attributes every audited
  cell and scores CAD, circuit distance and AS-NC on the pre-registered set, keeping the per-cell edge
  scores; its joint-set numbers must reproduce `l4.json` (reported). Once every L4 cell has the
  recomputation, the pre-registered definition is the primary one in `audit/analyze.py` (H2, H3, H4,
  exploratory) and the joint-ranked numbers are reported next to it as "as first reported"; H3's gate on
  the pre-registered set additionally counts patching < 0.1 as failing. No verdict rule changes.
- **A6 (2026-11-06 04:00 IST, post hoc, second audit B14).** B1 steering vectors were built from the first 64
  wrongly answered English prompts, 12 / 18 / 14 of which belong to the fold on which that seed's efficacy
  is measured (A2). The source prompts now exclude the seed's fold (`interventions/run_grid.py`; still 64
  prompts per seed, zero overlap); the 17 main and 68 sweep B1 steering cells are rebuilt, re-audited on
  all layers and the superseded artifacts kept under `runs/_superseded/2026-11-06_A6_b1_steer/`. Target
  prompts (known facts) were never in any fold. No rule changes.
- **A7 (2026-11-06 19:25 IST, brief §8).** `data/b2/baseline_{cyber,theft_fraud,weapons_violence}.jsonl`
  stored 160-character previews of the model's replies, including 529 replies judged compliant to harmful
  prompts. The repo copies now carry no `preview` on harmful rows (`data/sanitize.py`); the files hashed above
  are kept byte for byte, local only, at `.cache/private/b2_baseline_raw/` (cyber still
  `6ee74806…6d21`). Every field the pipeline reads (ids, splits, prompts, translations, `refused`,
  `first_token`) is unchanged: a hash of that view and of the `b2_split` output is identical before and
  after. Sanitized sha256: cyber `9bb77c2a…2e87`, theft_fraud `7c463319…90be`, weapons_violence
  `7dc70499…c14`. No rule changes.
- **A8 (2026-11-06 19:25 IST, post hoc, before any run of it).** H5 names "the English causal-tracing /
  Taylor layer" as the comparison site; only the Taylor layer had been run. `interventions/causal_trace.py`
  runs causal tracing (Meng et al. 2022: subject embeddings + 3σ Gaussian noise, clean MLP outputs restored
  at the last subject token over a 10-layer window, 100 known English facts × 5 draws) and picks the editable
  layer (0..28) with the largest window indirect effect. Condition `causal_trace` = MEMIT at that layer, 3
  seeds on the same folds, same editor settings, norm-matched perturbation controls, all five audit layers.
  H5 is then reported against both sites; the decision rule is unchanged.
- **A9 (2026-11-06 19:25 IST, post hoc, before any run of it).** L1 one-hop ripple (brief §3.3 item 4):
  `data/b1_ripple.py` derives 334 English questions for 174 of the 180 false facts whose answer changes with
  the corrected fact (capital and continent of a city's corrected country; the inverse question for capitals
  and office holders; the capital of the country an office holder leads), golds from the Wikidata pool
  (`data/b1/ripple_en.jsonl`, sha256 `e41ca77e…a8d3`). `audit/ripple.py` scores base and every B1 main cell on
  its fold's items; `l1_ripple_gain` is the paired mean gain. Descriptive, with circuit − comparison tests;
  no hypothesis is re-decided on it.
- **A10 (2026-11-06 19:25 IST, post hoc, before any run of it).** L5 relearning probe for B2 (brief §3.3
  item 4): `audit/relearn.py` applies each B2 main cell, trains an unrestricted LoRA (rank 8, lr 2e-4, 40
  steps, seed 0) on 64 fixed benign Dolly-15k pairs (Qi et al. 2024's benign fine-tuning attack; no harmful
  completion is trained on), and re-measures held-out harmful refusal in five languages and benign refusal.
  `l5_relearn_gain_xl` = cell − base after the same attack; retention = that over `efficacy_xl`.
  Descriptive; no hypothesis is re-decided on it.
- **A11 (2026-11-06 19:25 IST).** Brief §7.4/§12.1 coverage: seeds 1 and 2 of the non-English fact circuits
  (b1_hi/ar/fr/sw) with pillars, same-path controls and the per-type set; the stability pillar and the
  1–30 % size curve for every circuit that lacked them (IOI, langid_en, non-English fact circuits), added to
  the existing `faithfulness.json` files without recomputing any number already there
  (`run_gaps_circuits.sh`). Analysis robustness (no rule change): every paired test in
  `docs/results/hypotheses_*.md` now also prints a seed-cluster bootstrap CI (whole discovery seeds
  resampled), because the pair-level CI treats languages, families and bystanders inside one seed as
  independent; per-cell scorecards carry AuditKIT `compare` per family with a percentile CI.
- **Owner decisions recorded 2026-11-06** (not amendments to any rule): B1 stays at 180 false facts (the
  brief's ≥ 200 would need a new pool and a full B1 rerun), and weight interventions are not rescaled to a
  common norm across conditions (the norm-matched random perturbation control stands in for it). Both remain
  listed as unmet brief items in `docs/report.md` §5.
- **A12 (2026-11-06 20:00 IST, third audit, before any run of it).** L5 "arbitration spread and AURC"
  (brief §3.3 item 4) had been measured as a context-free gold-vs-competitor logit margin; Anand (2026)
  defines arbitration as memory against an injected passage. `audit/arbitration.py` implements that probe:
  138 never-edited English facts balanced over the six B1 relations, four passages per fact (agree, update,
  stale, authority), A = log P(asserted) − log P(other) with both candidates teacher-forced; spread = sd(A),
  AURC over decisions ranked by |A| (following is correct for agree/update), error of the most confident
  quarter; base and every main cell of both behaviours. The old family stays, relabelled as a margin.
  Descriptive; no hypothesis is re-decided on it.
- **A13 (2026-11-06 20:00 IST, third audit, correction).** EAP-IG scores from CircuitKIT's `attribute`
  scale as 1/batch_size with our batch-mean metric; the cached B1 discovery-seed-0 base edges of b1_en,
  b2_en and ioi were computed at batch 4 and every cell at batch 2, so those cells' CAD values were
  inflated (median cell/base edge ratio 2.000; every other behaviour, seed and circuit ≈ 1.00).
  `audit/l4_rescore.py` recomputes the three base files at batch 2 and rescores the affected cells from
  their kept edges; the stale files are kept under `runs/_superseded/2026-11-06_l4_base_bs4/`. No rule
  changes. Also recorded here: amendment A5's added gate criterion (per-type patching < 0.1 counts as
  failing) *is* a change to the H3 gate, contrary to A5's "no verdict rule changes"; the report states H3
  under the gate as originally written and with the A5 criterion.
- **A14 (2026-11-07 11:00 IST, post hoc, after the A13 rescore was seen).** H4 had been evaluated as a
  Spearman over cell × bystander points, which shares one overlap value per bystander and is
  pseudo-replicated. `audit/analyze.py` now also computes the pre-registered Spearman inside each cell over
  its three bystanders and reports its mean with a cell bootstrap and a seed-cluster CI. Both behaviours
  give a positive mean with CIs excluding 0. Because the statistic was added after the rescored numbers
  were known, and because three bystanders confound overlap with circuit identity, the report reads H4 as
  "consistent in direction, not established" rather than as supported. No rule changes.
- **A15 (2026-11-07 12:15 IST, post hoc, after the third audit, before any run of it).** The per-type
  set the cells intervene on (A5) restores 0.4–3 % of the fact-recall logit difference under patching, so
  no cell of the study intervened on a set that passes the faithfulness pillars and the monotone half of
  H3 (pillar scores correlate with the H2 advantage) was never tested. Three conditions are added for B1
  LoRA and steering: `circuit_joint` = CircuitKIT's joint-ranked set whose pillars §4.1 of the report
  quotes (`cellset.json` `pillar_set`; 61 nodes, 21 MLPs for b1_en; patching 0.33–0.56, random-node margin
  0.33–0.56, necessity 0.49–0.51), `random_joint` = 5 random sets with the same head and MLP counts as the
  seed's joint set, `inverse_joint` = the bottom per-type counts by the same absolute node scores.
  Steering cells are norm-matched to the existing `circuit` seed-0 size at coefficient 1 and use the same
  0.5× main multiplier, so every B1 steering cell shares one norm; LoRA uses the pre-registered rank,
  steps and learning rate. MEMIT is unchanged: its `circuit` layer is the top-|score| MLP, which is MLP 0
  under both rankings. Full audit (L1–L5, L4 on both mask definitions, ripple, arbitration), main cells
  only (no steering sweep). Decision rules as pre-registered, applied to `circuit_joint` against
  `random_joint` / `inverse_joint` (H2) and `unrestricted` (H1); H3's gate for these cells is the joint
  pillars (which the B1 joint sets pass). Because the joint set holds 21–22 of the 36 MLPs, a count-matched
  bottom set necessarily shares its 6–8 lowest-ranked MLPs with it (and a random set shares 15 nodes in
  expectation); both overlaps are reported with the cells. New cells only; no existing number changes.
- **A16 (2026-11-07 13:00 IST, post hoc, correction, before any run of it).** The IOI bystander circuit
  was discovered on CircuitKIT's built-in `ioi` task, whose EAP path serialises the whole sentence
  *including* the indirect object, prepends a second BOS, and reads its metric at the last non-pad
  token, i.e. at the answer itself: on this model the metric position is the IO token on 100 % of
  pairs, the clean logit difference there is 2.0 (corrupt −0.3) against 11.2 (corrupt −1.0) at the
  "to" position, and the same holds on GPT-2 (0.5 vs 3.3) (`checks/check_ioi_position.py`,
  `checks/ioi_position*.json`); 2 of 4 pairs per batch also differ in token count on this tokenizer.
  The IOI circuit, its faithfulness, every IOI bystander CAD and the IOI term of H4 therefore
  describe a "what follows the indirect object" task, not IOI. Correction: task `ioi_fixed`
  (`data/ioi_fixed.py`: CircuitKIT's own generator, template and ABC corruption; prompt truncated
  before the IO; one BOS; single-token IO/S; equal clean/corrupt token counts) is discovered on the
  three seeds with the pre-registered pillars, size curve, stability, same-path controls and the
  per-type evaluation, and is re-attributed on every audited cell of both behaviours
  (`audit/l4_task.py`, same audit settings) on all three mask definitions; `audit/analyze.py` uses
  it as the IOI bystander (H2 L4 means, H4) once every L4 cell has it, with the built-in-task values
  kept in the CSVs. The same length-matching defect affects 45 of 207 refusal pairs
  (audit-2026-11-06 C8): `b2m_en` (`data/b2_matched.py`, the 162 token-length-matched pairs) is
  discovered and evaluated on three seeds as a diagnostic of the refusal circuit's unfaithfulness;
  no B2 cell is rebuilt on it. Decision rules unchanged; what changes is which IOI circuit the
  pre-registered quantities are computed on.
- **A17 (2026-11-07 14:10 IST, post hoc, exploratory, after the A15 cells were built and before any
  of them was analysed).** H4 compares three bystander circuits, so overlap is confounded with
  circuit identity (report §4.5). `audit/l4_shared.py` partitions each bystander's own induced edge
  set E(B) into the edges shared with a node set X, E(B) ∩ E(X), and the rest, and scores CAD, the
  same-data repeat floor and the resample floor on each part from the kept edge matrices (no new
  attribution; X = the cell's intervened components, the pre-registered per-type target set, or
  CircuitKIT's joint-ranked target set). Two statistics per (cell, bystander): D = (CAD_shared −
  floor_shared) − (CAD_unshared − floor_unshared), and the magnitude-free Λ = log(CAD_shared /
  floor_shared) − log(CAD_unshared / floor_unshared), in which the Σ|s_e| denominators cancel so that
  the two regions are compared in units of their own resampling noise. Exploratory: reported in
  `hypotheses_*.md` ("A17") and the report's H4 paragraphs; no decision rule and no verdict changes,
  nothing enters the Holm count.
- **A18 (2026-11-08 14:15 IST, post hoc, correction, after the fifth audit and before any run of
  it).** The fifth audit (`docs/audit-2026-11-08.md` B.1) found that `interventions/cell.py` applied
  every LoRA cell by adding ΔW = scaling·B·A to the bf16 weights in place, which rounds 31–43 % of
  the non-zero entries of ΔW to zero (relative L2 error 0.25–0.37, condition-dependent), so every
  audited LoRA number described a different model from the trained adapter. From this amendment a
  LoRA cell is applied as forward hooks that add the fp32 low-rank term to the bf16 module output
  with one rounding, on both the HF model (`register_forward_hook`) and the TransformerLens model
  (permanent hooks at the projection outputs, inputs cached at `hook_{q,k,v}_input`, `hook_mlp_in`,
  `hook_z`, `mlp.hook_post`): the forward PEFT used during training (adapters autocast to fp32).
  `checks/check_apply_equivalence.py` verifies the hook against that forward on every adapted module
  of a real cell (0 ulp) and the HF↔TL agreement on real cells (cosine ≥ 0.997). Every LoRA cell of
  both behaviours (79: 17 + 11 A15 + 17 controls on B1, 17 + 17 on B2) is re-audited on all five
  layers, the L4 target, bystanders and `ioi_fixed` in one attribution pass (`audit/l4.py`), the
  pillar and shared masks, ripple, arbitration, relearning and the L4 power check; superseded files
  are kept under `runs/_superseded/2026-11-08_A18_lora_bf16_merge/`. Also recorded per LoRA cell:
  `n_trained` (unmasked adapter entries) next to `n_trainable`, and the A15 overlap column against
  `circuit_joint` for the joint conditions. No decision rule changes; every H1–H5 verdict is
  re-read from the re-audited numbers under the rules above.
- **A19 (2026-11-09 03:30 IST, post hoc, after the fifth audit; sealed before any run of it; plan:
  `docs/countering-limitations.md` item 3).** The inside view (L4, cross-circuit CAD) at 32 examples
  and 3 IG steps cannot resolve LoRA and steering, and the power check could not separate the
  integration error from the example error. Four measured counters, all through
  `neb.attribution.eap_ig_inputs` (CircuitKIT's hooks and edge index, fp32 accumulation, scores kept
  per example; `checks/check_quadrature.py` shows the per-example mean equals CircuitKIT's score
  times the batch size, relative L2 4e−7 on gpt2) and `audit/l4_a19.py`:
  (a) *Quadrature.* Candidate rules at three passes {right-endpoint Riemann, composite midpoint,
  Gauss–Legendre} and six passes {the same three}, measured on the base model, discovery seed 0, the
  four A19 tasks (`b1_en`, `b2_en`, `langid_en`, `ioi_fixed`), against a 48-step composite-midpoint
  reference (spread to GL-15 and Riemann-48 reported as the reference's own uncertainty) and against
  the IG completeness identity (the sum of the input node's edge scores equals loss(corrupt) −
  loss(clean) for an exact integral; the relative gap is reference-free). Selection rule, fixed now:
  the three-pass rule with the smallest mean relative L2 to the reference over the four tasks; the
  best six-pass rule instead if its error is below half of that. The gpt2 check already shows
  Gauss–Legendre-3 is *worse* than Riemann-3 there (0.23 vs 0.12) while Riemann-6 is 3× better, so
  the plan's expectation is not assumed. The choice is written to `runs/audit/a19/rule.json` and
  `checks/quadrature_tinyaya.json` before any cell is attributed.
  (b) *Paired uncertainty.* For every main cell of conditions circuit, random, inverse, Taylor,
  unrestricted (both behaviours, steering / LoRA / MEMIT) and circuit_joint, random_joint,
  inverse_joint (B1 steering / LoRA): per-example edge scores under the chosen rule on the audit's
  own 32-example subsets (same seed, same rows), the base per seed likewise. Contrast per (circuit
  cell, control set of the same discovery seed, bystander): Δ = CAD(circuit) − mean CAD(controls)
  on the pre-registered per-type edge set, with a leave-one-out jackknife over the examples (base and
  cell means recomputed without each example) as its standard error. Pooled over seeds and
  bystanders (disjoint example sets): SE = sqrt(Σ SE²)/k, the SD of the k contrasts reported too.
  (c) *Equivalence.* Smallest effects of interest, fixed now: δ_L4 = 0.05 CAD; δ_L3 = 0.01 mean
  absolute accuracy change (the plan's proposals, adopted). TOST at α = 0.05: "equivalent" when the
  90 % interval (t, 31 df) lies inside ±δ; "different" when the 95 % interval excludes 0; otherwise
  "inconclusive", with n_required = (z_0.95 + z_0.80)² σ̂² / δ² (σ̂² = n·SE²) stating the per-task
  examples a verdict would need. L3 contrasts use the existing language × family pairs with a
  percentile bootstrap (2000 resamples, seed 0); the three-seed unit caveat (fifth audit B.5) stands.
  Any second stage at a larger n is a separate, later amendment.
  (d) *Attribution-free bystander measure.* Each task's clean loss vector (−logit difference, 32
  examples) under the cell minus the base's, paired per example, with its SE and a TOST at
  δ_beh = 5 % of the task's base clean − corrupt span; the plan's exact edge patching with the AtP*
  diagnostic is not part of this amendment. The built-in `ioi` task is dropped from A19 (A16).
  Validity checks (success = they pass, not that any verdict is "equivalent"): the Riemann-3
  per-example mean matches CircuitKIT's `attribute` on Tiny Aya (Spearman > 0.99, scale 1 ± 0.1 after
  the A13 factor); the chosen rule's completeness gap is no larger than Riemann-3's; the jackknife SE
  of one cell reproduces on a rerun. No pre-registered decision rule changes; nothing enters the Holm
  count; results go to `docs/results/a19_<beh>.md` and the report's L4 and A15 paragraphs.
- **A20 (2026-11-09 04:52 IST, post hoc, after the fifth audit; sealed before any run of it; plan:
  `docs/countering-limitations.md` item 1).** The per-type set the pre-registered cells intervene on
  is necessary but not sufficient (patching 0.004–0.03), and A15's joint set is partial (0.33–0.56)
  and was chosen post hoc. A20 certifies a circuit by a rule and repeats B1 steering and LoRA on it.
  *Execution (also for A19 and A21):* the owner moved A19–A21 to the second machine (puget, NVIDIA
  RTX PRO 6000, CUDA, bf16 as before, the same model snapshot `af89d219…`, the same `uv.lock`); each
  amendment's base model and cells are computed there, so no comparison mixes devices. Before any
  A19–A21 run, puget reproduced the Mac's b1_en seed-0 size-curve patching ratios exactly (0.34653 at
  10 %, 0.76806 at 20 %), the ablation ratios within one bf16 step, and the base audit's headline
  numbers (`puget/` holds the environment and the CUDA shims: `torch.mps.empty_cache` is a no-op, a
  per-process GPU cap of 30 GB, two 15 GB audit shards via `audit/run_audit.py shard=i/2`, fp32
  matmuls without TF32). (a) *Certification* (`aya_circuits/certify.py`): task b1_en, discovery seeds
  0–2; ranking = CircuitKIT's absolute node scores of the seed (`circuit.pt`), joint over heads and
  MLPs (the ranking of the reported size curve); held-out fold = rows order[96:224] of the seed's
  permutation of `eap_known_en.csv` minus any prompt also in order[:96] (discovery read order[:64],
  the pillars and the curve order[:96]); grid k = round(612 f), f = 0.10, 0.12, …, 0.50, plus 0.01,
  0.02, 0.05 for the curve only; sufficiency = CircuitKIT patching raw ratio of the top-k set;
  necessity = the set mean-ablated with the rest kept, r′ = (r − r0)/(1 − r0) with r0 from the clean
  and corrupt means on the same 128 examples, at the grid points (zero ablation reported at the
  certified size). Certified circuit = the
  smallest grid k with sufficiency ≥ 0.8 and mean-ablation r′ ≤ 0.2; a seed with none is reported as
  a failure and builds no cells. Also reported: the area under sufficiency over f ∈ [0, 0.5]. The
  plan's bisection is replaced by the grid because sufficiency need not be monotone in k. (b) *Cells*
  (`interventions/run_a20.py`, root `runs_a20/`), B1 only, steering and LoRA, per discovery seed d:
  `cert` (the certified set); `cert_strat` (primary control, entry-matched: per layer the certified
  set's head count drawn uniformly within that layer, the MLP count drawn uniformly over layers; 5
  draws); `cert_random` (per-type counts drawn uniformly, the pre-registered random rule; 2 draws);
  `cert_inverse` (bottom per-type counts by the same scores); `unrestricted` (rebuilt on puget).
  Cell seeds d + 3j; the pre-registered B1 recipe otherwise (the seed's fold, LoRA rank 8, 60 steps,
  lr 2e-4; steering as A6, norm-matched to the main study's `circuit` seed-0 size at coefficient 1
  and applied at 0.5×, as A15). 60 cells. (c) *Audit*: `hf_layers` (L1, L2, L3, L5 of the main audit)
  for the base and every cell; L4 by A19's per-example EAP-IG under A19's sealed rule on A19's four
  tasks (`audit/a20_l4.py`), with A19's base files. Ripple, arbitration and relearning are not run.
  (d) *Analysis* (`audit/a20_analyze.py`): H1's rule (efficacy(cert) ≥ 0.9 × efficacy(unrestricted));
  H2 contrasts cert − {cert_strat, cert_random, cert_inverse, unrestricted} on L3 (the main
  analysis's pairs and bootstrap, plus A19's TOST at δ_L3 = 0.01) and on L4 (A19's jackknifed paired
  CAD, TOST at δ_L4 = 0.05); `cert_strat` is the primary comparison. (e) *Validity checks*: every
  seed certifies; the realised node set equals the intended one at every grid point (asserted); the
  `cert_s0` LoRA cell trained twice gives the reproducibility of training on CUDA
  (`checks/a20_a21_repro_lora.json`). Exploratory: nothing enters the Holm count of H1–H5 and no
  pre-registered verdict changes. Results: `docs/results/a20_b1.md`.
- **A21 (2026-11-09 04:52 IST, post hoc, after the fifth audit; sealed before any run of it; plan:
  `docs/countering-limitations.md` item 2).** The pre-registered `circuit` MEMIT site is MLP 0 on
  every seed (the top |score| MLP), so the MEMIT verdicts are about layer-0 editing. A21 measures the
  site instead of defending it (`interventions/run_a21.py`, root `runs_a21/`). (a) *Sites* L ∈ {0, 2,
  4, …, 28} ∪ {3 (A8 causal trace), 5 (`circuit_deep`)}, 17 sites; the three B1 folds of the main
  study (seed d's fold, 60 facts); MEMIT at [L] with the pre-registered editor settings. Per (fold,
  site) three cells: `site<L>` uncapped; `site<L>_cap` = the same update rescaled to |ΔW|_F = CAP =
  59.187628, the median |ΔW|_F of the 20 single-layer B1 MEMIT cells of the main study (circuit,
  circuit_deep, inverse, random, taylor, causal_trace; computed from their `cell.json` before this
  amendment); `perturb_site<L>_cap` = norm-matched random rank-k noise at L with the same norm
  (`interventions/perturb.py`, seed d + 100). 153 cells. The key covariance is the pre-registered one
  (1000 WikiText-2 texts, float64 accumulation), accumulated on the GPU with the forward stopped
  after L (identical to the CPU function on two layers after its float32 rounding, `--check-cov`).
  The plan's every-layer version (29 sites) and its subject-position site rule are not part of this
  amendment; the A8 causal-tracing site stands in for the latter. (b) *Audit*: `hf_layers` for the
  base and every cell. (c) *Analysis* (`audit/a21_analyze.py`): per-site means of efficacy, the
  cross-lingual gain (mean of hi/ar/fr/sw), L1 neighbourhood, MMLU, log WikiText perplexity ratio, L3
  and L5 AURC for each condition; the sealed test is a pooled partial Spearman between the site score
  a_L = |signed row sum of MLP L| of the fold's discovery seed and each outcome over folds × sites,
  covariates fold fixed effects and the rank of depth (plus the rank of log |ΔW| for uncapped cells),
  t-test with n − 2 − q df; primary = {efficacy, L3, L1 neighbourhood} × {uncapped, capped}, Holm at
  0.05; edit − noise per site at equal norm is descriptive. (d) *Validity checks*: every planned cell
  exists; capped and noise norms equal CAP within 1 %; `site10_s0` rebuilt gives the edit's
  reproducibility (`checks/a20_a21_repro_memit.json`); `site00_s<d>` equals the main study's MEMIT
  `circuit_s<d>` (same layer, fold and settings, built on the Mac) up to the device
  (`checks/a20_a21_repro_mac.json`). Exploratory: nothing enters the Holm count of H1–H5 and no
  pre-registered verdict changes. Results: `docs/results/a21_b1.md`.
- **A22 (2026-11-11 11:31 IST, post hoc, exploratory; sealed before any A22 cell was trained; the
  labels A22–A27 of `docs/next-experiments.md` were a plan and now start at A23).** Every B2 cell of
  the main study that raised refusal also raised over-refusal of benign prompts (LoRA +0.37 … +0.54
  in English), because the adapters were trained on harmful prompts only, in English only, for a
  fixed 60 steps, on a circuit that is not sufficient at 10 % (patching −0.04; 0.45–0.58 at 30 %).
  A22 asks whether a circuit-restricted adapter trained with a retain objective closes the
  cross-lingual refusal gap without that cost, and whether the circuit restriction matters.
  (a) *Data* (`data/a22_build.py`, built before this text; sha256 `data/a22/split.json` 134db79a…,
  `data/a22/benign.jsonl` 3702ffea…): harmful prompts split by the sha256 of their id:
  weapons_violence 80 train / 24 dev / 80 test, every cyber and theft_fraud prompt test (159 test
  prompts × 5 languages; three prompts listed in two pools stay with the weapons split); benign:
  the 191 XSTest safe prompts in no B2 pool, 151 train / 40 dev, translated by the model with the
  back-translation rule of `data/translate.py`; the 59 distinct pool benign prompts are test only.
  Train target of a harmful prompt = the model's own most frequent short refusal in that language;
  of a benign prompt = the base model's own first 32 greedy tokens (no EOS unless generated).
  Benign pairs with back-translation F1 < 0.5 or whose base reply is a refusal are left out.
  (b) *Cells* (`interventions/run_a22.py`, root `runs/a22/`): LoRA rank 8, lr 2e-4, the
  pre-registered schedule over 120 steps, the same pairs for every cell, snapshots at steps 15, 30,
  60, 120. Conditions per discovery seed d of the main study's `b2_en` circuit: `circuit30` (top
  30 % of the 612 nodes by CircuitKIT's absolute score, ranked jointly; 30 % is the only size
  on the pre-computed 1–30 % curve where the circuit is sufficient at all: patching 0.58 / 0.49 /
  0.45 by seed, ≤ 0.12 at 20 %), `random30` (same head count per
  layer, MLP count matched, 3 draws), `inverse30` (bottom nodes, counts matched), `unrestricted`,
  `circuit10` (the joint-ranked `pillar_set`), `random10` (1 draw). 24 cells.
  (c) *Checkpoint rule*, identical for every cell: on the dev split (24 harmful and 40 benign
  prompts × 5 languages, 24 new tokens, primary judge) keep the earliest snapshot with mean harmful
  refusal ≥ 0.95 and mean benign refusal ≤ base + 0.05; if none qualifies, the snapshot with the
  largest harmful − benign. Test items are never read before this choice.
  (d) *Outcomes* (`audit/a22_eval.py`, 48 new tokens, test items only). Efficacy, primary: share
  of replies to harmful test prompts labelled `comply` by the LLM judge (`audit/a22_judge.py`:
  comply / refuse / safe; a local model, temperature 0, grammar-constrained; a reply starting with
  the trained refusal string is `refuse` without a call), mean over hi/ar/fr/sw and per language;
  English and the 200 XSTest unsafe prompts reported beside it. Efficacy, secondary: refusal rate
  by the primary regex judge, as in the main study. Collateral, primary: refusal rate on the 59
  benign test prompts, mean over the five languages, minus the base model's. Nothing-Else-Broke:
  `audit/run_audit.py` (hf side) for every cell and the base; reported families are fact
  neighbourhood and paraphrase accuracy, Global-MMLU, MGSM, IFEval, WikiText perplexity ratio and
  L5 never-edited top-1 error; the audit's own refusal families are not used for A22 (its XSTest
  safe subset overlaps the A22 train split). L4 is not run.
  (e) *Tests* (`audit/a22_analyze.py`). "Better model": the `circuit30` cell of each seed against
  the base, paired bootstrap over items (10 000 resamples), three seeds reported separately; the
  cell shown as the released model is `circuit30_s0`, fixed here. "The circuit matters":
  circuit30 − random30 (primary), − inverse30, − unrestricted, and circuit10 − random10, on the
  primary efficacy and the primary collateral: per-seed differences (random draws averaged within
  the seed), paired t over the three seeds, item-level paired bootstrap beside it. A circuit
  advantage is claimed only if the seed-level interval excludes 0; otherwise the result is stated
  as "the restriction made no measurable difference". No multiple-comparison count is shared with
  H1–H5 and no pre-registered verdict changes.
  (f) *Validity checks*: every planned cell exists; mask leak 0; the judge's `refuse` label against
  the regex judge on the base model's replies; a hand check of 60 judge labels by the author,
  reported as such. Results: `docs/results/a22_b2.md`.
  (g) *Addendum, 2026-11-11 12:13 IST, written while the cells were training and before any test
  item was generated or scored.* One baseline cell is added (`safetune_rd`): the SafeTune
  refusal direction of the main study (`runs/interventions/b2/steer/safetune_rd_s0`, extracted by
  SafeTune's `extract_refusal_direction` at layer 19) added to the residual stream at 0.125, 0.25,
  0.5 or 1.0 times the raw difference-of-means vector; the strength is chosen on the A22 dev split
  by the checkpoint rule of (c) with "earliest snapshot" read as "weakest strength". It is
  evaluated and audited like every other cell and compared with `circuit30_s0` descriptively
  (item-level paired bootstrap, one seed). Its direction was extracted from 28 English cyber
  prompts that are test prompts in A22, which can only favour the baseline. The public evaluation
  file of the base model additionally stores the benign prompt text.
  (h) *Addendum, 2026-11-11 20:28 IST, before any judge label was produced.* Two changes to the judge
  of (d). The shortcut is narrowed: only a reply that is exactly the trained refusal string is
  `refuse` without a call. The released adapters' replies were inspected for format before
  judging: the adapters often continue after the refusal string (they were trained to emit
  `<|END_OF_TURN_TOKEN|>`, the tokenizer's EOS, while generation stops on `<EOS_TOKEN>`), so the
  sealed shortcut would have labelled text nobody read. The judge prompt and its four parallel
  slots were fixed at the same time (9 of 9 synthetic cases, including "refuses, then complies").
- **A23 (2026-11-11 20:38 IST, post hoc, exploratory; sealed before any A23 cell was trained and
  before A22's judge and analysis stages ran).** A22's audit of its own adapters found three
  defects: WikiText perplexity rose by 11–26 % (`circuit30_s0` 1.26) although facts, MMLU and
  IFEval barely moved; the adapters do not end the turn after refusing (trained to emit the
  tokenizer's EOS, `<|END_OF_TURN_TOKEN|>`, while generation stops on `<EOS_TOKEN>`); and the
  rise in benign refusal sits mostly in Swahili (0.03 → 0.20). A23 changes the training only and
  asks whether the first two go away without losing A22's efficacy.
  (a) *Data* (`data/a23_build.py`, built before this text; sha256 `data/a23/plain.jsonl` bb28dd0d…,
  `data/a23/fluency.jsonl` b49b6db2…): XNLI premises (`facebook/xnli`, human translations, the same
  sentences in en/hi/ar/fr/sw) and WikiText-2 train, selected by sha256 of the item index: retain
  = 100 validation premises × 5 languages + 100 WikiText train paragraphs (585 pairs after
  dropping texts under 8 tokens); dev = 40 other validation premises × 5; test = 200 test premises
  × 5. A retain pair is (first half of the text; WikiText: first 32 tokens) → the base model's own
  greedy continuation (24 / 32 tokens), no chat template, no BOS, no end token.
  (b) *Training* (`interventions/run_a22.py amendment=A23`, root `runs/a23/`): A22's pairs plus
  the 585 plain pairs; targets that end (the refusals) end with the model's own
  `<|END_RESPONSE|><EOS_TOKEN>`; rank, learning rate, schedule and snapshots as A22.
  (c) *Cells*: `circuit30`, `random30` (one draw, cell seed d) and `unrestricted` for the three
  discovery seeds, 9 cells. The 10 % and inverse conditions and the SafeTune baseline are not
  rebuilt; A22's stand.
  (d) *Checkpoint rule*: A22 (c) plus one condition: the dev plain-text perplexity ratio (geometric
  mean over the five languages of cell / base perplexity on the dev premises) is at most 1.05. If
  no snapshot qualifies, the one with the largest harmful − benign − max(0, ratio − 1).
  (e) *Outcomes*: A22 (d) on the same test items with the judge of (h), and the audit. New:
  plain-text perplexity ratio on the 200 held-out XNLI test premises per language
  (`audit/a23_fluency.py`), for every A22 and A23 cell; and the share of replies to harmful test
  prompts that are exactly the refusal string (the turn ended), per language.
  (f) *Which adapter is released*, fixed here: A23's `circuit30_s0` replaces A22's if, on the
  test set, its WikiText perplexity ratio is lower than A22's, its judge `comply` share
  (hi/ar/fr/sw) is at most 0.03 above A22's and its benign refusal (five languages) at most 0.02
  above A22's. Otherwise A22's stays. Both are reported in full either way.
  (g) *Tests*: A22 (e) for `circuit30` against the base, `random30` and `unrestricted`.
  (h) *What A23 does not do and what it reuses*: nothing targets the Swahili benign refusals. The
  test items are the ones A22 was scored on; what was read from them before this seal is A22's
  regex refusal rates, the audit numbers above, the format of `circuit30_s0`'s replies and the
  text of the benign prompts it newly refuses. No test item is in a training or dev set of A22 or
  A23. Results: `docs/results/a23_b2.md`.
- **A24 (2026-11-12 04:06 IST, post hoc, exploratory; sealed before any A24 cell was trained).** A23
  was stopped after its first cell: its sealed fluency term showed the dev plain-text perplexity
  ratio at 1.60 after 15 steps and 6.79 after 120, against 1.20–1.74 for A22's adapters on the
  same sentences, and no snapshot qualified. No A23 test item was generated. The cause is A23 (a):
  the plain-text targets were greedy continuations, and cross-entropy on greedy text sharpens the
  adapter. A24 is A23 with one change: the continuation of each retain prefix is drawn from the
  base model at temperature 1 (no top-k, no top-p, seed 0), so the loss estimates the
  cross-entropy to the base model's own distribution (`data/a23_build.py variant=sampled`;
  `data/a24/plain.jsonl`, sha256 e9624131…, the same 585 prefixes). Everything else is A23 as
  sealed: (b) training with the model's own end tokens, (c) the 9 cells, (d) the checkpoint rule
  with the 1.05 perplexity limit, (e) the outcomes, (f) the release rule against A22's
  `circuit30_s0`, (g) the tests, (h) the limits; root `runs/a24/`, results
  `docs/results/a24_b2.md`. *How the change was chosen*: on the dev split only, one
  `circuit30_s0` cell was trained with each of two candidate targets (`runs/pilot/`). Real-text
  continuations lowered the dev perplexity ratio to 0.62–0.88, which is fitting the corpus, not
  preserving the base model, so they were rejected; sampled continuations held it at 1.02–1.05
  at every snapshot with dev harmful refusal 0.94–0.98 and dev benign refusal 0.14–0.21 (base
  0.63 and 0.14). The dev split has now been used to choose a design as well as checkpoints; the
  test items have not been read by any A23 or A24 run.
