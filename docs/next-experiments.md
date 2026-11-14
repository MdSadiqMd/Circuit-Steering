# Next experiments (plan after A21, 2026-11-09)

Status: **plan only. Nothing here is sealed or run.** Each block becomes a dated amendment (A22
onward) appended to `docs/preregistration.md`, with the file's sha256 recorded before its first run,
as for A19–A21. Where an amendment differs from this text, the amendment rules.

This file is self-contained: it restates what the study found, what a reviewer at a main ML venue
would object to, the evidence already in our own outputs, and the experiments that answer each
objection, with cost, the rules to seal and what each outcome would mean.

## 1. Where the study stands

Question (report §1): does confining steering, LoRA and knowledge editing (MEMIT) to the components an
EAP-IG circuit points to change the target behaviour with less collateral damage than confining them
to size-matched random, inverse, Taylor-selected or unrestricted components, on
`CohereLabs/tiny-aya-global` (≈ 3 B parameters, 36 layers, 612 heads + MLPs)?

Answer so far (report v0.20, `docs/results/puget-a19-a21.md`):

- **Pre-registered study (H1–H5).** Circuit-restricted interventions are not more local than their
  controls on any audit layer; H5 is refuted at MLP 0. The pre-registered per-type circuit restores
  only 0.4–3 % of the fact logit difference, so those cells tested an unfaithful circuit.
- **A19 (inside view).** With a 6-point midpoint rule and jackknifed paired CAD, most LoRA and steering
  L4 contrasts are "equivalent within ±0.05"; MEMIT at MLP 0 moves every bystander circuit by
  0.23–0.37. Several LoRA verdicts do not survive the seed-clustered SE (3 seeds, 2 df).
- **A20 (certified circuit).** A circuit with held-out sufficiency ≥ 0.8 exists on every seed, but it
  holds 14–22 % of the nodes and 29–34 of the 36 MLPs. LoRA on it fixes 99 % of the edited facts, as
  does every control (ceiling); steering at the shared norm reaches 0.02–0.03 efficacy everywhere. On
  side effects it is no more local than layer-matched random head sets.
- **A21 (MEMIT site sweep, 17 layers × 3 folds).** Layer 0, the pre-registered site, is the worst:
  efficacy 0.37 against 0.83–0.88 at layers 4–10, cross-lingual gain 0.03 against 0.28–0.35. Its
  large collateral is mostly edit size (L3 0.185 uncapped, 0.050 at the common norm), and it is the
  only layer where norm-matched noise does real damage. The sealed test (partial Spearman of the
  circuit's per-layer score with each outcome, given depth and size) rejects none of its six primary
  tests; outside them the score goes with *worse* cross-lingual transfer (ρ −0.38 / −0.39).

## 2. What a main-track reviewer would object to

1. **Novelty.** Hase et al. (NeurIPS 2023) showed that where causal tracing localizes a fact does not
   predict where editing it works; A21 is close to a multilingual rerun of that design with circuit
   scores. The report has no related-work section.
2. **One model.** Every number is on Tiny Aya; plan item 4 (siblings, other families, more methods
   and seeds) has not run.
3. **The circuit is not sparse.** With 29–34 of 36 MLPs inside every certified set, count-matched
   controls share almost all MLPs, so A20's contrast is mostly about which heads are trained.
4. **Two of three methods cannot discriminate on B1.** LoRA is at ceiling (0.99 in every condition),
   steering at the floor (0.02–0.03). Only MEMIT is informative there.
5. **Fragile pieces.** Seed-level uncertainty (3 seeds), a sealed A21 test with no rejection, ±1 item
   in 60 of audit-order noise in the main study and A20 (fixed only from A21 on), and MEMIT edits at
   layer 0 that differ by 1.7–15 % between the Mac and puget.

## 3. Evidence already in our outputs

Measured from `runs/a20/audit/b1/*/hf.json` and `docs/results/hypotheses_b2.md` on 2026-11-09; none of
it is a sealed analysis, so any use in the paper is exploratory until an amendment seals it.

- **LoRA is at ceiling only in English and the Latin-script languages.** Edited-fact accuracy after
  LoRA, mean over the 3 seeds, every A20 condition: en 1.00, fr 0.95–0.96, sw 0.82–0.83, but
  hi 0.18–0.21 and ar 0.22–0.24 (base model: en 0.006, fr 0.12, sw 0.04, hi 0.06, ar 0.11). Hindi and
  Arabic efficacy is far from the ceiling, so it can separate conditions.
- **French and Swahili "transfer" is probably string copying.** The edited answers are mostly names
  (137 of 180 B1 facts are office holders), spelled identically in Latin-script languages; Hindi and
  Arabic need a different script. Measuring the share of targets whose fr/sw label equals the English
  one is a data-only check (no model) and should come first.
- **Steering works on refusal, not on facts.** B2 cross-lingual refusal gain: circuit 0.215, random
  0.104, inverse 0.049, Taylor 0.125, unrestricted 0.312, SafeTune refusal direction 0.340 (English is
  capped at +0.083 because the base model refuses 0.917). On B1 one difference-of-means "recall
  confidence" direction cannot write 60 different answers; efficacy stays at 0.02–0.03 in every
  condition, a mismatch of method and task rather than a finding about circuits. This matches AxBench
  (steering trails prompting and fine-tuning) and Tan et al. (steering is unreliable across inputs).

## 4. Experiments

Ordered by value per hour. Costs are from measured stage times: puget (RTX PRO 6000) ran A20 (3
certifications, 60 cells, audits) in 3.6 h and A21 (153 MEMIT cells, audits) in 4.6 h; on the Mac an
audited LoRA cell takes about 13 min and an HF-side audit about 7 min per cell.

### A22. Remove the fragile pieces (cheap, do first)

- **Edited-fact re-audit.** Re-score only the edited-fact families (en and hi/ar/fr/sw, plain and
  paraphrase) of every main-study and A20 cell with the order fix (`sorted(edited)` in
  `audit/run_audit.py`). A20 showed every other family identical across processes, so this removes
  the ±1-in-60 caveat completely.
  - Code: a `families=` filter in `audit/run_audit.py` (not present today); results written beside
    the existing `hf.json` and merged by `audit/analyze.py`.
  - Cost: about 3–5 h on puget, 1–2 days on the Mac.
  - Rule to seal: the new numbers replace the old ones in every table; any verdict that changes is
    reported as changed.
- **Frozen MEMIT prefixes.** MEMIT's context prefixes are sampled from the model
  (`interventions/edit.py`, `n_prefixes`, `prefix_seed`), so they differ across devices. Generate them
  once, save them under `data/`, load them everywhere, and rerun the layer-0 Mac-vs-CUDA check
  (`checks/check_a20_a21_repro.py mac`). Expected: the 1.7–15 % differences fall to float rounding.
  Cost: under 1 h.
- **Seed-clustered SE as the primary test** for every new paired contrast (A19 showed the jackknife SE
  covers example sampling only, and the cell-to-cell spread is about 10× larger).

### A23. LoRA at matched efficacy

- **Why.** At rank 8 / 60 steps any parameter subset memorises 60 facts (Aghajanyan et al.: fine-tuning
  fits in small random subspaces), so H1 cannot separate conditions and side effects are compared at
  an over-trained point. Lee et al. (EMNLP 2025) found that even training the region known to hold the
  knowledge does not necessarily improve the forget/retain trade-off; the right comparison is a trade-off curve.
- **Design.**
  - Save adapter checkpoints at steps 1, 2, 4, 8, 16, 32, 60 for every A20 condition (`cert`,
    `cert_strat` × 5, `cert_random` × 2, `cert_inverse`, `unrestricted`) and seed.
  - Light audit at every checkpoint: edited facts en/hi/ar, neighbourhood, a fixed L3 subset.
  - Full audit at the two checkpoints closest to the matched efficacy levels.
  - Report side effects against efficacy per condition, side effects at efficacy 0.5 and 0.8
    (linear interpolation between checkpoints), and steps to reach each level.
- **Rules to seal.** Primary efficacy = mean of hi and ar edited-fact accuracy (not at ceiling, not
  confounded by shared spelling); the two efficacy levels; the interpolation; TOST on L3 at
  δ = 0.01 with the seed-clustered SE.
- **Code.** A `save_steps` option in `interventions/lora.py`; the light audit as a config override of
  `audit.l1` / `audit.l3` sizes (the dry run in `scripts/puget/dry_run.sh` already uses such sizes).
- **Cost.** About 6–8 h on puget.
- **Outcomes.** If the certified circuit reaches matched efficacy with less collateral than its
  controls, that is the first positive locality result of the study; if not, "no locality advantage
  at any efficacy level" replaces the ceiling caveat.

### A24. Steering where it fits

- Retire B1 steering from the hypothesis tests (kept as a reported negative: one direction cannot
  write specific facts) and keep steering on B2, where cross-lingual efficacy is not at ceiling.
- Optional new behaviour B3, output-language steering: a direction by construction, multilingual, and
  the language-ID circuit (`langid_en`) already exists with three seeds.
- Rule to seal: steering conditions compared at matched cross-lingual efficacy using the existing
  coefficient sweep (`interventions.steering.coefficients` in `configs/base.yaml`).
- Cost: mostly writing for B1/B2; B3 a few hours of cells and audits.

### A25. Sparse circuits

- **Why.** Node-level scores summed over token positions pull in almost every MLP: fact recall enriches
  the subject in many early and middle MLPs at the subject tokens (Geva et al. 2023; Fierro et al.
  2025 find this step language-independent). Haklay et al. (ACL 2025) show position-blind scores let
  contributions cancel and that position-aware circuits trade size for faithfulness better.
- **(a) MLP-neuron circuits.** Score every MLP neuron with the same EAP-IG pass (per-neuron attribution
  at the MLP's hidden activation), certify by the A20 rule with neurons as units, and restrict LoRA to
  the gate/up rows and down columns of those neurons. `interventions/lora.py` already masks rows for
  heads; neurons need the same mask on MLP modules. Controls: count-matched random neurons in the same
  layers, so circuit and control share layers but not units. Neuron-level editing improved locality in
  FiNE, so a positive result is possible here.
- **(b) Position-aware circuits.** Score each component separately at the subject span, the relation
  span and the final token (the per-example attribution in `neb/attribution.py` keeps the position
  axis if it is not summed). Intervene only there: steering adds its vector at those positions; LoRA
  is replaced by a ReFT-style intervention at the chosen layers and positions (pyreft, a new
  dependency); MEMIT already writes at the subject's last token.
- **(c) Learned mask as a lower bound.** Optimise a hard-concrete node mask for sufficiency under a
  size budget, initialised from EAP-IG scores (the warm start that beat EAP-IG in the MIB shared
  task), with a recorded seed. If even an optimised mask needs about 30 MLPs, non-sparsity is a
  measured property of the model, not of the ranking.
- **Rules to seal.** Units, the certification rule (A20's, with the unit changed), size grid,
  controls, and that (a) and (b) are reported whether or not they certify.
- **Cost.** About 8–12 h on puget for (a) and (b) with LoRA cells and audits; (c) about 3–4 h.

### A26. Lookup-based versus output-traced localization (the headline)

- **Why.** Recent work disagrees on whether localization helps: Guo et al. (ICML 2025), LoFiT, CT-SFT
  (2026) and SCAN (2026) say yes; Hase et al. (2023), Lee et al. (2025) and this study say no. Guo et
  al. reconcile them: localizing to the *fact-lookup* mechanism (MLPs that enrich the subject) makes
  edits robust and local, localizing to components that *write the output* (causal tracing,
  attribution patching) does not. Our EAP-IG circuit is output-traced (scored at the final position,
  which is why it ranks MLP 0 first), and A21 found its per-layer score goes with *worse*
  cross-lingual transfer. Chen et al. (NeurIPS 2026) argue static localization only looks useful on
  MLP-dominated tasks; fact recall is MLP-dominated and still shows no benefit here.
- **Design.** One grid, same facts and folds:
  - localizations: output-traced EAP-IG (as now), subject-position scores from A25(b), causal tracing
    at the subject's last token (A8's site rule), random, unrestricted;
  - interventions: MEMIT (site = the localization's top MLP layer) and LoRA (restricted to the
    localization's set, at matched efficacy as in A23);
  - outcomes: hi/ar efficacy, cross-lingual gain, L1 neighbourhood, L3, MMLU.
- **Rules to seal.** Primary contrast: subject-position minus output-traced on cross-lingual gain and on
  L3, seed-clustered TOST; secondary: each against random. Because the hypothesis came from A21's
  data, it is tested on new folds (A27) and at least one other model, never only on the A21 folds.
- **Outcomes.** Either is publishable: "localization helps only when it targets the lookup mechanism",
  or "it does not help even then".
- **Cost.** About 10 h on puget on top of A23 and A25.

### A27. Generality and statistical power

- **Folds.** 10 disjoint folds of the 180 B1 facts by hashed fact ID instead of 3 seeds (exchangeable
  and reproducible); seed-clustered tests then have 9 df instead of 2.
- **Random reference sets.** 50 layer-stratified random sets per LoRA condition, 100 per steering
  condition; exact Monte Carlo p = (1 + #{random at least as good}) / (N + 1).
- **A21 at full width.** All 29 editable layers × 10 folds: the partial-ρ interval narrows from about
  ±0.27 to about ±0.12, and the negative score-transfer relation becomes a sealed confirmatory test.
- **Models.** The Tiny Aya regional siblings (Earth, Fire, Water: same architecture and tokenizer
  if the merge description holds, so the 612-node map is shared; check the Hub repository names and
  licences first) and one other family that fits the 30 GB budget. Gemma-2-2B is the practical
  choice: public Gemma Scope transcoders would allow a feature-level circuit arm (as in SCAN) without
  training any.
- **Cost.** About 60–70 h (plan item 4) plus about a day for the full-width A21.

### Optional: the parts of item 2 that did not run

- The subject-position site rule for MEMIT (covered by A26) and AlphaEdit (edits projected into the
  null space of preserved keys; tests whether the site still matters once collateral is held down by
  construction). About 1 day.

## 5. Order and budget

| step | amendment | closes | puget time |
|---|---|---|---|
| 0 | none (writing) | related work; B1 steering retired; hi/ar efficacy as exploratory; spelling confound measured | none |
| 1 | A22 | audit-order noise, cross-device MEMIT, primary SE | about 5 h |
| 2 | A23, A24, A25 | ceiling, steering scope, non-sparsity | about 15–20 h |
| 3 | A26 | novelty: lookup vs output localization | about 10 h |
| 4 | A27 | seeds, power, other models | about 3–4 days |

Every block ends as the earlier ones did: tables and analysis, `just check`, a report version bump,
an entry in `docs/decisions.md`, and nothing published or committed without an explicit instruction.

## 6. What stays true even after all of it

- One architecture class at about 3 B parameters under the 30 GB budget; the other family tests "is
  this Aya-specific?", not scale.
- The pre-registered H1–H5 verdicts stay as they are; everything above is labelled post hoc with its
  seal time and hash.
- A best-paper outcome is very unlikely at this scope. Steps 0–3 make a credible main-track
  submission (stronger if A26 comes out positive); step 4 makes it competitive. A workshop paper or
  TMLR fits the current state.

## Sources

- Hase et al., Does Localization Inform Editing?, NeurIPS 2023. https://arxiv.org/abs/2301.04213
- Guo et al., Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic
  Localization, ICML 2025. https://arxiv.org/abs/2410.12949
- Lee et al., Does Localization Inform Unlearning?, EMNLP 2025. https://arxiv.org/abs/2505.16252
- Chen et al., Navigating by Old Maps: The Pitfalls of Static Mechanistic Localization in LLM
  Post-Training, NeurIPS 2026. https://arxiv.org/abs/2605.06076
- Yin et al., LoFiT: Localized Fine-tuning on LLM Representations, 2024. https://arxiv.org/abs/2406.01563
- Wu et al., ReFT: Representation Finetuning for Language Models, NeurIPS 2024.
  https://arxiv.org/abs/2404.03592
- Wu et al., AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders, ICML 2025.
  https://arxiv.org/abs/2501.17148
- Tan et al., Analysing the Generalisation and Reliability of Steering Vectors, 2024.
  https://arxiv.org/abs/2407.12404
- Haklay et al., Position-aware Automatic Circuit Discovery, ACL 2025. https://arxiv.org/abs/2502.04577
- Mueller et al., MIB: A Mechanistic Interpretability Benchmark, ICML 2025.
  https://arxiv.org/abs/2504.13151
- Mondorf et al., BlackboxNLP 2025 MIB shared task: ensemble strategies for circuit localization.
  https://aclanthology.org/2025.blackboxnlp-1.31.pdf
- Geva et al., Dissecting Recall of Factual Associations, EMNLP 2023.
  https://aclanthology.org/2023.emnlp-main.751
- Fierro et al., How Do Multilingual Language Models Remember Facts?, ACL Findings 2025.
  https://arxiv.org/abs/2410.14387
- Lu et al., English-centric factual recall followed by translation in multilingual LMs, EMNLP 2025.
  https://arxiv.org/abs/2505.20546
- Circuit-targeted supervised fine-tuning (CT-SFT), 2026. https://arxiv.org/abs/2601.08146
- SCAN: sparse transcoder-circuit knowledge editing, 2026. https://arxiv.org/abs/2603.15226
- FiNE: Precise Localization of Memories, neuron-level knowledge editing.
  https://arxiv.org/abs/2503.01090
- Aghajanyan et al., Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning,
  ACL 2021. https://arxiv.org/abs/2012.13255
- Fang et al., AlphaEdit: Null-Space Constrained Knowledge Editing, ICLR 2025.
  https://arxiv.org/abs/2410.02355
- NeurIPS 2025 best paper awards (for the bar). https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/
