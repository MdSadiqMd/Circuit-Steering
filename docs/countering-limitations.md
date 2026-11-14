# Countering the four open limitations (plan, 2026-11-08)

Status (2026-11-09 16:10 IST): **items 3, 1 and 2 ran as amendments A19, A20 and A21 on the puget
machine; results in `docs/results/puget-a19-a21.md` and report §4.6. Item 4 has not run; the plan
after A21, including item 4, is `docs/next-experiments.md`.** The text
below is the plan as written on 2026-11-08 (where the sealed amendment differs, the amendment rules).
Each block below becomes a dated amendment (A19 onward) in
`docs/preregistration.md`, with the amendment's sha256 recorded before its first run. The A18 LoRA re-audit
(`run_a18.sh`, started 2026-11-08 14:13 IST, about 13 h) must finish first. All compute estimates are
from the measured stage times in `docs/technical-report.md` §5 and are approximate.

The four limitations, as stated at the end of the explainer film and in report §5:

1. The pre-registered circuit was not sufficient.
2. The MEMIT edit site was layer 0 by construction.
3. The inside view (L4, cross-circuit CAD) is under-powered for small changes.
4. One model, one discovery method, three seeds.

## Verdict

- Items 1, 2 and 3 can be countered on this machine, in the sense that the reviewer's objection stops
  applying.
- Item 4 can be reduced (sibling models, one other family, several methods, many folds and random
  sets), but "one architecture class at about 3B parameters" remains the honest scope under the 30 GB cap.
- Nothing changes the pre-registered results themselves. They stay reported as they are. Every fix
  below is post hoc and must be sealed (written, dated, hashed) before it runs, so that its result
  cannot shape its rule.

## What "deterministic" means here

Two separate properties, both required:

- **Nothing left to choose after the seal.** Every threshold, set definition, site list, effect size of
  interest and test statistic is fixed in the amendment text before the run. Results can only fill in
  numbers, never choose rules.
- **Reproducible numbers.**
  - Wherever possible: enumerated designs instead of random draws (folds by hashed fact ID, sites by
    layer index, fixed rankings).
  - Closed-form or exact procedures (MEMIT solve, exact activation patching, quadrature, jackknife, exact
    Monte Carlo p-values).
  - Any remaining random draw uses a recorded seed.
  - EAP-IG and steering already reproduce bit-exactly on MPS.
  - For LoRA training, either run the certification subset on CPU in fp32 or rerun a sample of cells
    twice and report the maximum difference.

## Item 1: the pre-registered circuit was not sufficient

**The objection.** The per-type set (58 heads + 4 MLPs) restores 0.02–0.03 of the logit difference.
Ablating it removes the behaviour, so it is necessary but not sufficient. The pre-registered H1–H3 cells
therefore tested an unfaithful circuit. A15 used the joint-ranked set (0.33–0.56), but that is still
partial and was chosen post hoc.

**Counter: a certified circuit defined by a sealed rule instead of a fixed size.**

- **Certification rule (to seal).** The circuit is the smallest k, in a fixed ranking order, such that
  exact activation patching on held-out examples gives:
  - sufficiency ≥ τ = 0.8, and
  - necessity read as r′ = m/cl ≤ 0.2 when the set is ablated.

  k is found by bisection over the ranking. Rankings are computed on the discovery fold and
  certification uses a disjoint fold.
- **Expected size.** The existing curve gives 0.77 at 20 % and 0.97 at 30 % for b1_en seed 0, so the
  certified circuit will be about 20–30 % of the 612 nodes. That is itself a result: fact recall in
  Tiny Aya is not sparse at node level.
- **Report the whole curve, not one point.**
  - Use MIB's CPR (area under the faithfulness-vs-size curve) and CMD (area between the curve and 1).
  - This removes the "why 10 %?" question.
- **Smaller sufficient circuits, two sealed alternatives:**
  - position-aware attribution (separate scores at the subject token and the final token);
  - a learned node mask optimised for faithfulness under a size budget (Edge Pruning style, hard-concrete
    with a recorded seed, CPU fp32).
- **Interventions on the certified set.**
  - LoRA and steering against count-matched random sets (5 draws, plus the reference sets of item 4) and
    the inverse set.
  - Matched on entries actually trained, not node count. This also closes the capacity confound from the
    fifth audit (A6, A9).
- **Cost.** About 9–10 h:
  - certification: about 4.5 h;
  - 21 LoRA cells: about 13 min each with audit and L4;
  - steering: minutes.

## Item 2: the edit site was layer 0 by construction

**The objection.** A logit-difference attribution scored at the final position always ranks MLP 0 first.
The MEMIT verdicts on H1, H2 and H5 are therefore about layer-0 editing, and layer 0 is known to be
fragile.

**Counter: measure every site instead of defending one.** This mirrors the design of Hase et al.
(NeurIPS 2023).

- **Site sweep.** MEMIT at every editable MLP layer (0–28; the last layers' edit gradient is zero), with
  the same fixed fact sets, three folds, in two conditions:
  - **uncapped**, as the method produces it;
  - **norm-capped**, every site's |Δθ| rescaled to the same value (sealed: the median single-layer
    |Δθ| of the existing cells, about 50).
- **Norm-matched noise at every site.** This separates "where" from "how big". It is the test the fifth
  audit (B10) said was missing.
- **The test (to seal).** Across sites, does the circuit's attribution score at that layer predict
  efficacy and collateral (L1, L3, L5) after controlling for depth and |Δθ|?
  - Partial Spearman, n ≈ 29 sites per fold.
  - MEMIT is a closed-form least-squares solve, so in fp32 on CPU it is deterministic.
- **A principled site rule (to seal).** Score MLPs at the subject's last token, where MEMIT writes,
  instead of the final position. This follows Geva et al. 2023 (subject enrichment happens in early-to-mid
  MLPs at the subject token) and Lad et al. 2024 (the first layers detokenize and are the most fragile).
- **Optional, AlphaEdit.** Project each edit into the null space of the preserved-knowledge keys
  (closed form). This tests whether the site still matters once collateral is held in check by
  construction.
- **Cost.**
  - Every second layer, capped condition with its noise control: about 27 h.
  - All layers, both conditions: about 55 h.

## Item 3: the inside view is under-powered for small changes

**The objection.** At 32 examples and 3 IG steps, LoRA and steering CADs sit below the resample floor, so
H2 and H3 on L4 cannot see them. The power check changed examples and steps together and could not
say which one drives the floor.

**Counter, part (a): remove the integration error.** This is a deterministic error, and the fix is new as
far as I could find.

- CircuitKIT's EAP-IG uses a right-endpoint Riemann sum, `α = k/steps` for k = 1…m
  (`circuitkit/backends/eap/attribute.py:264`). That is a first-order rule. At 3 steps it evaluates
  only α ∈ {1/3, 2/3, 1}.
- Replace it, at the same number of passes, with either:
  - 3-node Gauss–Legendre quadrature (exact for polynomial paths up to degree 5), or
  - Richardson extrapolation, R ≈ 2·S₂ₘ − Sₘ.
- Validate against a 32-step reference on the same examples.
- This separates step error from example error, which the power check could not do.

**Counter, part (b): use the right yardstick.**

- The resample floor measures how the baseline attribution varies when the examples change. H2 compares
  two conditions on the same examples.
- The relevant uncertainty is that of the paired contrast CAD(circuit) − CAD(random), with common random
  numbers (identical inputs).
- Estimate it with a leave-one-out jackknife over the 32–64 examples. This uses no random number
  generator.
- It needs per-example edge scores. Small patch to the attribution loop: accumulate per item instead of
  only the mean, keeping the 1/batch scaling identical.

**Counter, part (c): turn nulls into answers.**

- Seal a smallest effect size of interest δ per layer. Proposed (owner to fix before sealing):
  - δ_L4 = 0.05 CAD;
  - δ_L3 = 0.01 mean absolute accuracy change, the size of the A15 positive result.
- Run an equivalence test (TOST). Each H2 comparison then reads "equivalent within ±δ", "different", or
  "inconclusive".
- The number of examples is fixed in advance from the jackknife variance:
  n = (z₁₋α + z₁₋β)² σ² / δ².

**Counter, part (d): a check that uses no attribution.**

- Exact activation patching on each bystander's top edges, with AtP*'s false-negative diagnostic.
- Bystander-task behaviour (logit difference) as a direct measure.

**Cost.** About 5–6 h:

- quadrature validation: about 2 h of base attributions;
- per-example re-attribution of the H2 cells: about 3 h.

## Item 4: one model, one discovery method, three seeds

**The objection.**
- Three seeds are three data points. The A15 positive result's paired t over seeds gives p ≈ 0.13.
- Circuits are not unique (Méloux et al. 2025), so one method's circuit may not be "the" circuit.
- Results on one model may not transfer.

**Counter for seeds: change the statistical unit.**

- Replace seeded discovery subsets with K = 10 disjoint folds, enumerated by hashing fact IDs. Folds are
  reproducible and exchangeable.
- Compare each circuit with a large count-matched random reference set:
  - 100 sets for steering (6 s per cell);
  - 50 sets for LoRA (about 10–13 min per cell with audit);
  - stratified by layer so every set has the circuit's per-layer counts.
- Exact Monte Carlo p-value, with folds as blocks:
  p = (1 + #{random sets at least as good}) / (N + 1).
  - This is valid at any N.
  - It never reports p = 0.
- Where averages are needed, report the interquartile mean (IQM) with stratified bootstrap
  (Agarwal et al. 2021).

**Counter for the method: show the conclusion holds across circuits, not that the circuit is unique.**

- Discover with:
  - exact node activation patching (the definition itself, deterministic, about 50 min per fold per task);
  - EAP-IG with the quadrature fix;
  - EAP-GP;
  - CDT (contextual decomposition).
- ACDC and aggressive ibcircuit stay off on Tiny Aya.
- Combine them with stability selection across folds × methods. Keep a node if it is selected with
  frequency ≥ π. The expected number of false selections is bounded by E(V) ≤ q² / ((2π − 1) p)
  (Meinshausen and Bühlmann 2010; check the exact form against the paper before quoting it).
- Certify each method's circuit by the item 1 rule, then run the interventions on every certified
  circuit. The claim becomes "the conclusion holds for every sufficient circuit we found". This is the
  manipulability standard that Méloux et al. propose in place of uniqueness.

**Counter for the model: use the siblings, then one other family.**

- Tiny Aya Earth, Fire and Water are region fine-tunes merged with Global (Tiny Aya paper, 2026).
  - If they share the architecture and tokenizer, as the merge implies, the 612-node map is identical.
  - That gives a replication with no code change and directly comparable circuit overlap.
  - Check the Hub repository names and licences before planning on them.
- One other family that fits the 30 GB cap and CircuitKIT's production list (Qwen-2.5 3B or Gemma-2-2B)
  answers "is this Aya-specific?".
- Reduced B1 grid per model: certified circuit, joint set, random, inverse, for LoRA, plus MEMIT at the
  sealed site, three folds.

**Cost.**
- Three sibling models: about 25 h.
- LoRA random reference sets: about 30 h.
- Methods: about 3 h.
- The other family: about 10 h more.

## Order and budget

Queued after A18 finishes, ranked by value per hour:

| block | amendment | what it closes | compute |
|---|---|---|---|
| item 3 | A19 | under-powered L4; nulls without a margin | about 6 h |
| item 1 | A20 | insufficient circuit; capacity confound | about 10 h |
| item 2 | A21 | layer 0 by construction; size versus site | about 27 h (55 h full) |
| item 4 | A22 | seeds, methods, model | about 60–70 h |

Total: about 100–130 h of machine time. Each block is independent and resumable. Each ends with tables,
analysis, `make check`, and a report version bump.

## Success criteria (sealed with each amendment)

"Success" means the method ran as sealed and its validity checks passed. It does **not** mean the
hypotheses came out positive.

- **Item 1.** A certified circuit exists for every fold: sufficiency ≥ 0.8 and necessity ≤ 0.2 on
  held-out examples. Every intervention cell on it has count-matched and entry-matched controls.
- **Item 2.** Every planned site has an uncapped cell, a capped cell and a noise cell. The capped cells'
  |Δθ| are equal within 1 %. The partial correlation is computed on all sites.
- **Item 3.** The validity checks pass:
  - quadrature attribution is within a sealed tolerance of the 32-step reference, and closer to it than
    the 3-step Riemann rule;
  - the jackknife SE is reproducible to the last digit on a rerun;
  - every H2 L4 and L3 comparison has a TOST verdict.
- **Item 4.** At least 3 discovery methods give a certified circuit. At least 2 sibling models complete
  the reduced grid. Every circuit comparison has an exact p-value against at least 50 random sets.

## If everything succeeds: what it looks like

### The closing "perfect?" slide, rewritten

| before | after (if every block succeeds) |
|---|---|
| the pre-registered circuit was not sufficient | the conclusions are tested on certified circuits (sufficiency ≥ 0.8, necessity ≤ 0.2) from several methods, and the pre-registered result is reported beside them |
| the edit site was layer 0 by construction | every editable layer was edited at equal \|Δθ\| with matched noise; circuit score explains a stated share of collateral after depth and size |
| the inside view is under-powered for small changes | integration error removed; paired contrasts with jackknife SEs; every L4 and L3 test is "equivalent within δ", "different" or "inconclusive" |
| one model, one discovery method, three seeds | 10 folds, 4 methods, 50–100 random reference sets per circuit with exact p-values, 3 sibling models plus 1 other family |

### What the headline could read like

The answer to "did the circuit buy anything?" is not decided in advance. Each outcome is a clean result
once the objections are closed.

- **If the certified circuit beats its random reference set on L3 or L4, across methods and models:**
  - "Confining LoRA to a sufficient circuit reduces cross-lingual collateral by X [CI] relative to
    count- and entry-matched random sets (exact p = …, 10 folds, replicated on 3 sibling models)."
  - That would turn the A15 hint into a finding.
- **If the comparisons are equivalent within δ:**
  - "On Tiny Aya, confining an intervention to a sufficient circuit buys no locality beyond ±δ compared
    with random sets of the same trained size; collateral is set by site and size."
  - This is a positive statement of absence, not a failure to detect. It answers the field's open
    dispute (Chen et al. 2026; CircuitKIT E7) with the strongest controls so far.
- **If results split by method or model:**
  - "Whether a circuit helps depends on how it was found."
  - This is still a reportable answer, given the identifiability results.
- **For MEMIT:** a site–response curve. Efficacy and collateral are plotted against layer at equal
  |Δθ|, with the circuit's per-layer score overlaid. The layer-0 objection is replaced by a measured
  relation.

### What remains even then

- One architecture class at about 3 B parameters (the 30 GB cap). The other family only tests "is this
  Aya-specific?".
- Node-level circuits for the interventions, unless the position-aware or learned-mask variant becomes
  the certified one.
- The pre-registered H1–H5 verdicts stay as they were. The new results are labelled as post-hoc
  amendments with their seal timestamps and hashes.
- The new methods (Gauss–Legendre EAP-IG, jackknifed paired CAD, stability-selected circuits) each rest
  on their own validation check and are reported as such.

## Sources

- Mueller et al., MIB: A Mechanistic Interpretability Benchmark, ICML 2025. https://arxiv.org/pdf/2504.13151
- Bhaskar et al., Finding Transformer Circuits with Edge Pruning, NeurIPS 2024. https://arxiv.org/abs/2406.16778
- Haklay et al., Position-aware Automatic Circuit Discovery, ACL 2025. https://aclanthology.org/2025.acl-long.141
- Shi et al., Hypothesis Testing the Circuit Hypothesis in LLMs, NeurIPS 2024. https://arxiv.org/abs/2410.13032
- Kramár et al., AtP*: localizing LLM behaviour to components, 2024. https://arxiv.org/abs/2403.00745
- Hase et al., Does Localization Inform Editing?, NeurIPS 2023. https://arxiv.org/abs/2301.04213
- Geva et al., Dissecting Recall of Factual Associations, EMNLP 2023. https://aclanthology.org/2023.emnlp-main.751
- Lad, Gurnee, Tegmark, The Remarkable Robustness of LLMs: Stages of Inference?, 2024. https://arxiv.org/abs/2406.19384
- Fang et al., AlphaEdit: Null-Space Constrained Knowledge Editing, ICLR 2025. https://arxiv.org/abs/2410.02355
- Riemann Sum Optimization for Accurate Integrated Gradients Computation, NeurIPS 2024 workshop. https://arxiv.org/abs/2410.04118
- Lakens, Equivalence Tests: A Practical Primer. https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906
- Agarwal et al., Deep RL at the Edge of the Statistical Precipice, NeurIPS 2021. https://arxiv.org/abs/2108.13264
- Meinshausen and Bühlmann, Stability Selection, JRSS-B 2010. https://arxiv.org/abs/0809.2932
- Méloux et al., Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?, 2025. https://arxiv.org/abs/2502.20914
- Chen et al., Navigating by Old Maps, 2026. https://www.alphaxiv.org/abs/2605.06076
- van der Wal et al., PolyPythias, ICLR 2025. https://arxiv.org/abs/2503.09543
- Tiny Aya technical report, 2026. https://arxiv.org/abs/2603.11510
