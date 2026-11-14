# A19, A20, A21 on puget (2026-11-09): what ran, what was checked, what was found

Plan: `docs/countering-limitations.md` items 3, 1, 2. Rules: `docs/preregistration.md` A19 (sealed 03:30 IST),
A20 and A21 (sealed 04:52 IST, file sha256 `24fcdc95…` before either ran). All three ran on the second
machine (puget: NVIDIA RTX PRO 6000 Blackwell, CUDA 13, torch 2.15, the same `uv.lock` and model snapshot
`af89d219…`), one after another, entirely on its RAM disk `/dev/shm/cs`; every output was copied back,
verified byte for byte (`rsync -c`: 0 differences) and the RAM disk was deleted. Each amendment's base model
and cells were computed on puget, so no comparison mixes devices.

Detailed tables: `a19_b1.md`, `a19_b2.md` (+ `.csv`), `a20_b1.md` (+ `.csv`, `_l4.csv`), `a21_b1.md`
(+ `.csv`). Raw outputs: `runs/audit/a19/`, `runs/audit/<beh>/<cell>/a19.*`, `runs/a20/`, `runs/a21/`,
`runs/a20_repro/`, `runs/a21_repro/`. Logs: `.cache/logs/puget/`. Environment and runners as run:
`scripts/puget/`; `scripts/chains/15_a20.sh` and `scripts/chains/16_a21.sh` run the same stages on any device.

## Timeline (IST)

| run | start | end | what |
|---|---|---|---|
| A19 | 04:38 | 07:51 | quadrature validation (4 tasks × 9 rules), base (4 tasks × 3 seeds), 124 cells (73 B1, 51 B2) |
| A20 | 07:51 | 11:28 | A19 rerun check; certification (3 seeds × 24 sizes); 30 steering + 30 LoRA cells; LoRA rerun; HF audit (2 shards); L4 under A19's rule |
| A21 | 11:28 | 16:04 | 51 MEMIT edits (17 sites × 3 folds) + capped + noise = 153 cells; MEMIT rerun; Mac layer-0 check; HF audit (2 shards) |

GPU: one experiment at a time, ≤ 30 GB (audits as two 15 GB shards), thermal guard pausing at 80 °C
(one 15 s pause at 80 °C during A20's LoRA training; otherwise 55–79 °C, 240–420 W).

## Checks before and during the runs

- **Device transfer.** puget reproduced the Mac's b1_en seed-0 size-curve patching ratios exactly (0.34653
  at 10 %, 0.76806 at 20 %), the ablation ratios within one bf16 step, and the base audit's headlines
  (WikiText −3.3711 vs −3.3726; MMLU, refusal, facts identical). Uploaded cell artifacts: 393 cells,
  0 checksum differences.
- **Dry run** of every A20/A21 stage at tiny sizes: caught one wrong assertion in `certify.py`
  (CircuitKIT's node count includes `input`), fixed before any real run.
- **A19 validity (all pass).** Riemann-3 per-example mean vs CircuitKIT `attribute`: Spearman 0.9995–0.9998,
  relative L2 0.0067–0.0096. Chosen rule `midpoint6` (mean relative L2 to the 48-step reference 0.078 vs
  0.199 for the best 3-pass rule, below half, so the 6-pass rule is taken as sealed). Completeness gap
  smaller than Riemann-3's on every task (0.46 / 0.92 / 0.26 / 0.79 vs 0.67 / 1.25 / 0.77 / 1.12): better,
  but far from exact; the integrand is not smooth. `b1_lora_circuit_s0` re-attributed: per-example scores
  and jackknife SEs identical to the last bit (`checks/results/a19_repro.json`).
- **A20.** Every seed certified; the realised node set equals the intended one at every evaluation (node
  count asserted, certified sets checked node by node). `cert_s0` LoRA trained twice: max |ΔW| difference
  0.0 (`checks/results/a20_a21_repro_lora.json`). Entry matching: trained entries cert 11.79 M vs `cert_strat`
  11.81 M (0.16 %), `cert_random` 12.22 M, unrestricted 15.11 M.
- **A21.** All 153 cells exist, no edit failed (50–60 of 60 facts written per edit); capped and noise
  norms 59.16–59.17 against the cap 59.19 (float32 summation in the norm helper; within the 1 % check).
  `site10_s0` rebuilt: relative L2 0.0 (`checks/results/a20_a21_repro_memit.json`; the rebuild reused the cached
  covariance, so it covers the z-optimisation and the solve, not the covariance pass). GPU float64
  covariance = the Mac's CPU function after its float32 rounding (difference 0 on layers 0 and 17).
  puget `site00_s<d>` vs the Mac's MEMIT `circuit_s<d>` (same layer, fold, settings): relative L2 2.8 % /
  1.7 % / 15.1 %, norms equal to 0.01 % for folds 0 and 1, 132.6 vs 131.1 for fold 2
  (`checks/results/a20_a21_repro_mac.json`); a cross-device difference only, every A21 comparison is within puget.
- **Every hf.json** of A20 (61) and A21 (154) has finite headline numbers.

## Item 3 (A19): the inside view with a paired uncertainty

Sealed analysis: jackknifed paired CAD (circuit cell − same-seed controls, identical 32 examples per task),
pooled over seeds × bystanders, TOST at δ_L4 = 0.05. Two post-hoc sensitivity columns were added after the
runs (not sealed): SE = SD of the k contrasts / √k with k − 1 df, and the seed-clustered version (bystanders
averaged within a seed first, k = 3, 2 df), because the jackknife SE covers example sampling only and the
contrasts' spread across cells is about 10× larger. A verdict is called robust only when all three agree.

| | B1 (fact recall) | B2 (refusal) |
|---|---|---|
| MEMIT circuit − every control | +0.32 … +0.37 CAD, different (robust) | +0.23 … +0.26, different (robust) |
| steering circuit − inverse | −0.055, different (robust) | +0.096, different under two of three |
| steering circuit − taylor | −0.026, equivalent (robust) | +0.058, different (robust) |
| steering circuit − random | +0.017, equivalent (robust) | +0.027, not robust |
| LoRA circuit − random | +0.011, equivalent (robust) | +0.010, equivalent (robust) |
| LoRA circuit − unrestricted | −0.046, different (sealed) but inconclusive under SD/√k | −0.011, equivalent (robust) |
| A15 joint sets (LoRA, steering) | −0.001 … −0.024, equivalent under the sealed and seed-clustered SE | – |

Reading: the circuit-site MEMIT edit (MLP 0) disturbs every bystander circuit by far more than the ±0.05
margin; for steering and LoRA the inside view now gives answers instead of under-powered nulls, and most of
them are "equivalent within ±0.05". L3 contrasts with TOST at δ_L3 = 0.01 are in the same files; B1
steering on the pre-registered set is *more* disruptive on L3 than random (+0.013, different), LoRA is
equivalent to random. The attribution-free bystander measure (part d) agrees in direction with L4: MEMIT
moves the target and the language-ID and refusal bystanders by more than 5 % of their spans, LoRA moves
the refusal and language-ID bystanders, steering moves none (B1, `a19_b1.md`).

## Item 1 (A20): a certified circuit, and LoRA / steering on it

| seed | certified size | heads + MLPs | sufficiency (held-out 128) | necessity r′ (mean / zero) |
|---|---|---|---|---|
| 0 | 135 (22 %) | 101 + 34 | 0.953 | −0.021 / −0.034 |
| 1 | 122 (20 %) | 91 + 31 | 0.829 | −0.048 / −0.048 |
| 2 | 86 (14 %) | 57 + 29 | 0.829 | −0.023 / −0.019 |

A sufficient circuit exists on every seed, but it holds 14–22 % of the nodes and 29–34 of the 36 MLPs:
fact recall in Tiny Aya is not sparse at node level. Area under sufficiency over [0, 0.5]: 0.70–0.74.

- **Efficacy.** LoRA fixes 0.99 of the edited facts in every condition, inverse included, so H1 on the
  certified set ("met", ratio 1.00) is a ceiling result and uninformative. Steering at the shared norm
  (0.5 × the main study's circuit size) moves efficacy by only 0.02–0.03 in every condition.
- **Locality.** No condition is more local than its controls. L3: every cert − control difference is below
  0.01 (TOST at δ_L3 = 0.01: equivalent, except LoRA cert − unrestricted −0.0085, inconclusive). L4 under
  A19's rule: all eight contrasts equivalent within ±0.05 under the sealed and both post-hoc SEs (largest
  LoRA cert − unrestricted −0.015).
- **Conclusion.** Restricting LoRA or steering to a certified, sufficient circuit buys no measurable
  locality over random heads of the same size and placement; the earlier null was not an artefact of an
  unfaithful circuit. Caveat: with 29–34 of 36 MLPs inside every certified set, count-matched controls
  share almost all MLPs, so the contrast is mostly about which heads are trained.

## Item 2 (A21): MEMIT at 17 sites, uncapped, at equal norm, and noise at equal norm

Means over the three folds (full tables in `a21_b1.md`):

| site | uncapped |ΔW| | efficacy unc. / capped | cross-lingual gain unc. / capped | L3 unc. / capped / noise | MMLU noise |
|---|---|---|---|---|---|
| 0 (pre-registered circuit) | 131 | 0.37 / 0.38 | 0.03 / 0.06 | 0.185 / 0.050 / 0.082 | −0.117 |
| 3 (A8 causal trace) | 69 | 0.79 / 0.77 | 0.25 / 0.22 | 0.095 / 0.072 / 0.015 | −0.022 |
| 4–10 | 56–73 | 0.83–0.88 / 0.83–0.88 | 0.28–0.35 / 0.26–0.35 | 0.09–0.16 / 0.10–0.15 / ≈ 0.014 | −0.03 … +0.03 |
| 24–28 | 35–41 | 0.45–0.57 / 0.23–0.37 | 0.04–0.10 / −0.01–0.05 | 0.04–0.06 / 0.11–0.15 / ≈ 0.016 | +0.01 … +0.03 |

- **Layer 0 is the worst edit site.** It writes the fewest facts (0.37 against 0.83–0.88 at layers 4–10) and
  propagates least to hi/ar/fr/sw (0.03 against 0.28–0.35). Its large L3 collateral (0.185) is mostly
  size: at the common norm it drops to 0.050, among the lowest (layer 2: 0.047; layers 4–28: 0.10–0.15),
  while its efficacy stays at 0.38.
  Layer 0 is also the only site where random noise of the same norm does real damage (L3 0.082 vs ≈ 0.014,
  MMLU −0.117 vs within ±0.03), i.e. "layer 0 is fragile" is a property of the site, not of MEMIT.
- **The sealed test is negative.** The partial Spearman between the circuit's score at a layer and the
  outcome, given depth (and |ΔW| when uncapped), rejects none of the six primary tests under Holm
  (efficacy −0.17 / −0.26, L3 −0.11 / +0.09, L1 neighbourhood −0.17 / −0.20; uncapped / capped). Outside
  the primary set, the score is negatively related to cross-lingual gain (−0.38 / −0.39, p = 0.009 /
  0.007) and to L5 AURC change: where the attribution score is high, edits propagate *worse*.
- **Deep edits damage selective prediction.** L5 AURC change is +0.4–1.0 for capped edits at layers 14–26
  (+0.03 at layer 0, +0.06–0.29 at layers 2–12), which noise of the same norm does not do (≤ 0.002).
- **Reading for H5 and the MEMIT verdicts.** The pre-registered MEMIT results are about the worst site on
  this model. The circuit attribution score does not identify a good edit site; the causal-tracing site
  (layer 3) and the early-middle layers 4–10 write and propagate facts far better at the same norm.

## Caveats carried into the report

- **Audit order noise (main study and A20).** Until 2026-11-09 `audit/run_audit.py` iterated the edited
  facts in Python set order, which follows each process's hash seed; batching and bf16 padding then flip
  near-tie answers. Measured on one A20 cell audited twice: 2 of 1990 scored items differed (edited-fact
  families only; MMLU, MGSM, IFEval, refusal, WikiText and hidden identical). About ±1 item in 60 per
  edited-fact family per cell, unbiased. Fixed (`sorted(edited)`) before A21's audit, so A21 is exactly
  reproducible across processes; the main study and A20 are not re-run.
- **A19's sealed SE covers example sampling only.** Verdicts are reported with the two post-hoc
  sensitivity columns; only the claims that survive all three are stated above as robust.
- **Quadrature.** `midpoint6` halves the integration error of CircuitKIT's rule but leaves a 26–92 %
  completeness gap; L4 numbers are more accurate, not exact.
- **A21 scope.** 17 of the 29 editable layers (all even layers plus 3 and 5); the subject-position site
  rule of the plan and AlphaEdit were not run.
- **Cross-device.** MEMIT edits at layer 0 differ from the Mac's by 1.7–15 % in relative L2 (prefix
  sampling differs across devices); no comparison mixes devices.
