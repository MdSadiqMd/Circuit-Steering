# circuit-steering

Controlled test of circuit-targeted interventions on `CohereLabs/tiny-aya-global`: activation steering,
circuit-restricted LoRA and knowledge editing confined to the components an EAP-IG circuit points to,
compared against size-matched random, inverse, Taylor and unrestricted conditions, with a five-part
**Nothing-Else-Broke audit** (behavioural, safety, cross-lingual in en/hi/ar/fr/sw, mechanistic
cross-circuit CAD, hidden failures). Built on Lexsi's CircuitKIT, AuditKIT and SafeTune.

**Start here:** the write-up is `docs/article.md` (abstract: `docs/abstract.md`, form answers:
`docs/submission.md`); the numbers behind it are in `docs/results/a22_b2.md`, `a24_b2.md` and
`standard_checks.md`. In the code and result files the first adapter is `a22` and the second `a24`.

## Headline results

Held-out test prompts: 159 harmful per language (two of the three harm categories never trained
on) and 59 harmless look-alikes per language. Harmful compliance is decided by an LLM judge that
reads the reply.

| | base | first adapter | second adapter | SafeTune steering |
|---|---|---|---|---|
| harmful compliance, Hindi / Arabic / French / Swahili | 12.4 % | 3.8 % | 7.1 % | 8.8 % |
| harmful compliance, English | 8.8 % | 1.9 % | 4.4 % | 3.8 % |
| refusal of harmless look-alikes, 5 languages | 3.1 % | 8.8 % | 6.1 % | 15.3 % |
| perplexity, WikiText | 1.00× | 1.26× | 1.02× | 1.04× |
| perplexity, unseen text in 5 languages | 1.00× | 1.40× | 1.02× | 1.04× |
| replies to harmful prompts that are only the refusal, hi / ar / fr / sw | – | 32 / 2 / 11 / 1 % | 87 / 87 / 84 / 91 % | – |

- **The adapters make the model safer.** First adapter: −8.6 points outside English (95 % CI −11.2
  to −6.3). Second: −5.3 (−7.7 to −3.1). Over three seeds they average 5.9 % and 6.6 %.
- **Against SafeTune's refusal-direction steering** (strength tuned on the same dev split, one
  run): the first adapter is better on compliance (−5.0 points, CI −7.2 to −3.0) and on harmless
  refusals. The second is about level on compliance and refuses far fewer harmless prompts.
- **The audit found two defects in our first adapter**: perplexity +26 %, and outside English it
  usually kept writing after refusing. The second adapter fixes both.
- **The circuit is not why it works.** Adapters on random components of the same size differ by
  0.2–0.7 points. The 30 % refusal circuit holds 33–35 of the model's 36 MLPs.
- **The usual side-effect check is too weak.** Of the 92 earlier interventions that pass
  "perplexity within 5 %, MMLU within 5 points", 31 moved something we audit by ≥ 10 points
  (random noise of the same size: 2 of 68). The threshold was chosen after the study.
- **Counting refusal phrases misleads outside English.** Our phrase detector shows a 41.5-point
  refusal gap between English and Hindi; the judge finds 23 points, and 2.5 points in compliance.
- **Costs that remain**: harmless prompts are refused 3–6 points more often (15–17 in Swahili) and
  accuracy on reworded facts falls 3.5–4 points. A rule fixed in advance keeps the first adapter as
  the official release; both are published and we recommend the second.

- Spec: `docs/project-brief.md` · plan: `docs/plan.md` · pre-registration: `docs/preregistration.md`
- API discrepancies and measurements: `docs/notes.md` · decisions: `docs/decisions.md`
- Layout: three sibling repos in one folder; this one's folders are listed under "Repository layout".
- Next experiments (the plan after A21, with context): `docs/next-experiments.md`
- Toolkit defects found, ready to file: `docs/upstream-issues.md` · model cards: `docs/model_cards/`
- On Hugging Face: [models and demo](https://huggingface.co/collections/MdSadiqMd/circuit-steering-on-tiny-aya-models-and-demo-6af87a1b0a621b28feed3bf7)
  · [datasets and logs](https://huggingface.co/collections/MdSadiqMd/circuit-steering-on-tiny-aya-datasets-and-logs-6af87a1cee0c921b998574ef).
  The built refusal sets (`data/b2/`, `data/a22/`) contain harmful prompt text, so they are not in
  this repo; they are in the dataset
  [`circuit-steering-data`](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-data).

## Repository layout (three repos, one pipeline)
```
<folder>/
  circuit-steering/      this repo: the study (folders below). Depends on the two below as editable
                         path packages.
  tiny-aya-circuits/     package `aya_circuits`: CircuitKIT discovery, faithfulness pillars, same-path
                         random controls, overlap, circuits table; the shared model/config/manifest layer.
  nothing-else-broke/    package `neb`: the five-layer audit (behavioural, safety, cross-lingual,
                         mechanistic CAD, hidden failures), AuditKIT scoring, the two refusal judges.
```
`pyproject.toml` points at `../tiny-aya-circuits` and `../nothing-else-broke`, so the three checkouts sit
side by side (same layout locally and in production); `uv sync` here installs both. Every CLI runs from
this directory (`configs/base.yaml`, `data/`, `runs/` are resolved from the working directory, or from
`CIRCUIT_STEERING_ROOT`). The sibling repos have their own `uv.lock` and tests (`uv sync && uv run pytest`).

```
circuit-steering/
  data/             dataset builders and the built sets (b1/, b2/, bystanders/, a22/–a24/, MANIFEST.json)
  interventions/    steering, LoRA and MEMIT cells (run_grid.py; run_a20.py, run_a21.py, run_a22.py for
                    the amendments; export_peft.py writes a cell as a PEFT adapter and checks parity)
  audit/            audit glue (run_audit.py), L4 attribution variants (l4_*.py), analyses (*analyze.py),
                    A22/A24 test refusal, LLM judge and fluency (a22_eval.py, a22_judge.py, a23_fluency.py)
  checks/           stand-alone checks, one script each; what they measured is in checks/results/
  tests/            unit and entry-point tests (`just test`)
  configs/          base.yaml, the single source of defaults
  scripts/          lib.sh; chains/ = pipeline chains, NN_<name>.sh in run order; puget/ = A19-A21 as run
  docs/             brief, plan, pre-registration, report, notes, decisions; results/ = generated tables
  space/            the Hugging Face Space app
  runs/             outputs, git-ignored: the main study at the top level, side studies in a20/, a21/,
                    a22/, a23/ (one cell, stopped), a24/, pilot/, a20_repro/, a21_repro/, smoke/
  cache/, logs/     written by CircuitKIT into the working directory (built-in task data, its log), git-ignored
```

## Track repos chained (Lexsi track: CircuitKIT → SafeTune → AuditKIT, one pipeline)
Every stage below runs in `scripts/chains/01_full.sh` / `scripts/chains/03_safetune.sh`; `checks/smoke_lexsi_stack.py` proves the three
packages load and talk to each other on the real model before anything else runs.

| stage | repo | what is called |
|---|---|---|
| circuit discovery, signed edge scores | CircuitKIT 0.2.0 | `backends.eap.attribute`, `backends.eap.graph.Graph`, task registry (`aya_circuits/discover.py`, `aya_circuits/tasks.py`) |
| faithfulness pillars, same-path controls | CircuitKIT | `evaluation.full.run_full_faithfulness`, `api._reconstruct_circuit_graph` (`aya_circuits/faithfulness.py`, `aya_circuits/random_baseline.py`) |
| steering hook points, MEMIT/ROME edits | CircuitKIT | `applications.steering.ActivationSteering`, `applications.editing.MemitHandler` + covariance (`interventions/steer.py`, `interventions/edit.py`) |
| refusal-direction steering baseline (B2 condition `safetune_rd`) | SafeTune 0.1.10 | `interventions.steer.refusal_direction.extract_refusal_direction` (Arditi sweep) and `RefusalDirectionModel(mode="steer")` at apply time (`interventions/refusal_direction.py`, `interventions/cell.py`) |
| second refusal judge on every stored generation | SafeTune | the detector behind `core.eval.metrics.safety.OverRefusalMetric` (`neb/judge.py::refused_safetune`, "Judge robustness" in `docs/results/hypotheses_*.md`) |
| every before/after score and comparison | AuditKIT 1.1.3 | `ak.Sample` / `ak.scorer` / `ak.evaluate` for all L1–L3 families, `redteam.detectors.builtin.RefusalDetector` as the primary judge, `_bootstrap.paired_bootstrap` for every Δ (`neb/behavioural.py`, `neb/judge.py`, `audit/analyze.py`) |
| cross-circuit mechanistic audit (L4) | CircuitKIT | EAP-IG re-attribution under each intervention, induced-edge CAD (`neb/mechanistic.py`) |

Data flow: CircuitKIT circuits choose the components → interventions are built on them (CircuitKIT hooks /
MEMIT; SafeTune's direction as the baseline) → the same cells are scored by AuditKIT and re-judged by SafeTune
→ `audit/analyze.py` turns the AuditKIT runs into the hypothesis tests.

## Main study results (full write-up: `docs/report.md` v0.21; numbers from `audit/analyze.py`, `docs/results/`)
Pre-registered study (`docs/preregistration.md`, frozen before the first cell; 24 dated amendments (A22–A24 are the adapters above), the
frozen text never edited; A18 re-applied every LoRA cell as the exact trained adapter), 16 circuits plus the corrected IOI and length-matched refusal tasks (A16),
393 audited intervention cells (231 B1, 162 B2, sweep cells included) with a `scorecard.json` next to
every cell artifact, four audit records (`docs/audit-2026-11-0{5,6,7,8}.md`), every stage resumable. Brief requirements not met
(180 instead of ≥ 200 B1 facts; weight norm-matching replaced by a norm-matched perturbation control;
publication pending) are listed in `docs/report.md` §5.

- **Fact recall is localised, refusal is not, and the pre-registered cells intervened on neither.**
  CircuitKIT's joint-ranked 10 % fact circuit restores 33–56 % of the logit difference under patching
  (random node sets 0.000) and is necessary (ablating it removes the whole clean logit difference); the
  refusal circuit restores −0.04. The pre-registered per-type set (58 heads + 4 MLPs) restores only
  0.4–3 % for facts, because the four-MLP cap drops the MLPs that carry recall.
- **Confining an intervention to the pre-registered circuit did not buy locality anywhere.** H2 is
  refuted or null for all three interventions on both behaviours. MEMIT at the circuit's top MLP
  (layer 0) needs a 1.9–3.6× larger weight change than any other site and causes the most collateral; a
  random perturbation of the same norm at the same site is worse still, so at MLP 0 a change of that
  size is damaging whatever is written (at every other site the real edit does 4–12× the damage of
  equal-norm noise, so there it is the content).
- **On the faithful set (amendment A15, LoRA): parity from capacity, and one small locality
  advantage.** LoRA restricted to the joint-ranked fact circuit (40 heads + 21 MLPs) reaches the
  unrestricted adapter's efficacy (1.00 vs 0.99 with 53 % of its trained parameters) and carries the
  corrections into hi/ar/fr/sw (+0.45) — but count-matched random and inverse sets do the same, so parity
  is about adapting 21 MLP layers. Against count-matched random sets the circuit adapter damages known
  non-English facts less (L3 −0.008 [−0.017, −0.000], uncorrected p = 0.047, below the Holm line once
  the adapters are applied exactly, A18), the study's only locality advantage; it rests on three seeds
  (per-seed −0.013 / −0.007 / +0.000, paired p = 0.21), is null against the inverse set (so H2's
  pre-registered rule is not met) and null on the mechanistic layer. Steering on the same
  set is vacuous (efficacy ≤ 0.07) like every B1 steering cell.
- **Efficacy parity (H1) fails for the pre-registered site**: MEMIT 0.37 vs 0.86 unrestricted; LoRA
  0.41 vs 0.99 with random heads at 0.64; across the 28 non-control B1 LoRA cells efficacy tracks
  adapter size (Spearman 0.85). The circuit's deepest MLP (exploratory) reaches 0.88 with the same
  neighbourhood damage as every other single-layer edit.
- **Metrics that would have missed it**: every single-layer MEMIT raises never-edited top-1 error from
  0.02 to 0.28–0.65 and drops English fact neighbourhood accuracy 18–39 points while WikiText
  perplexity moves ≤ 1 % and MMLU ≤ 6 points; B2 LoRA raises benign over-refusal by 0.37–0.54 in English;
  MEMIT lowers refusal at most sites. Cells that weaken refusal are withheld from the Hub.
- **H3**: all 18 pre-registered circuit cells are gated on the set they intervene on and none shows an
  L4 advantage; the six A15 cells pass the gate, and the three LoRA ones show the predicted small
  advantage. **H4** is inconclusive: with the corrected IOI bystander, raw bystander CAD ranks with
  overlap (per-cell Spearman +0.71 / +0.68) but so do the bystanders' own attribution-noise floors, and
  the floor-adjusted ranking reverses (−0.25 / −0.39); the within-bystander split (A17) shows that where a
  bystander's edges move follows where the parameters changed, with no consistent difference between
  trained changes and norm-matched noise. **H5** is refuted at MLP 0 against the causal-tracing and Taylor sites.
- **The mechanistic layer is under-sampled, not blind**: at 64 examples / 10 IG steps the EAP-IG
  resample floors halve while the cells' CADs fall less (target 0.38–0.75×, bystanders 0.71–1.08×); on
  the per-type masks the strongest weight interventions come within 0.02 of the floor and LoRA at the
  circuit and steering stay 0.12–0.24 below, on the joint masks unrestricted LoRA and Taylor-layer
  MEMIT exceed it. A full rerun at that
  budget (≈ 7× the L4 campaign) was not done.
- **SafeTune's refusal direction as the steering baseline (A4)**: the strongest refusal strengthener
  (multilingual gain 0.34 vs 0.22) and the least local cell of the study (benign over-refusal +0.81,
  neighbourhood −0.77, perplexity ×1.5). Re-judging every refusal generation with SafeTune's detector
  agrees with AuditKIT's on 99.2–99.5 %.
- **Toolkit findings reported upstream-ready**: EAP-IG scores scale as 1/batch_size with a batch-mean
  metric; CircuitKIT's built-in IOI EAP path reads its metric at the answer token with a double BOS
  (`checks/check_ioi_position.py`; the corrected task `ioi_fixed` is necessary, 2–5 % sufficient at
  10 %, with random sets at 0.000 and stability 0.86, and its use as the bystander turns H4 from
  "consistent in direction" into inconclusive); the ratio floor of
  signed ablation is −c/(cl − c), not 0.5; partitioned CAD comparisons are magnitude-confounded;
  merging a LoRA adapter into bf16 weights rounds 31–43 % of its entries to zero (A18 applies the
  adapters as exact fp32 hooks, the forward PEFT trains with); EAP-IG on the corrected IOI task agrees
  with exact patching at ρ 0.67 only.
- Measurement caveats with numbers are in `docs/report.md` §5, including the fifth audit's (closed by
  A18): `n_trainable` counts masked adapter entries (trained entries reported beside it), k/v adapters
  cover whole GQA groups, and the fact circuit was discovered on few office-holder questions while most
  edit targets are office holders.
- **Post-hoc countermeasures (A19–A21, sealed before running; `docs/results/puget-a19-a21.md`, report
  §4.6).** A19: per-example EAP-IG with a 6-point midpoint rule and jackknifed paired CAD turns most
  LoRA and steering L4 nulls into "equivalent within ±0.05"; MEMIT at MLP 0 moves every bystander
  circuit by 0.23–0.37. A20: a certified circuit (sufficiency ≥ 0.8 on held-out examples) exists on
  every seed but holds 14–22 % of the nodes and 29–34 of 36 MLPs; LoRA and steering on it are no more
  local than layer-matched random sets. A21: MEMIT at 17 layers shows layer 0, the pre-registered site,
  is the worst one (efficacy 0.37 against 0.83–0.88 at layers 4–10); the circuit's per-layer score
  does not predict a good edit site. What to run next: `docs/next-experiments.md`.

## Run

Recipes live in the `justfile` (`just` lists them); the pipeline chains are `scripts/chains/NN_<name>.sh`, numbered in run order.

```bash
# three checkouts side by side; the audit package's GitHub repository is named bystander-circuits
git clone https://github.com/MdSadiqMd/Circuit-Steering circuit-steering
git clone https://github.com/MdSadiqMd/tiny-aya-circuits
git clone https://github.com/MdSadiqMd/bystander-circuits nothing-else-broke
cd circuit-steering
uv sync --locked                          # Python 3.12, locked deps, the two sibling packages, dev tools
export HF_TOKEN=...                       # Cohere licence accepted on the Hub
just check                                # lint, tests, dry acceptance check, leak scan, shell, chain plans
just smoke                                # real-model acceptance check -> checks/results/stack_check.json
uv run python data/facts_wikidata.py && uv run python data/b1_build.py && uv run python data/b2_build.py
uv run python data/filter_model.py && uv run python data/translate.py && uv run python data/b2_baseline.py && uv run python data/b2_eap.py
just chain circuits                       # discovery + faithfulness, 3 seeds, all tasks (aya_circuits CLIs)
uv run python checks/check_eap_vs_patching.py task=b1_en   # EAP-IG vs true patching, top-50 nodes (brief §12.3)
uv run python interventions/run_grid.py grid.behaviour=b1 grid.intervention=edit   # and lora / steer, b2
uv run python interventions/refusal_direction.py                                   # SafeTune baseline (B2)
uv run python audit/run_audit.py grid.behaviour=b1 && uv run python audit/analyze.py grid.behaviour=b1
just chain full                           # or everything above as one resumable chain
JUDGE_GGUF=/path/to/chat-model.gguf just chain a22   # refusal adapter, controls, test, audit, LLM judge (we used Qwen3.8-27B, Q5_K_M)
JUDGE_GGUF=/path/to/chat-model.gguf just chain a24   # the fluency-preserving adapter (needs a22)
uv run python interventions/export_peft.py cell=runs/a24/interventions/b2/lora/circuit30_s0 \
    to=runs/a24/release/tiny-aya-global-refusal-circuit-lora   # PEFT adapter + parity check
just reproduce                            # every chain in the order it was first run, full .. a24
just analyze                              # every table and analysis from the stored runs (CPU)
```

Each chain is resumable (every stage skips outputs that exist), stops at the first failing stage, and
logs to `.cache/logs/<name>.log`; `DRY_RUN=1 scripts/chains/NN_<name>.sh` prints its stages without running
them. A20 and A21 ran on a CUDA machine (`scripts/puget/`, kept as run); `scripts/chains/15_a20.sh` and
`scripts/chains/16_a21.sh` are the same stages for any device, e.g. `just chain a21 device=cuda`.

With nix (flake in this directory; uv, Python, just and bash come from nixpkgs, the recipes run from
the checkout):

```bash
nix develop                 # shell with uv, python 3.12, git, just, bash
nix run .#check             # just check
nix run .#full              # just chain full   (also .#tail, .#circuits, .#safetune)
```

Every script takes `key.sub=value` overrides of `configs/base.yaml`; the circuit, grid and audit stages
write a manifest (git SHA, `uv.lock` hash, model revision, seed, device, peak memory; artifact hashes
for the circuit and grid stages, none in the audit manifests), the data scripts do not (`data/MANIFEST.json` hashes their outputs post hoc). The recorded git SHA is
the skeleton commit for every artifact, because the code was never committed (report §5).
