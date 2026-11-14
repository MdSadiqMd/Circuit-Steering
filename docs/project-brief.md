# Research brief: circuit-level interventions on Tiny Aya, verified

**Task:** Change behaviour only at the components a discovered circuit points to, through circuit steering, circuit-restricted fine-tuning or knowledge editing that corrects false or harmful facts. Verify nothing else broke.

Single source of truth for this project: the task, the prior art, what to build, on what, and how we will know it is done. Event logistics (dates, prizes, judging, team rules) are deliberately omitted; the delivery requirements in Section 5 are the only external constraints that matter.

---

## 1. Problem statement

> Change behaviour only at the components a discovered circuit points to, through circuit steering, circuit-restricted fine-tuning or knowledge editing that corrects false or harmful facts. Verify nothing else broke.

Decomposed:

1. **Discover** a circuit (a sparse set of attention heads and MLP components) that is causally responsible for one target behaviour in the model.
2. **Intervene only there**, using each of the three named mechanisms:
   - circuit steering (activation-space, inference-time, no weight change),
   - circuit-restricted fine-tuning (LoRA or gradient mask on circuit components only),
   - knowledge editing at circuit components (ROME/MEMIT-style weight edits) that replaces a false or harmful fact.
3. **Verify nothing else broke.** Show, with measurements and controls, that everything outside the target behaviour is unchanged: other facts, other languages, general capability, safety and over-refusal, and the model's other internal circuits.

The hard part is step 3. Step 1 and 2 are implemented in Lexsi's CircuitKIT today. Step 3 is not implemented anywhere, and the field disputes whether step 2's "only there" buys anything at all.

---

## 2. Current state (verified against primary sources, 2026-11-02)

### 2.1 What Lexsi already built (reuse it, cite it)

#### CircuitKIT (arXiv 2607.19317, Seth, Gosalia, Kasliwal, Sankarapu, July 2026; github.com/Lexsi-Labs/CircuitKIT; docs circuitkit.lexsi.ai)
- Discovery: 13 algorithms, 6 stable (`eap`, `eap-ig` default, `eap-gp`, `acdc`, `ibcircuit`, `cdt`), node or neuron level, TransformerLens backend.
- Evaluation: six faithfulness pillars (causal patching, ablation with zero/mean/mean-positional, stability across resamples, robustness to corruptions, size-matched random and magnitude baselines, generalization to a related task) plus an optional seventh, intervention reliability across seeds. Normalized faithfulness F(C) = (y_circuit − y_corrupt)/(y_clean − y_corrupt).
- Applications already implemented: structural pruning, circuit-aware quantization, **selective fine-tuning** (`selective_finetune` returns per-layer q/k/v/o and MLP neuron indices for a gradient mask or PEFT `target_modules`), **knowledge editing** (`CircuitKnowledgeEditor.edit_via_circuit`, ROME at the most circuit-important MLP layer, MEMIT across top-K circuit layers, plus CaKE and MCircKE variants and a fine-tune fallback), **activation steering** (`ActivationSteering` at circuit heads via difference of means, `CircuitWeightSteering` = C-ΔΘ, `SteeringComposer`), an `UnlearningVerifier`, and HF checkpoint export.
- Model support: GPT-2, Llama-3, Gemma-2/3, Qwen-2.5/3 production; **Tiny Aya (cohere2), Command R7B, Aya Expanse 8B experimental but validated across discovery, evaluation and interventions**; Aya Expanse 32B registered only (64 GiB, multi-GPU TransformerLens bug).
- **What the paper found that matters for us.** Experiment E7 (selective fine-tuning, BoolQ, WikiText-2 perplexity): across 16 selector-model cells the mean circuit-mask minus random-mask accuracy difference is −0.001. The authors attribute the perplexity protection to freezing 70% of parameters, "not to which parameters are chosen". Taylor (gradient-magnitude) selection did best. E5 pruning: patch faithfulness vs retention ρ = −0.78 (n = 10, one cell). E6 quantization: ablation faithfulness vs retention ρ = +0.73 (n = 9). Most studies single-seed; E5 to E7 one model-task cell each. Stated future work: multi-seed intervention reliability, SAE features as a granularity.
- **What it does not have.** No locality, side-effect or collateral-damage evaluation for steering, selective LoRA or editing. The docs' application page reports none. Case study 23 (refusal localization on Llama-3.2-1B, Qwen2.5-1.5B, gemma-2-2b) measures only a coherence check and states "We do not claim circuit-guided intervention universally helps." Case study 22 (gender bias) selects selective-fine-tuning targets but never trains them and runs three qualitative prompts as its capability check. This is the hole the task's "verify nothing else broke" clause asks us to fill.

#### C-ΔΘ: Circuit-Restricted Weight Arithmetic for Selective Refusal (arXiv 2602.04521, Kasliwal, Seth, Sankarapu, Feb 2026, v2 Jul 2026)
- EAP-IG on refusal-vs-compliance template prefixes scores MLP output-projection channels; keep top κ = 15 to 20% ("<5% of parameters"). Fine-tune two masked copies (refusal templates, compliance templates), take Δθ = θ+ − θ−, add α·Δθ. Six Llama/Gemma models, five harm categories. Harmful refusal 14 to 93%, benign over-refusal 0.6 to 12.8%, vs activation steering at 22 to 68% over-refusal. Ablation without the EAP-IG mask (WS-MLP2): 86.0% refusal at 15.4% over-refusal vs C-ΔΘ 83.6% at 2.6%; inverse (bottom-κ) mask drops refusal by 20.2 points. MMLU within about 1.5 points, IFEval down up to 8.5 points, GSM8K in appendix.
- Limits the authors state: single seed, LLM judge, transfer attacks only, EAP-IG "behavioral relevance but not a complete causal account". **No size-matched random-mask control. No Aya or multilingual model. No mechanistic collateral measure.** The per-category circuits overlap heavily (mean pairwise Jaccard 0.571), which is a direct hook for an overlap-predicts-interference test.

#### Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution (arXiv 2605.15138, Sadhu, Seth, Sankarapu, May 2026)
- Introduces **Circuit Attribution Divergence** CAD(C) = Σ_e |s_e(θ) − s_e(θ′)| / Σ_e |s_e(θ)| over edges of the *forget* circuit, recomputed with EAP-IG after the intervention (0 = intact, ≈1 = dismantled, >1 = sign-flipped), and a companion **AS-NC**, activation shift outside the circuit. Four properties: forget, retain within 2 pp, non-positive post-quantization gap, structural erasure. Llama-3.1-8B-Instruct and Qwen-3-8B on WMDP and MUSE.
- CAD is computed only on the target circuit. **Nobody, including Lexsi, computes CAD on bystander circuits.** That extension is our mechanistic "nothing else broke" metric and it reuses their code path.

#### SafeTune (EMNLP 2026 System Demos; github.com/Lexsi-Labs/SafeTune)
- One API over published safety methods: `harden` (SafeGrad, LISA), `recover` (ReSta weight patching, Tiny Aya 3.35B supported with α≈0.25), `steer` (RefusalDirection, CAA, CAST, AdaSteer, C-ΔΘ), `unlearn` (RMU, NPO), `interpret.identify_safety_neurons`, `evaluate()` on HarmBench and custom prompts with the WildGuard judge. Each method carries a Faithful/Simplified/Variant/Wrong/Stub audit badge.

#### AuditKIT (v1.1.2, 2026; github.com/Lexsi-Labs/AuditKIT)
- `ak.evaluate()`, `ak.compare(base, tuned)` with bootstrap significance, `compare_models`, `evaluate_many`, red-team probes (jailbreak, injection, over-refusal), toxicity and bias, lm-evaluation-harness cross-check, sha256 run fingerprints, MLflow. No locality or ripple evals built in; we add those as datasets.

#### Framing paper
- Interpretability as Alignment: Making Internal Understanding a Design Principle (arXiv 2509.08592, Sengupta, Seth, Sankarapu, 2025). Argues interpretability must be constructive and pre-deployment. Our audit operationalizes that claim; cite it in the write-up.

Other Lexsi 2026 papers (AlignTune, CuratorKIT, ALIGNBEAM, tabular work) are not directly relevant.

### 2.2 Has "this" already been done?

Short answer: the three mechanisms exist, two of them inside CircuitKIT itself, but the *verification* half of the task has never been done properly, and the field is split on whether the localization step matters at all. Closest prior work per intervention:

| Intervention | Closest work | What it showed | What it lacks |
|---|---|---|---|
| Circuit-restricted fine-tuning | Circuit-tuning / "Fine-Tuning is Subgraph Search" (Li, Yuan, Wang, arXiv 2502.06106); circuit-aware LoRA (arXiv 2502.11812, +2.46% over LoRA, beats Random-LoRA on 5 tasks); CT-SFT (Nur'aini et al., EMNLP 2026 Findings, arXiv 2601.08146, CD-T heads, low-resource NusaX, "most consistently avoids catastrophic forgetting"); TopKLoRA (arXiv 2610.32866, Oct 2026, backdoor circuits); CircuitKIT E7 (null vs random) | Mixed: small gains or null vs random | Random controls inconsistent, single seeds, no mechanistic collateral, no multilingual model, no steering/editing comparison |
| Circuit-guided knowledge editing | ROME/MEMIT layer choice via causal tracing (Meng 2022/23); Hase et al. NeurIPS 2023 (tracing does not predict edit success); CaKE (Yao et al., EMNLP 2025, "localization area is not correlated to the performance"); MCircKE (Zhao et al., EMNLP 2026 Findings, arXiv 2604.05876, update only inside mapped reasoning circuit, MQuAKE); Anatomy of an Edit (Cao, Wang, Schütze, arXiv 2603.20795, post-edit attribution + MEGA steering); KLOD (EMNLP 2026 Findings, locality as non-target distribution preservation); FiNE (ICLR 2025, neuron-level localization) | Circuit-aware edits help multi-hop; layer localization does not predict edit success | No size-matched random-circuit control in MCircKE/CaKE as far as visible; locality is behavioural only; English only |
| Circuit steering | C-ΔΘ (above); CRaFT circuit-guided refusal features (arXiv 2604.01604); "From Trait Vectors to Circuits" (arXiv 2609.22224: steering through selected graph keeps 93.4% of effect vs matched random graph which does not); Active Circuit Discovery (Symmetry 2026: circuit-selected features "only marginally, and not significantly" more steerable than random at large multipliers); Anthropic attribution-graph interventions (2025) | Graph-restricted steering mostly works, selectivity vs random is contested | Side effects rarely measured beyond one capability benchmark; no multilingual |
| Multilingual editing | LU-LAFNs language-agnostic factual neurons (COLING 2025, arXiv 2406.16416); X-KDE (ACL 2025 Findings); BMIKE-53; survey Durrani et al. EMNLP 2025 (parameter editors "suffer from poor cross-lingual propagation"); Paths Not Taken (EMNLP 2025, arXiv 2505.20546: English-centric recall then late translation, +35% recall via residual vectors); How Do Multilingual LMs Remember Facts (ACL 2025 Findings: subject enrichment language-independent, extraction language-specific) | Facts share language-agnostic neurons; edits propagate poorly across languages | **Zero knowledge-editing or circuit-intervention work on any Aya model** (searched 2026-11-02). Aya interpretability so far: language-specific neurons in Aya Expanse (arXiv 2507.22608), Tiny Aya SAE at layer 28 (Cohere Labs community, ~76% universal features, Farseen0/tiny-aya-saes) |

### 2.3 The dispute we must not pretend away (reviewers from the last six years know all of this)
Evidence that localization does **not** inform intervention:
- Hase, Bansal, Kim, Ghandeharioun, NeurIPS 2023: causal-tracing layer has near-zero correlation with best edit layer.
- Wang and Veitch, "Does Editing Provide Evidence for Localization?" (ICLR 2025 Blogposts, arXiv 2502.11447): optimal rank-1 edits at 16 random heads match probing-localized heads; 5 of 24 single random heads match full fine-tuning; "no single privileged location". Demands random-location baselines and falsifiable claims.
- Chen et al., "Navigating by Old Maps: The Pitfalls of Static Mechanistic Localization in LLM Post-Training" (arXiv 2605.06076, May 2026): Mistral-7B and Llama-3-8B, 15 tasks, conditions Free / Mech / Random (same count). Circuits migrate during SFT ("Free Evolution"); Mech shows "no substantial advantage over the Random strategy" on skill tasks (Induction: Random 0.92 vs Mech 0.89); the apparent success of locate-then-edit on knowledge tasks is an "illusion of effectiveness" because MLP-dominated circuits happen to migrate little (WMDP-Bio: Mech 0.85 vs Random 0.75 vs Free 0.72). New metrics: Circuit Distance, Circuit Stability, Circuit Conflict. Recommends validating against random baselines, not just against full updates.
- CircuitKIT E7: circuit mask vs random mask, mean Δ accuracy −0.001 over 16 cells.
- Probe Prompting (arXiv 2511.07002): count-matched random feature bag suppresses the source answer as often as the labelled circuit, but never produces the target.
- "A Circuit, Not The Circuit" (arXiv 2606.00930): near-disjoint component sets match the causal effect; localization non-unique.
- Faithfulness itself is fragile: Miller, Chughtai, Saunders 2024 (scores swing with ablation type and token handling); Shi et al. NeurIPS 2024 (hypothesis tests; discovered circuits satisfy idealized criteria "to varying degrees").

Evidence that it **does**, when done the right way:
- Guo, Syed, Sheshadri, Ewart, Dziugaite, "Mechanistic Unlearning", ICML 2025: editing the fact-lookup mechanism (not output-traced components) generalizes to MCQ (+40 points), resists relearning (20% recovered vs 63% for causal tracing), lowers latent-probe recoverability, with Random, Random-MLP, All-MLP and Nonlocalized controls. Key distinction: output-preserving localization vs mechanism localization.
- C-ΔΘ (above): inverse mask −20 points, mask-free variant 6× the over-refusal.
- REdit, "Reforming the Mechanism" (Lei et al., arXiv 2603.06923, Mar 2026): **Circuit-Interference Law**, "edit interference between reasoning patterns is proportional to the overlap of their neural circuits" (body: monotone negative relation with weighted Jaccard / edit distance / OT distance, robust to budgets, seeds, subsamples; no coefficients printed). Reshaping circuits before LoRA improves locality to ~94%. One backbone (Qwen2.5-3B-Instruct), propositional logic.
- Sharma et al., "A Low-Rank Subspace Analysis of LLM Interventions" (MI Workshop ICML 2026, arXiv 2606.14388): side effects between behaviours scale with subspace overlap and with proximity to the refuse/comply decision subspace, asymmetric upstream/downstream structure, 7B to 70B.
- CT-SFT, circuit-aware LoRA, MCircKE, Emotion circuits (arXiv 2510.11328, random-unit controls negligible).

The resolution the field is converging on, and which our design tests directly: localization helps **conditionally**, when (i) the localized object is a mechanism rather than an output trace, (ii) the circuit is faithful and stable, and (iii) the target circuit is disentangled from bystanders. Nobody has tested (i) to (iii) jointly, across steering, restricted fine-tuning and editing, with the same controls on the same model.

### 2.4 "Verify nothing else broke": what exists and what is missing
- Behavioural locality for edits: CounterFact/zsRE neighbourhood and paraphrase; ripple effects (Cohen et al. 2023); general-ability degradation (Gu et al. 2024); butterfly-effect collapse (Yang et al. 2024); sequential collapse and REVIVE (arXiv 2601.11042); KLOD distributional locality (2026).
- **Hidden failures that standard checks miss**: Anand, arXiv 2609.29587 (Aug 2026): after 1,000 LoRA edits on Qwen2.5-7B-Instruct, edit success 1.00, locality clean, MMLU unchanged, yet arbitration spread on never-edited facts drops 36%, high-confidence error rises 0.217 to 0.342; 3 of 5 model-method pairs collapse to chance MMLU while edit metrics still look perfect. Proposes arbitration score, risk-coverage AURC, and a **norm-matched random perturbation control**.
- Steering collateral: COAST (arXiv 2605.01167) minimizes expected squared change along non-target directions; "Analysing the Safety Pitfalls of Steering Vectors" (arXiv 2603.24543); "Safety Cost of Steering Vectors Is Separable" (arXiv 2608.08383); AxBench (ICML 2025: prompting beats steering, difference-in-means beats SAEs); Tan et al. NeurIPS 2024 on steering-vector reliability.
- Mechanistic collateral: CAD and AS-NC (Lexsi 2026, target circuit only); post-edit attribution (Anatomy of an Edit); Circuit Distance/Stability/Conflict during SFT (Old Maps); Circuit-Interference Law (REdit); subspace overlap (Sharma et al.).
- Survey: "Locate, Steer, and Improve" (Zhang et al., 29 authors, arXiv 2601.14004, Jan to Apr 2026) states "measuring unintended consequences is a first-class requirement" and that the field "lacks unified benchmarks or standardized evaluation protocols for localization methods". Its minimal framework per domain: safety = refusal rate + attack success, side effect = over-refusal (OR-Bench); knowledge editing = edit success, side effect = locality + fluency.

**Gaps nobody has filled**: (a) a mechanistic locality metric on *bystander* circuits; (b) cross-lingual collateral of any circuit-targeted intervention; (c) one protocol comparing steering vs restricted LoRA vs editing with identical controls; (d) a test of whether circuit faithfulness pillars predict downstream locality (CircuitKIT hints at it with n ≤ 10); (e) the Circuit-Interference Law outside one model and one task family.

---

## 3. End goal

### 3.1 One sentence
Produce, on Tiny Aya, a controlled measurement of whether confining steering, LoRA and knowledge editing to a discovered circuit changes the target behaviour with less collateral damage than confining them to size-matched random, inverse, gradient-selected, or no components, and ship the reusable **Nothing-Else-Broke audit** that produced the measurement.

### 3.2 Pre-registered hypotheses (write them to `docs/preregistration.md` with thresholds *before* the first intervention run; never edit after)
- **H1 efficacy parity.** Circuit-targeted intervention reaches ≥ 90% of the unrestricted intervention's target effect.
- **H2 locality advantage.** Circuit-targeted collateral is lower than size-matched random and inverse on the mechanistic and cross-lingual layers, 95% bootstrap CI excluding zero, ≥ 3 seeds. A null here is a valid, reportable outcome.
- **H3 faithfulness gates locality.** Across cells, CircuitKIT pillar scores (P1 patching, signed P2 ablation under zero and mean replacement, P3 stability, P5 random-baseline margin) correlate with the H2 advantage; cells failing P2 or P5 show no advantage.
- **H4 overlap predicts interference.** Collateral on bystander behaviour B is monotone in weighted-Jaccard overlap between the target circuit and B's circuit.
- **H5 cross-lingual propagation.** Editing a fact at circuit components propagates to the other languages at least as well as editing at English causal-tracing layers, with no larger collateral on unrelated facts in those languages.

### 3.3 What "done" means (all required)
1. Two target behaviours on Tiny Aya Global: **B1** false-fact correction (≥ 200 subject-relation-object triples, parallel prompts in en, hi, ar, fr, sw) and **B2** one harmful-request category with matched benign near-miss prompts.
2. Three bystander circuits discovered and evaluated: refusal (bystander for B1), fact recall (bystander for B2), and IOI as neutral control. A language-identification circuit is a fourth if time allows.
3. Full grid run: 3 interventions × 5 conditions × ≥ 3 seeds for each behaviour. Conditions: circuit-targeted, size-matched random (≥ 5 draws), inverse bottom-k, Taylor gradient-magnitude, unrestricted. Weight interventions norm-matched across conditions.
4. Nothing-Else-Broke audit computed for every cell, five layers:
   - **L1 behavioural locality:** neighbourhood and paraphrase accuracy, one-hop ripple, Global-MMLU subset, IFEval subset, GSM8K subset, WikiText perplexity.
   - **L2 safety:** refusal on HarmBench/SORRY-Bench category via SafeTune, over-refusal via XSTest / AuditKIT red-team probes.
   - **L3 cross-lingual collateral:** L1 and L2 repeated in the five languages; cross-lingual edit propagation rate.
   - **L4 mechanistic collateral:** **cross-circuit CAD** (CAD of each bystander circuit after the intervention), AS-NC, Circuit Distance of the target circuit, and target-vs-bystander circuit overlap.
   - **L5 hidden failures:** arbitration spread and risk-coverage AURC on never-edited facts (Anand 2026); relearning probe for B2.
5. One scorecard per cell (JSON + table), exported next to each HF checkpoint/adapter, with bootstrap CIs from AuditKIT `compare`.
6. Written report that states, for each of H1 to H5, supported / refuted / inconclusive with the numbers, and that names the strongest alternative explanation for each result.
7. Public repository, public HF artifacts (datasets, adapters, edited checkpoints, logs, scorecards), and a public write-up with before-and-after metrics (more than one) and the rationale for every metric, dataset and model choice. A small HF Space showing base vs intervened outputs with the audit badge.

### 3.4 What this does not claim
"Verified" means every layer of the audit was measured and reported, with controls. It is necessary evidence, not a proof of absence of side effects. Say so in the report.

---

## 4. Model

| Role | Model | Why |
|---|---|---|
| **Primary (all results)** | `CohereLabs/tiny-aya-global` | 3.35B, 36 layers, d=2048, 16 heads / 4 KV heads, 70 languages, instruction-tuned (has refusal behaviour, needed for B2). `cohere2` architecture, supported by CircuitKIT, SafeTune (ReSta, steering layer ≈ 24) and AuditKIT `hf:` backend. Fits one 16 GB GPU in bf16 for LoRA and steering; EAP-IG discovery needs ~24 to 40 GB or the memory-optimization settings. |
| Replication (stretch) | `CohereLabs/aya-expanse-8b` | `cohere1`, 23 languages, validated in CircuitKIT. Only after the full Tiny Aya grid is complete. |
| Pipeline smoke only | `gpt2` on CPU | CircuitKIT's documented quickstart. Never report numbers from it. |

Not allowed as primary: `tiny-aya-base` (no refusal behaviour), Fire/Water/Earth (merged region variants, confounds), Aya Expanse 32B (CircuitKIT interventions do not run), any non-Cohere model.

Both Cohere models are gated (CC-BY-NC 4.0). Accept the licence on the HF model page and set `HF_TOKEN` before anything else.

---

## 5. Hard constraints

**MUST**
- Use Lexsi packages for the core pipeline: `circuitkit` for discovery, faithfulness evaluation, steering, selective LoRA masks, and editing; `auditkit` for every before/after comparison; `safetune` for refusal/over-refusal evaluation and the refusal-direction steering baseline. At least two of these must be chained in the same pipeline; we chain all three. `curatorkit` may be used for building the multilingual pairs.
- Use `CohereLabs/tiny-aya-global` for all reported results (Section 4).
- Host every model, adapter, edited checkpoint, dataset, log and scorecard on Hugging Face.
- Report more than one before-and-after metric per intervention, each with bootstrap CIs, and explain why each metric, dataset and model was chosen.
- Run every intervention in all five conditions of Section 3.3 item 3, with ≥ 3 seeds, before drawing any conclusion.
- Report CircuitKIT ablation faithfulness **signed**, under both zero and mean replacement. Never clamp or report a single faithfulness number.
- Pre-register H1 to H5 with thresholds before the first intervention run.
- Keep the repository structure flat and named by stage: `data/`, `circuits/`, `interventions/`, `audit/`, `checks/`, `docs/`.
- Follow the engineering standards (Section 11), the verifiable-research standards (Section 12) and deliver the four required novel contributions (Section 13). Verify every external API against installed source or docs, with a web search of the primary paper when semantics matter, before writing the call.

**MUST NOT**
- Do not substitute Qwen/Llama/Gemma for the primary model, even if CircuitKIT's examples use them.
- Do not report an intervention as "local" or "safe" without the random, inverse and unrestricted controls in the same table.
- Do not publish refusal-weakening artifacts (negative-α deltas or inverted refusal circuits). Release only safety-strengthening checkpoints.
- Do not use `acdc` on Tiny Aya (impractically slow) or `ibcircuit` at aggressive settings above 3B (OOM).
- Do not edit the pre-registration after runs start; append amendments with dates instead.

---

## 6. Acceptance check: the Lexsi stack is really in use, on the right model

Create `checks/smoke_lexsi_stack.py`. It must pass on a GPU box before any experiment code is written, and its JSON output is committed. If a call signature differs from the installed version, fix the script against `help()` of the installed object; do not swap in a non-Lexsi library.

```python
"""Proves the three Lexsi packages load, run on CohereLabs/tiny-aya-global, and talk to each other.
Writes checks/stack_check.json. Exit code 0 only if every assert passes."""

import json, os, sys, importlib.metadata as md

import circuitkit as ck  # must be imported before transformer_lens
import auditkit as ak
import safetune

MODEL = "CohereLabs/tiny-aya-global"
assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set; accept the Cohere licence on HF first"

versions = {
    p: md.version(p)
    for p in ("circuitkit", "auditkit", "safetune", "transformer-lens", "transformers")
}

# 1. CircuitKIT loads the model and discovers + evaluates a circuit on it.
model = ck.load_model(MODEL)
cfg = getattr(getattr(model, "cfg", None), "original_architecture", None) or str(type(model))
assert "cohere" in str(cfg).lower(), f"not a Cohere architecture: {cfg}"
circuit = ck.discover(model, "ioi", algorithm="eap-ig", n_examples=16)
assert len(circuit.scores) > 0
report = (
    ck.evaluate_circuit(
        {"model": MODEL, "circuit": circuit, "pillars": ["patching", "ablation"], "n_examples": 16}
    )
    if hasattr(ck, "evaluate_circuit")
    else ck.Pipeline(MODEL, task="ioi").evaluate(pillars=["patching", "ablation"], n_examples=16)
)

# 2. The three intervention entry points exist and accept this circuit.
from circuitkit.applications.steering import ActivationSteering

steer = ActivationSteering(model, circuit.scores)
masks = ck.selective_finetune(circuit, model_name=MODEL, top_fraction=0.2, scope="both")
assert masks.attn or masks.mlp
from circuitkit.applications.editing import CircuitKnowledgeEditor

editor = CircuitKnowledgeEditor(model)  # construction only; the edit itself is an experiment

# 3. AuditKIT scores the same model and produces a comparison object.
samples = [
    ak.Sample(input="The capital of France is", target="Paris"),
    ak.Sample(input="La capitale de la France est", target="Paris"),
]
base = ak.evaluate(samples, f"hf:{MODEL}", ["exact_match", "f1_score"])
cmp = ak.compare(base, base)
assert cmp is not None

# 4. SafeTune exposes refusal evaluation and steering for this model family.
from safetune.runner import steer as st_steer
from safetune.evaluate import evaluate as st_eval

assert callable(st_eval) and hasattr(st_steer, "RefusalDirectionTrainer")

json.dump(
    {
        "model": MODEL,
        "arch": str(cfg),
        "versions": versions,
        "n_circuit_nodes": len(circuit.scores),
        "patching": getattr(report, "patching_score", None),
        "ablation_signed": getattr(report, "ablation_score", None),
        "selective_lora_layers": {"attn": len(masks.attn), "mlp": len(masks.mlp)},
        "auditkit_headline": getattr(base, "headline", None),
    },
    open("checks/stack_check.json", "w"),
    indent=2,
)
print("OK", versions)
```

Pass criteria: exit 0; `stack_check.json` shows `"model": "CohereLabs/tiny-aya-global"`, a Cohere architecture string, non-empty circuit, a patching score, non-empty selective-LoRA mask, an AuditKIT headline, and versions for all three Lexsi packages. Any `ImportError` from a Lexsi package, any model id other than the one above, or any fallback to a non-Lexsi path is a failure.

---

## 7. Execution order (each step is a hard prerequisite of the next)

1. Environment: `uv init` with Python 3.12 (SafeTune needs ≥ 3.12, CircuitKIT supports 3.10 to 3.12); `uv add circuitkit==0.2.0 auditkit safetune peft ruff pytest`; commit `uv.lock`; `make check` runs ruff + pytest + the dry acceptance check. On Google Colab runtimes run `uv pip uninstall torchao` before importing SafeTune (SafeTune README). Import `circuitkit` before `transformer_lens`. Use `dtype=` not `torch_dtype=`.
2. Run the acceptance check (Section 6). Commit `checks/stack_check.json`.
3. Build data. CircuitKIT custom data is a CSV with columns `prompt, answer, corrupted_prompt, corrupted_answer`, loaded via `Pipeline.from_custom_data(MODEL, csv_path, clean_prompt="{prompt}", clean_answer="{answer}", corrupt_prompt="{corrupted_prompt}", corrupt_answer="{corrupted_answer}", task_name=...)`. Metric `logit_diff` for single-token answers, `kl` for multi-token. Build B1 (5 languages, back-translation checked), B2 (harmful category + benign near-miss), and the bystander tasks. Push to HF.
4. Discover and evaluate every circuit (target and bystander), EAP-IG node level, 3 seeds, pillars patching/ablation/stability/baselines, both ablation replacements, signed. Record overlap matrix between all circuits. Smoke-test one ROME edit on Tiny Aya here; if `cohere2` editing fails, file the failure and fall back to MEMIT via CircuitKIT's fine-tune editing path, still circuit-restricted.
5. Write and freeze `docs/preregistration.md`.
6. Run the intervention grid. Save each checkpoint/adapter/steering vector with its condition, seed, and component list.
7. Run the five-layer audit on every cell through AuditKIT `compare` (L1, L3), SafeTune `evaluate` + AuditKIT red-team (L2, L3), CircuitKIT re-discovery for cross-circuit CAD / AS-NC / Circuit Distance (L4), arbitration + AURC scripts (L5).
8. Report H1 to H5, publish artifacts, the public write-up and the demo Space.

---

## 8. Known pitfalls (all from primary sources)

- Tiny Aya's chat template prepends a ~366-token system preamble; leave `max_len` unset or ≥ 512 in SafeTune loaders.
- CircuitKIT ROME picks MLP nodes only; a circuit with no MLP nodes raises `ValueError`. Discover with `scope="both"`.
- Older CircuitKIT exports dropped ROME/MEMIT and weight-steering edits; on 0.2.0 `export_checkpoint` writes current weights. Verify by reloading and re-scoring.
- Refusal circuits can invert under ablation (negative signed P2). This is real signal; report it.
- CircuitKIT's robustness pillar is uninterpretable for signed metrics; skip it on B2 and say why.
- WildGuard judge is a gated 7B (~14.5 GB); cache judgments, pre-filter with AuditKIT's keyword refusal detector.
- Faithfulness scores depend on ablation type and token handling (Miller et al. 2024). That is why both replacements are mandatory.
- Edits can pass edit-success, locality and MMLU while silently degrading evidence arbitration (Anand 2026). That is why L5 exists.
- Keep generated harmful content out of the repo and HF artifacts; store only prompts, judgments and aggregate scores.

---


## 9. Objections a senior reviewer will raise, and the built-in answer
| Objection | Answer in the design |
|---|---|
| "Random baselines beat localization (Hase, Wang and Veitch, Old Maps)." | They are the primary comparison, with inverse and gradient selectors too. A null is reported as a null. |
| "Faithfulness scores are not robust (Miller 2024)." | Both ablation replacements, stability across seeds, and faithfulness used as a stratifier, not a trophy. |
| "Edit success and locality can look perfect while the model is damaged (Anand 2026)." | Arbitration and AURC checks, norm-matched perturbation control. |
| "Circuits drift during fine-tuning so static circuits mislead (Old Maps)." | Circuit Distance of the target circuit is measured post-intervention; we test whether drift correlates with collateral. |
| "Only one model / toy scale." | Tiny Aya for the primary grid, Aya Expanse 8B and a non-Cohere family for the replication; 3.35B is above the 1 to 3B used by CircuitKIT, C-ΔΘ small cells and REdit. |
| "Steering effects are generic activation scaling (Active Circuit Discovery)." | Size-matched random steering at matched coefficient norms. |
| "Multilingual results reflect translation artifacts." | Parallel prompts verified by back-translation, language-ID bystander circuit, per-language locality reported separately. |
| "You just ran a toolkit." | The toolkit has no locality evaluation; the audit, the controls and the overlap law test are the contribution, and the toolkit authors' own E7 null is the motivating result. |

## 10. Risks and mitigations
- Tiny Aya is experimental in CircuitKIT; ROME/MEMIT need a cohere2 covariance corpus and registry entries. Smoke-test discovery + edit first; fall back to Aya Expanse 8B if cohere2 editing is broken, or to MEMIT via the fine-tune fallback.
- EAP-IG on 3.35B on a T4 is memory-tight. Use bf16, batch of 8, 5 IG steps, a 40 GB class GPU for discovery; LoRA and steering runs fit on 16 GB.
- Refusal circuits can invert under ablation (CircuitKIT case 23 and E4). Report signed P2; never clamp.
- Dual use: do not publish the inverted (−α) refusal deltas; release only safety-strengthening checkpoints, mirroring C-ΔΘ's gating.

## 11. Engineering standards (hard requirements)

**Toolchain**
- `uv` for everything Python: `uv init`, `pyproject.toml` with pinned `circuitkit==0.2.0`, `auditkit`, `safetune`, `peft`, `torch`; `uv lock` committed; run with `uv run`. No bare `pip`, no conda, no notebooks as the source of truth (notebooks may call package code, never hold logic).
- `ruff` for lint and format (`ruff check --fix`, `ruff format`), configured in `pyproject.toml` with `select = ["E","F","I","B","UP","SIM","PERF","NPY"]`, line length 100. CI-equivalent: a `make check` target that runs `ruff`, `uv run pytest -q`, and the acceptance check in dry mode.
- Python 3.12 (intersection of CircuitKIT 3.10 to 3.12 and SafeTune ≥ 3.12).
- Every script takes a config (YAML) and writes a run manifest: git SHA, `uv.lock` hash, model id and revision, seed, dtype, device, wall time, peak VRAM, and the sha256 of every input dataset and output artifact. AuditKIT already fingerprints runs; reuse it and never write a second tracker.
- Lean code: one module per stage (`data`, `circuits`, `interventions`, `audit`), functions over classes, no abstractions for one use, no option that is never set. Comments only where a formula, invariant or source is non-obvious; put the paper or doc URL in the comment.

**Verify before writing.** Before writing any call into CircuitKIT, SafeTune, AuditKIT, PEFT, TransformerLens or Transformers, open the installed source (`uv run python -c "import inspect, circuitkit; print(inspect.getsource(circuitkit.selective_finetune))"`) or the official docs, and web-search the method's primary paper when behaviour matters (ablation semantics, IG steps, norm conventions). Record the source URL or file path in a comment. A guessed signature or a half-remembered formula is a bug. If docs and source disagree, source wins; file the discrepancy in `docs/notes.md`.

**GPU code (write it tight, measure it, then stop)**
- bf16 weights and activations everywhere; fp32 only for attribution accumulators and the covariance statistics ROME/MEMIT need. Use `dtype=` (CircuitKIT 0.2.0), not `torch_dtype=`.
- Discovery (EAP-IG) is the cost centre: O(ig_steps) forward+backward passes per batch over 36 layers. Budget it: start at `n_examples=64, ig_steps=5, batch_size=4` on Tiny Aya, then scale only what the stability pillar says is under-sampled. Set `CIRCUITKIT_TL_MEMORY_PATCH=1` once you have confirmed scores match the unpatched run on a 16-example subset (it changes bf16 accumulation order). Never run `ibcircuit` at aggressive settings above 3B, never run `acdc` on Aya.
- Cache every activation or attribution tensor that is reused (bystander circuits, clean/corrupt logits, EAP-IG scores) to disk as safetensors keyed by dataset hash + seed; recompute only on cache miss.
- LoRA runs: PEFT with `gradient_checkpointing`, SDPA attention (`attn_implementation="sdpa"`), fused AdamW, batch by token count not example count, `torch.compile` only on the plain HF model path (not under TransformerLens hooks), and a 20-step profile with `torch.profiler` before any longer run. Report tokens/s and peak VRAM in the manifest.
- Evaluation: batch generation with left padding, KV cache on, deterministic decoding, max_new_tokens fixed per task; cache WildGuard judgments by prompt+completion hash.
- One GPU at a time; no multi-GPU TransformerLens (broken in 3.8.0). Aya Expanse 8B in bf16 fits a 24 GB card for steering/LoRA; discovery on 8B needs 40 to 80 GB or the memory patch plus small batches.
- Determinism: fixed seeds for torch/numpy/python, `torch.use_deterministic_algorithms(True)` where it does not break SDPA, and seeds in every manifest. Random-component control draws are generated from a seeded RNG and the component lists are saved.
- Optimise only what the profiler shows is slow. A lean correct script beats a clever one.

## 12. Verifiable-research standards (what the mech-interp community will check)

Adopt these from the literature; they are not optional.

1. **Report faithfulness as a curve, not a point.** MIB (Mueller et al., ICML 2025) scores circuits by the area under faithfulness vs circuit size. For every discovered circuit report faithfulness at 1, 2, 5, 10, 20, 30% of components and the area, under both zero and mean ablation, signed.
2. **Controls every time.** Size-matched random circuits (≥ 5 draws), magnitude/Taylor circuits, inverse (bottom-k) circuits, and the unrestricted intervention sit in every results table (Wang & Veitch 2025; Chen et al. 2026). A circuit result without them is not reported.
3. **Attribution is an approximation; check it.** EAP/EAP-IG is a first-order estimate of activation patching and can misrank components under downstream non-linearity (Zhang & Wang 2026, "When Attribution Patching Lies"; Syed et al. 2023). On a 32-example subset, compare EAP-IG scores against true activation patching for the top-50 components; report Spearman ρ. If ρ < 0.7, raise `ig_steps` or apply an HVP-corrected pass before trusting the circuit.
4. **Test the circuit hypothesis, not just the score.** Run sufficiency (circuit alone recovers behaviour), necessity (ablating the circuit removes it), and minimality (dropping random nodes from the circuit hurts) checks in the spirit of Shi et al. NeurIPS 2024 and the IOI completeness/minimality criteria. CircuitKIT pillars 1, 2 and 5 cover sufficiency, necessity and the random baseline; add the minimality check yourself.
5. **Beware subspace and pathway illusions.** A component that moves the output under patching can act through a dormant pathway (Makelov et al., ICLR 2024). Steering directions must be validated by patching the full component as well as the direction, and the cross-circuit CAD must be computed on edges, not only nodes.
6. **Seeds and intervals.** ≥ 3 discovery seeds, ≥ 3 intervention seeds, ≥ 5 random-control draws, bootstrap 95% CIs on every metric (AuditKIT `compare`), effect sizes alongside p-values. Single-seed numbers are labelled "directional" and never used to support a hypothesis.
7. **Pre-register, then do not move the goalposts.** `docs/preregistration.md` holds H1 to H5, the thresholds, the metrics, the exact datasets (by hash) and the analysis code path, dated before the first intervention run. Amendments are appended with dates and reasons.
8. **Mechanistic claims need mechanistic evidence.** "The edit landed in the circuit" is shown by re-running discovery after the edit (post-edit attribution, Cao et al. 2026) and by CAD on the target circuit, not by output accuracy alone. "Nothing else broke" is shown by cross-circuit CAD and AS-NC on bystanders plus the behavioural layers, never by one benchmark.
9. **Report negatives and failures.** A null on H2 is a finding. Failed runs, OOMs, judge disagreements and inverted circuits go in the report's limitations with numbers.
10. **Release everything needed to re-run.** Datasets with hashes, circuits as CircuitKIT artifacts (`.pt` + `_scores.json`), component lists for every condition, adapters and edited checkpoints, judgments, manifests, and the exact `uv.lock`.

## 13. Novelty requirements and idea backlog

The project must contribute at least the first four of these; the rest are ranked stretch ideas. Each one is checked against the prior art in Section 2 before being claimed.

**Required contributions**
1. **Cross-circuit CAD.** Extend Lexsi's Circuit Attribution Divergence from the target circuit to bystander circuits as a mechanistic locality metric, computed on edges, with AS-NC and target-circuit Circuit Distance. No prior work computes attribution drift on non-target circuits.
2. **Faithfulness-gated locality (H3).** Test whether CircuitKIT's pillar scores predict whether circuit targeting beats random targeting. CircuitKIT hinted at this with n ≤ 10 on pruning/quantization; nobody has tested it for steering, LoRA or editing.
3. **Circuit-Interference Law across interventions and languages (H4).** REdit stated it for LoRA edits on one model and one task family. Test it for steering, LoRA and editing on Tiny Aya with cross-lingual bystanders.
4. **First circuit-targeted editing/steering results on an Aya model with cross-lingual collateral (H5).** Compare circuit components against English causal-tracing layers for propagation into hi/ar/fr/sw and collateral there.

**Stretch ideas, in order of value per hour**
5. **Norm-matched perturbation control** (from Anand 2026): add a random weight perturbation with the same Δθ norm as each weight intervention to separate "localized edit" effects from "any perturbation of this size" effects.
6. **Circuit drift during restricted LoRA.** Track Circuit Distance of the target circuit every N steps during circuit-restricted vs random-restricted LoRA (Chen et al. 2026 did this for full SFT only). Prediction: circuit-restricted training keeps the circuit in place; random does not; drift correlates with collateral.
7. **Language-agnostic vs language-specific split.** Partition the B1 circuit into components shared across the five languages' circuits and language-specific ones (LAFN, COLING 2025; Paths Not Taken, EMNLP 2025). Edit only the shared part; predict propagation improves and language-specific collateral drops.
8. **Attribution reliability on Aya.** Apply the HVP correction or a reliability score to EAP-IG on Tiny Aya and report where first-order attribution mis-ranks components; this is the first such check on a cohere2 model.
9. **Intervention composition.** Apply the B1 edit and the B2 steering together; measure whether collateral is additive or super-additive, and whether circuit overlap predicts the interaction.
10. **Edit permanence under quantization.** Re-run the audit on an NF4-quantized edited checkpoint (links to Lexsi's quantization-permanent unlearning work). Stretch only.
11. **Causal-variable check on the fact.** Use interchange interventions (IIA, DAS via pyvene) to confirm the edited representation actually carries the new object variable, not a shortcut. Stretch; only if the main grid is complete.

Ideas not on this list require a one-paragraph prior-art check appended to Section 2.2 before any code.

## 14. Reading list (minimum)
Lexsi: CircuitKIT arXiv 2607.19317; C-ΔΘ arXiv 2602.04521; Forgetting That Sticks arXiv 2605.15138; Interpretability as Alignment arXiv 2509.08592.
Controls and critiques: Hase et al. arXiv 2301.04213; Wang & Veitch arXiv 2502.11447; Chen et al. arXiv 2605.06076; Miller et al. arXiv 2407.08734; Anand arXiv 2609.29587.
Positive mechanism results: Guo et al. arXiv 2410.12949; REdit arXiv 2603.06923; Sharma et al. arXiv 2606.14388; MCircKE arXiv 2604.05876.
Multilingual: LAFN arXiv 2406.16416; Paths Not Taken arXiv 2505.20546; Tiny Aya arXiv 2603.11510.
Verification standards: MIB arXiv 2504.13151; Shi et al. arXiv 2410.13032; Makelov et al. arXiv 2311.17030; Zhang & Wang arXiv 2606.09899; Palumbo et al. ICML 2025 arXiv 2407.13594; Sharkey et al. Open Problems arXiv 2501.16496; BlackboxNLP 2025 shared task arXiv 2511.18409.

## Appendix A. Verified vs unverified (as of 2026-11-02)
- Verified from primary sources: the task text and delivery requirements from the task's source page; CircuitKIT README, paper sections 1 to 10 and results E1 to E7, docs for applications, evaluation framework, algorithms, case-study notebooks 22 and 23; C-ΔΘ method, results, limitations (main text and part of appendices); Forgetting That Sticks CAD formula and numbers; SafeTune and AuditKIT READMEs; Tiny Aya architecture and language list; all papers cited in sections 3 and 4 at least at abstract level, most at full-text level.
- Not verified: CircuitKIT docs pages trust/audit-results and trust/scope-limitations returned 404; the CircuitKIT paper's editing and steering result tables were not in the fetched text; MCircKE's baselines and locality numbers (abstract only); REdit's correlation coefficients (figure-only); whether Tiny Aya ROME/MEMIT actually runs end to end (must be smoke-tested).
- Found nothing (searched 2026-11-02): knowledge editing or circuit interventions on any Aya model; any prior project matching this task with controls; a mechanistic locality metric evaluated on non-target circuits.

## Appendix B. Sources
Lexsi: https://github.com/Lexsi-Labs/CircuitKIT · https://circuitkit.lexsi.ai · https://arxiv.org/abs/2607.19317 (CircuitKIT) · https://arxiv.org/abs/2602.04521 (C-ΔΘ) · https://arxiv.org/abs/2605.15138 (Forgetting That Sticks, CAD) · https://github.com/Lexsi-Labs/SafeTune · https://github.com/Lexsi-Labs/AuditKIT · https://arxiv.org/abs/2509.08592 (Interpretability as Alignment) · https://lexsi.ai/resources/research-papers · https://huggingface.co/CohereLabs/tiny-aya-base · https://arxiv.org/abs/2603.11510 (Tiny Aya)

Localization vs intervention: https://arxiv.org/abs/2301.04213 (Hase) · https://arxiv.org/abs/2502.11447 (Wang, Veitch) · https://arxiv.org/abs/2605.06076 (Old Maps) · https://arxiv.org/abs/2410.12949 (Mechanistic Unlearning, ICML 2025) · https://arxiv.org/abs/2603.06923 (REdit) · https://arxiv.org/abs/2606.14388 (Low-Rank Subspace Analysis) · https://arxiv.org/abs/2502.06106 (Circuit-tuning) · https://arxiv.org/abs/2502.11812 (circuit-aware LoRA) · https://arxiv.org/abs/2601.08146 (CT-SFT) · https://arxiv.org/abs/2610.32866 (TopKLoRA) · https://arxiv.org/abs/2503.16356 (CaKE) · https://arxiv.org/abs/2604.05876 (MCircKE) · https://arxiv.org/abs/2603.20795 (Anatomy of an Edit) · https://arxiv.org/abs/2503.01090 (FiNE) · https://arxiv.org/abs/2405.17969 (Knowledge Circuits) · https://arxiv.org/abs/2403.19647 (Sparse Feature Circuits, SHIFT) · https://arxiv.org/abs/2507.16795 (CAFT) · https://arxiv.org/abs/2604.01604 (CRaFT) · https://arxiv.org/abs/2609.22224 (Trait Vectors to Circuits) · https://arxiv.org/abs/2511.07002 (Probe Prompting) · https://arxiv.org/abs/2606.00930 (A Circuit, Not The Circuit) · https://arxiv.org/abs/2510.11328 (Emotion circuits) · https://arxiv.org/abs/2407.08734 (Miller, faithfulness not robust) · https://arxiv.org/abs/2410.13032 (Shi, hypothesis testing)

Side effects and verification: https://arxiv.org/abs/2609.29587 (Anand, hidden failures) · https://arxiv.org/abs/2608.27839 (KLOD) · https://arxiv.org/abs/2601.11042 (REVIVE) · https://arxiv.org/abs/2402.09656 (Butterfly effect) · https://arxiv.org/abs/2605.01167 (COAST) · https://arxiv.org/abs/2603.24543 · https://arxiv.org/abs/2608.08383 · https://arxiv.org/abs/2501.17148 (AxBench) · https://arxiv.org/abs/2601.14004 (Locate, Steer, Improve survey) · https://arxiv.org/abs/2610.00320 (Refusal Localizes, Damage Relocates) · https://arxiv.org/abs/2605.01913 (RefusalGuard)

Multilingual: https://arxiv.org/abs/2406.16416 (LAFN) · https://arxiv.org/abs/2505.20546 (Paths Not Taken) · https://aclanthology.org/2025.findings-acl.827.pdf · https://arxiv.org/abs/2505.14393 (MKE survey) · https://arxiv.org/abs/2502.14645 (X-KDE) · https://arxiv.org/abs/2507.22608 (Language Arithmetics, Aya Expanse) · https://labscommunity.cohere.com/blog/2026/inside-tiny-aya-what-an-sae-sees/
