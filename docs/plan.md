# Execution plan (v0.1, 2026-11-02)

Companion to `project-brief.md`. The brief says *what* and *why*; this file says *how, on this machine, in what order*, and is revised (versioned, append-only changelog at the bottom) as reality intrudes.

## 0. Environment facts that shape every choice

| Fact | Consequence |
|---|---|
| Apple M4 Max, 48 GB unified memory, no CUDA. Budget granted: 30 GB. | Everything runs on `mps`. `torch.float64` is unsupported on MPS, so ROME/MEMIT covariance and linear solves run on CPU in float64 (`CircuitKnowledgeEditor` warns that MPS editing is numerically unreliable; we route the sensitive ops to CPU, see `interventions/edit.py`). |
| TransformerLens 3.8.0 + CircuitKIT cohere2 port; `ck.load_model` defaults to `cpu` when CUDA is absent. | Always pass `device="mps"` explicitly. |
| Tiny Aya 3.35B bf16 = 6.7 GB weights. TL `use_attn_result` materialises per-head results; EAP-IG node attribution allocates a `[batch, pos, 613, 2048]` bf16 buffer. | batch 4 × 64 tokens ≈ 0.6 GB buffer; batch 2 × 450 tokens ≈ 2.3 GB. Both fit. Discovery on raw-length prompts is 5 to 10× cheaper than with the 366-token default system preamble. |
| Network at build time: ~250 KB/s. Weights take hours. | All code is developed and smoke-tested on `gpt2` (CPU/MPS) first; Tiny Aya numbers only once weights land. gpt2 numbers are never reported. |
| Tiny Aya chat template always injects a ~366-token system + default preamble. | We define one **minimal chat format** `<BOS><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>` and use it identically for discovery, intervention and every audit layer (CircuitKIT's own rule: discovery and downstream must share the prompt format). The first thing we measure on the real model is whether refusal and fact accuracy under this format match the full template (`checks/format_parity.py`); if they do not, we switch to the full template and pay the cost. |
| CircuitKIT stores node scores as absolute values in `_scores.json`. | Cross-circuit CAD needs signed edge scores, so L4 calls the EAP backend (`Graph.from_model` + `attribute`/`attribute_node`) directly and keeps signed tensors. |
| SafeTune judges run through vLLM (CUDA only) and WildGuard is gated behind a form the owner must sign. | L2 refusal scoring uses AuditKIT's `RefusalDetector` (regex) as the always-on judge plus a cached HF-transformers run of a guard model when one is available; judge disagreement is reported, not hidden. |

## 1. Repository layout (v0.8: three sibling repos, see README "Repository layout"; the stage layout below is kept inside them)

```
../tiny-aya-circuits/aya_circuits/   (was common/ + circuits/) config, model + chat format, memory watcher, manifests, Hub;
                                     EAP-IG discovery (3 seeds, signed node+edge scores), faithfulness (zero+mean, curve + AUC), overlap, same-path random + minimality controls, circuits table
../nothing-else-broke/neb/           (was audit/{behavioural,judge,mechanistic}.py) L1..L5 metric families, refusal judges, cross-circuit CAD / AS-NC / circuit distance
data/          B1 facts (Wikidata, 5 languages), B2 refusal category + near-miss, bystander tasks, model-based filtering
interventions/ component selection (circuit / random / inverse / taylor / unrestricted), steering, restricted LoRA, editing, grid runner, SafeTune baseline
audit/         run_audit.py (all five layers per cell), analyze.py (H1..H5, per-cell scorecard.json)
checks/        acceptance smoke, format parity, unit checks
configs/       YAML configs (one base + per-stage)
docs/          brief, this plan, preregistration (frozen), notes (API discrepancies), report
```

## 2. Order of work and gates

1. Environment + smoke on gpt2 (done: discovery 5 s, patching 0.97 on IOI).
2. `common/` + `checks/smoke_lexsi_stack.py` (runs when weights land; JSON committed).
3. Data (no weights needed): Wikidata triples with labels in en/hi/ar/fr/sw; harmful category + near-miss prompts; bystander tasks. Push to HF.
4. Model-dependent data filtering: which facts Tiny Aya gets wrong (B1 = false-fact correction set), baseline refusal per category and language (B2 category choice), format parity.
5. Circuits: discover + evaluate every target and bystander circuit, 3 seeds, pillars patching/ablation(zero,mean)/stability/baselines, faithfulness curve at 1,2,5,10,20,30 %, EAP-IG vs activation patching Spearman on top-50, overlap matrix.
6. Freeze `docs/preregistration.md` with the dataset hashes.
7. Grid: interventions × conditions × seeds; every artifact saved with condition, seed, component list, Δθ norm.
8. Audit L1..L5 per cell; scorecards; H1..H5 analysis with bootstrap CIs.
9. Report, HF artifacts, Space.

## 3. Compute budget (measured on the smoke runs, patched TransformerLens, bf16, MPS)

| Stage | Measured | Full-run setting | Projected |
|---|---|---|---|
| EAP-IG node + edge pass | 12 s + 14 s at 16 ex / 3 IG steps (b1_en) | 64 ex / 5 IG steps | ~3 min per circuit-seed, 16 runs ≈ 1 h |
| Faithfulness pillars (patching, ablation×2, baselines, stability) | 6.5 min at n=16 unpatched | n=96 patched | ~15 min; curve (12 runs at n=48) ~15 min; 16 runs ≈ 6 h |
| Steering cell | 20 s | same | 2 × 17 cells < 15 min |
| LoRA cell | 3 s for 4 steps (280–390 tok/s) | 60 steps | ~1 min; 34 cells < 1 h |
| MEMIT cell (3 facts, 1 layer, fp32) | 100 s | 60 facts | ~20 min; 36-layer unrestricted ≈ 40 min + one-off covariances; ≈ 10 h for both behaviours |
| Audit HF side per cell | 52–60 s at smoke n | configs/base.yaml `audit` | ~7 min |
| Audit L4 per cell (4 edge passes) | 32 s at 8 ex / 2 steps | 32 ex / 3 steps | ~2 min |
| Whole grid audit | | ≈ 130 cells incl. sweeps and controls | ≈ 20 h |

Peak memory: discovery 18–25 GB (allocator cache included), editing 33 GB before the allocator cap, now capped at the 30 GB budget.

## 3b. Compute budget (original estimates)

Per EAP-IG node discovery on Tiny Aya, 64 examples, batch 4, 5 IG steps, ~60-token prompts: 16 batches × 17 forward-equivalents. Measured on gpt2/MPS: trivial. Tiny Aya estimate: 1 to 3 min. Faithfulness with stability (3 re-discoveries) and baselines (5 random draws): 10 to 20 min per circuit. Cross-circuit CAD per cell = one EAP pass per bystander circuit (3 to 4) ≈ 5 to 10 min. Grid of ~100 cells ⇒ 15 to 25 h of audit compute plus generation-based layers. Mitigation: the grid runner is resumable and caches every attribution tensor; n_examples for the audit layers scale with remaining time and are pre-registered as minimums.

## 4. Changelog
- v0.8 (2026-11-05, 12:50 IST): tree split into three sibling repos (`circuit-steering`, `tiny-aya-circuits`, `nothing-else-broke`) with path dependencies and a nix flake; outputs regenerated byte-identically and one cell re-audited identically after the split; equivalence check criterion fixed (cosine, effects above the bf16 floor).
- v0.7 (2026-11-05, 10:40 IST): hackathon rule "≥ 2 track repos chained, 3 preferred" audited against the code: SafeTune was only in the acceptance check, so it is now in the pipeline as the B2 refusal-direction baseline (`safetune_rd`, A4) and as a second refusal judge on every stored generation; README documents the CircuitKIT → SafeTune → AuditKIT chain per stage.
- v0.6 (2026-11-05, 06:00 IST): run completed: 16 circuits with faithfulness and same-path controls, 318 audited cells on all five layers (L4 at batch 2), analyses with floor-adjusted CAD and the A3 multilingual B2 efficacy; report v0.6 with §4.1–4.4 and §5; Hub publication and the Space left for an explicit instruction.
- v0.1 (2026-11-02): initial plan after reading the installed CircuitKIT 0.2.0 / AuditKIT 1.1.3 / SafeTune 0.1.10 source.
- v0.5 (2026-11-03, 22:30 IST): resumed 2026-11-02 23:37 IST as one chain; all grids built except 13 b1 LoRA cells; perturbation and same-path circuit controls done; b1 behavioural audit complete, b2 at 61/159, L4 pending (batch 2); report v0.2 with B1 results; paused again at the owner's request, state in `RESUME.md`.
- v0.4 (2026-11-02, 09:50 IST): full run executed through circuits (16), freeze (08:43), steering grid (17 cells + sweeps), 4 LoRA cells and 3 MEMIT circuit cells (60 facts each), then paused at the owner's request; state backed up (tarball + Hub); `RESUME.md` lists exactly how to continue.
- v0.3 (2026-11-02, 05:05 IST): end-to-end smoke of every stage on the real model passed after fixing: exact weight undo, AuditKIT Prediction fields, MEMIT float64-on-MPS no-op (two patches), fp32 editing memory (bf16 load + upcast, frozen params, EAP flags off, TL memory patch after parity check), CAD base recomputed under audit settings with repeat/resample floors, multilingual refusal judge. MPS numerics validated against CPU fp32. Master script `run_full.sh` written; pre-registration freezes automatically after the circuit stage.
- v0.2 (2026-11-02, 03:30 IST): weights downloaded; acceptance check passed on Tiny Aya/MPS (discovery 50 s on 16 IOI examples); minimal chat format adopted after a parity check; B1 pool built from Wikidata (1502 rows, 7 relations, aliases) and B2 pools (3 candidate categories, human ar/sw where available, model translations elsewhere); all stage modules written (`circuits/`, `interventions/`, `audit/`); end-to-end smoke chain queued. Open: B2 category choice (needs `data/b2/baseline_summary.json`), B1 false-fact count after the third filter pass, measured per-stage timings.
