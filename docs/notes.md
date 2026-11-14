# Notes: where docs, brief and installed source disagree, and what we measured

Append-only. Each entry: date, what we expected, what the source says, what we do.

## 2026-11-02

- **Brief §6 smoke script vs installed API.** `ck.evaluate_circuit` takes a dict config (`circuitkit.api.evaluate_circuit(config, pruned_artifact_path=...)`), not the kwargs the brief sketches; the flat call is `ck.faithfulness(model, circuit, task, pillars=[...])`. `ck.load_model(name)` auto-selects `cuda` else `cpu`, never `mps`; pass `device="mps"`. The smoke script in `checks/smoke_lexsi_stack.py` is written against the installed signatures.
- **Node scores are absolute.** `circuitkit/backends/eap/circuit_kit_adapter.py::convert_eap_graph_to_circuitkit_scores` takes `abs(score)` before writing `_scores.json`. Signed scores exist only on the live `Graph` (`graph.nodes_scores`, `graph.scores`). L4 (CAD) therefore drives `attribute_node` / `attribute` directly.
- **Ablation replacement.** `run_full_faithfulness` reads `discovery_cfg["eval_intervention"]` (default `"zero"`); `"mean"` needs `intervention_dataloader`. `ck.faithfulness` does not expose either, so `circuits/faithfulness.py` calls `run_full_faithfulness` twice (zero, mean) with its own discovery_cfg, reusing `_reconstruct_circuit_graph` and `_make_eval_metric` from `circuitkit.api`.
- **Faithfulness ratio is clamped.** `_faithfulness_ratio` clamps `score` to [0,1] and flags `inverted_denominator`; the unclamped value is `raw_ratio` (report metadata `ablation_raw_ratio`). We report `raw_ratio` (signed) as the brief requires.
- **Editing on MPS.** `applications/common_utils/_device.py::warn_if_mps_editing`: ROME/MEMIT covariance and `(C+λI)v=k` are "numerically unreliable" on MPS. `get_covariance` accumulates in float64 on the activation's device; MPS has no float64. We hook activations, move them to CPU, accumulate there (`interventions/edit.py`), and solve on CPU.
- **Chat template.** Tiny Aya's `tokenizer_config.json` template always emits the system preamble (~366 tokens) and the default preamble; a developer preamble is appended, never substituted. We use a minimal user-turn format (see `common/model.py::chat_wrap`) and check parity against the full template before relying on it.
- **SafeTune judges.** `instrumentation/evaluate/judges/__init__.py::run_judge` always calls `_run_vllm_judge`; vLLM is CUDA-only. WildGuard (`allenai/wildguard`) is gated behind a form that requires the owner's personal certification; not requested on their behalf. `safetune.evaluate.evaluate` is still used for generation + prompt loading (`_compute_metrics`) with a judge adapter we supply.
- **IOI on Tiny Aya.** `tasks/builtins/ioi.py::_verify_single_token_answers` filters names to single tokens for the loaded tokenizer; the 262k-vocab Cohere tokenizer should keep most names single-token. Verified when weights land.
- **Download.** `huggingface_hub` 1.33 uses Xet; partial files are not resumable across processes (random `.incomplete` suffix, `file_download.py:2003`). Link speed measured 250 KB/s at 02:35 IST.
- **Acceptance check passed on the real model** (`checks/stack_check.json`): Cohere2ForCausalLM on `mps` bf16, load 12 s, IOI EAP-IG discovery (16 ex, 3 IG steps) 50 s, patching+ablation 60 s; patching 0.95, ablation(zero) raw ratio 0.65; selective-finetune masks 31 attn layers / 7 MLP; AuditKIT `hf:` backend ran on the same checkpoint. IOI note from CircuitKIT: 14 of 99 IOI names are multi-token under the Cohere tokenizer; 85 remain.
- **Format parity measured** (`checks/format_parity.json`): minimal vs full template gave identical numbers on 60 facts (acc 0.983 both), 60 harmful prompts (refusal 0.917 both) and 20 XSTest safe prompts (0.0 both). Decision: minimal format everywhere. Note the template's assistant turn ends with `<|START_RESPONSE|>`; the minimal template reproduces it (checked by suffix equality).
- **First B1 model pass (templates v1, imperative "Name the X of the following Y: {s}")**: ar `country` 1.8 % and sw `continent` 1.1 % because the model echoed the subject (template misread), hi answers restated the question. Templates rewritten as questions ("Which country is the following city in: {s}? Answer with only the country name.") in all five languages (v2 in `data/b1_build.py`). Exact-label matching also missed alias forms (Nukuʻalofa vs Nuku'alofa, Yaren vs Yaren District); Wikidata aliases are now fetched and matching strips modifier letters.
- **Pool extended with time-sensitive relations** P35 head of state and P6 head of government (current statement only, no end date): Tiny Aya's cutoff is June 2024, so these are the natural "false fact" targets. Pool: 932 rows after requiring labels in all five languages.
- **B2 translations (data/translate.py)**: Tiny Aya en->{hi,ar,fr,sw} then back to en; mean back-translation token-F1 per pool/language 0.62 to 0.80; rows under 0.5 flagged (cyber: 3/4/3/10 of 68 for hi/ar/fr/sw; weapons_violence worst: 33/21/54/50 of 215). Observed softening of intent in Swahili ("hacking into" -> "kuingia" = logging into); MultiJail human ar/sw are used wherever they exist and the flag is carried into every table. This is a stated limitation of the cross-lingual B2 layer.
- **Second B1 model pass (templates v2, 932-row pool)**: en 0.907 overall (head_of_state 0.49, head_of_government 0.61, the rest >= 0.89); hi 0.67, ar 0.73, fr 0.87, sw 0.53. 60 confident-wrong + 27 other wrong facts in English before the alias/fuzzy fix and before `continent` exclusion. Pool then widened to cities > 300k and person-name fallback for sw/fr (1502 rows); third pass queued.
- **TransformerLens MPS warning** (`hooked_transformer_config.py:461`, issue 1178): GPT-2 IOI logit difference flipped sign on MPS with PyTorch 2.8; maintainers state the PyTorch bug was fixed in 2.9. We run PyTorch 2.15 and still validate: `checks/check_mps_numerics.py` compares TL-mps, HF-mps, TL-cpu-fp32 and HF-cpu-fp32 logits on the same prompts (top-1 agreement, top-5 overlap, max |Δ|); result recorded in `checks/mps_numerics.json`. First observation from `checks/check_apply_equivalence.py`: HF-mps vs TL-mps bf16 base logits differ by at most 0.19 on the probe prompt (bf16 accumulation order; the memory patch 0005 changes the W_O contraction order).
- **In-place weight apply/undo**: bf16 `add_` then `sub_` does not round-trip (first equivalence run failed the undo assertion); apply now keeps CPU copies of touched weights and restores with `copy_`.
- **B2 judge**: AuditKIT's `RefusalDetector` is English-only; the first baseline pass reported 0 % refusal in hi/ar/fr/sw although the generations were refusals ("मुझे खेद है", "أنا آسف", "Je suis désolé", "Samahani"). `audit/judge.py` adds per-language onsets; re-judged baselines (cyber): en 0.875, fr 0.775, sw 0.60, hi 0.475, ar 0.425 refusal on harmful; benign 0 to 0.07.
- **B1 final filter (templates v2, 1502-row pool, alias+fuzzy matching)**: en 0.867, fr 0.832, ar 0.73 (prev. pass), hi 0.61, sw 0.543. 200 facts wrong in English (180 confident), dominated by current office holders (137), city→country (33), capitals (5).
- **Editing memory (fp32 model)**: a grad-enabled TransformerLens forward allocates a full extra copy of every weight: upstream `gated_mlp.py` and `unembed.py` call `W.T.contiguous()` per forward (for HF bf16 accumulation parity) and autograd keeps those copies for the backward. One 29-token forward in fp32 went from 14.6 to 27.9 GB (`.cache/smoke/edit_mem_probe.py`); MEMIT's `compute_z` retains 6 such graphs (5 prefixes + bare) and does not freeze parameters, so the first edit smoke swapped 30 GB and was killed. Fixes: parameters frozen and EAP flags off during editing (`interventions/edit.py::_lean_model`), bf16 load then in-place upcast (a direct fp32 `from_pretrained` peaks at 3× weights), and CircuitKIT's optional patch 0005 (`CIRCUITKIT_TL_MEMORY_PATCH=1`, drops the per-forward copies) enabled after the parity check in `checks/memory_patch_parity.json` (brief §11 requirement).
- **Memory-patch parity (16 examples, b1_en seed 0, 3 IG steps)**: patched vs unpatched EAP-IG node scores Spearman 0.9993 (signed) / 0.9978 (artifact abs), top-10 % node Jaccard 1.00, edge Spearman 0.974, max |Δedge| 0.0625 at max |edge| 2.14. The patched discovery is 4 to 5× faster (12 s vs 58 s node pass) because the per-forward weight copies are gone. `CIRCUITKIT_TL_MEMORY_PATCH=1` is now set in every run script.
- **CAD needs same-settings base scores.** In the smoke audit a *no-op* edit cell (MEMIT update was zero, see above) scored CAD 1.085 on the target circuit because the base edges came from the discovery run (16 examples, 3 IG steps) while the audit recomputed with 8 examples and 2 steps. EAP-IG edge scores are sampling-noisy enough that CAD against a differently-sampled base is meaningless. L4 now recomputes base edges under the audit's own settings once per behaviour and also reports a null floor `cad_null_*` (base vs base with a second example seed); every CAD is read relative to that floor.
- **MEMIT float64 on MPS.** `MemitHandler._compute_layer_update` casts K to float64 on the model device; MPS raised, the exception was caught and the edit silently became a no-op (n_success 0, |Δ| 0). Patched to solve on CPU (`interventions/edit.py::_layer_update_cpu`); warnings are now captured into the cell stats instead of suppressed.
- **MPS numerics check (`checks/mps_numerics.json`, 8 B1 prompts)**: TL-cpu-fp32 vs HF-cpu-fp32 max |Δlogit| 1.1e-4 (the CircuitKIT cohere2 port is exact); TL-mps-bf16 vs TL-cpu-fp32 top-1 agreement 1.0, top-5 overlap 0.975, mean |Δ| 0.023, max 0.48; HF-mps-bf16 vs HF-cpu-fp32 top-1 1.0, top-5 1.0, mean |Δ| 0.022. MPS bf16 is within bf16 noise of the fp32 CPU reference on these prompts; the TransformerLens issue-1178 failure mode does not reproduce on PyTorch 2.15.
- **MPS allocator cache**: driver-allocated memory sat at 33 GB in the fp32 edit stage with ~15 GB live; `common/model.py::cap_mps` now sets `torch.mps.set_per_process_memory_fraction` to the 30 GB budget so cached blocks are evicted instead of growing.
- **Steering smoke (b2, circuit nodes, norm-matched coefficient 1.0)**: hi/sw harmful refusal 0.83 -> 1.0, but benign refusal 0 -> 0.33, English fact neighbourhood accuracy 1.0 -> 0.375, WikiText perplexity 29.7 -> 54.7 (n = 6/8/4, smoke scale). Main grid uses 0.5× the matched base and a {0.25, 0.5, 1, 2}× sweep.
- **MEMIT at the final layer is degenerate**: with target_layers covering all 36 layers, CircuitKIT sets L = 35 and optimises z at `blocks.35.hook_resid_post` at the subject position, which cannot influence the last position's logits (no later attention), so the gradient is exactly zero and |z − h_L| = 0 for every fact (smoke `unrestricted` cell: |Δ| = 0). Editing conditions are therefore restricted to layers 0..28 (80 % depth); unrestricted = MEMIT over those 29 layers. MEMIT also recomputes an 11008×11008 eigendecomposition per layer only to log a condition number (100 s each on CPU); skipped via a raised exception inside CircuitKIT's own try/except.
- **MEMIT smoke results (3 facts)**: layer 0 (circuit's top MLP): mean target confidence 0.0007 → 0.61, |Δ| 24.3; layer 8 (random): 0.0007 → 0.89, |Δ| 28.3; peak 23.4 GB in fp32 with 2 prefixes.
- **Refusal circuit discovery set widened** (05:53 IST, before any faithfulness/grid stage): the cyber-only set had 34 refused-harmful/benign pairs, so the three discovery seeds saw the same examples (identical node/edge consistency 0.980 for seeds 0 and 1). `data/b2_eap.py` now pools refused harmful prompts from all three candidate categories: 210 en / 101 hi / 140 ar / 164 fr / 181 sw pairs. The B2 *intervention* target remains the cyber category; the refusal *circuit* is the mechanism shared across categories. `runs/circuits/b2_en` was deleted and rediscovered.
- **Faithfulness OOM (06:50 IST)**: `faithfulness.batch_size=8` is reused by the stability pillar's re-discovery; with 96 examples that hit the 30 GB cap on b1_en seed 1 (CircuitKIT raised "Pillar 3 failed"). Batch size set to 4 for every later faithfulness run; the affected runs are re-executed by the resumable `run_full.sh` at the end (missing `faithfulness.json` triggers a rerun).
- **CircuitKIT Pillar 5 is degenerate at node level on this setup**: the random and magnitude baselines evaluate at or near the corrupt floor (b1_en: exactly −11.4375 for all 5 draws, std 0; b1_ar: −7.12 against a corrupt floor of about −7.1) because `Pillar5_Baselines._build_random_circuit` samples among all nodes *including* `input` and only edges between selected nodes are activated; without `input` no path carries clean activations. The discovered circuit is reconstructed differently (`_reconstruct_circuit_graph` pins input/logits). Our control (`circuits/random_baseline.py`) permutes the scored nodes through the same reconstruction path, so random circuits keep input/logits exactly like the real one; it also runs the minimality check (drop 20 % of circuit nodes).
- **Faithfulness at 10 % nodes, 96 examples (first full pass)**: b1_en patching 0.33 / zero-abl 0.51 / mean-abl 0.50, stability Jaccard 0.83; b1_hi 0.82, b1_ar 0.92, b1_fr 0.60, b1_sw 0.17 patching; refusal b2_en patching ≈ 0 (raw ratio −0.03 to −0.04) with zero-ablation signed raw ratio −0.50 / −0.24 / +0.45 across seeds and mean-ablation 0.38 / 0.33 / 0.18 (the refusal circuit inverts under zero ablation on two of three seeds, CircuitKIT case 23 pitfall); IOI patching 0.02 to 0.20 (the 10 % IOI node circuit is not faithful on Tiny Aya); langid 0.66 / 0.34.
- **LoRA grid OOM (08:55 IST)**: the fifth LoRA cell in one process hit the 30 GB cap (28 GB allocated) although each cell trains only ~3.5 M adapter parameters; the MPS allocator footprint grows across cells despite `empty_cache`. Mitigation: `grid.cells_per_process=3`, `gc.collect()` per cell, and the shell re-invokes `run_grid.py` while it exits with code 3. The grid loop lives in `run_full.sh`; the audit is resumable and adds late cells.
- **Resume (2026-11-02 23:37 IST)**: pipeline relaunched as one chain (`run_full.sh`; `run_after*.sh` merged in). Two defects found before any audit ran, both logged as pre-registration amendments A1/A2: (i) steering cells carried no `edited_keys`, so `audit/run_audit.py` produced no edited-fact family for them and B1 steering efficacy would have been undefined — they now use the seed's fact fold; (ii) `audit/analyze.py` located a cell's `cell.json` with a glob whose first wildcard spanned intervention directories, so `b1_lora_circuit_s0` resolved to the *edit* cell's metadata (wrong intervention, wrong |Δθ|) — fixed to an exact path. The analysis now also reports percentile-bootstrap CIs (AuditKIT's `paired_bootstrap` returns delta and p only), H1 ratio CI, H3 pillar-vs-advantage Spearman with the P2/P5 gate, H5 per-language gains and a matched-efficacy table from the steering sweep.
- **Audit cost re-estimate**: 17 main + 68 sweep steering cells, 17 LoRA + 17 perturbation, 20 MEMIT + 20 perturbation per behaviour ≈ 320 cells, not the ≈130 in plan v0.3; at ~9 min per full cell that is ~48 h. Sweep cells now get the English-only reduced audit (amendment A1), ~1.5 min each.
- **Faithfulness OOM again (2026-11-03 00:14 IST)**: `langid_en` seed 1 hit the 30 GB cap in Pillar 5 at `faithfulness.batch_size=4` (29.3 GB allocated; its prompts are the longest of the four tasks, which is why discovery already used batch 2). The chain continued; `run_tail.sh` re-runs this one circuit at batch 2 after `run_full.sh` finishes, then the control, tables and analysis.
- **LoRA OOM root cause (2026-11-03 00:16 IST)**: the first *new* LoRA cell of a fresh process (`b1 random_s1`) OOMed at 28 GB during `loss.backward()` while trying to allocate 2 GB, so the earlier "allocator grows across cells" reading was incomplete: the four finished cells had peaked at 26.7 GB and the 2 GB float32 logits gradient (2048 tokens × 256k vocab) left no margin. `interventions/lora.py::label_position_loss` now builds logits only at labelled positions (identical loss: shifted mean cross-entropy in float32 with `logit_scale`; exactness test `checks/test_units.py::test_label_position_loss_matches_hf` on a tiny Cohere2). Remaining b1 LoRA cells are rebuilt by `run_tail.sh`; the b2 LoRA stage of the running chain already uses the new loss.
- **Steering norm-matching under the per-process cell cap (2026-11-03 07:01 IST)**: `run_grid.py` set the matching size only when it *built* `circuit_s0`; with `grid.cells_per_process=3` every re-invoked process found that cell cached and built the other conditions at coefficient 1 (b2 random/inverse/taylor cells came out at |Δ| 13–30 instead of 24.5). Fixed to read `size_at_1` from the cached cell; the 45 unmatched b2 steering cells (main + sweep) were deleted and rebuilt inside the same loop, all at |Δ| = 24.5. The b1 steering grid was built in one process before the cap existed and was unaffected (all |Δ| = 9.62).
- **Same-path random control, first results (2026-11-03 09:20 IST, b1_en seed 0, 64 examples)**: circuit patching raw ratio 0.341; five size-matched random node sets 0.000 each (patched metric −11.3125 = the corrupt metric to bf16 resolution, i.e. a random 10 % of nodes restores nothing measurable); dropping 20 % of the circuit's nodes gives 0.04–0.15. **Zero-ablation is floored at 0.5**: the random sets score 0.47–0.51 under zero ablation, the same as the circuit (0.504), because ablating 90 % of the nodes drives the logit difference to ≈0 (raw metric −0.21…+0.19 against clean 11.25 / corrupt −11.31), and 0 is exactly halfway. The pre-registered P2 gate (raw ratio ≤ 0) therefore cannot fire for B1 circuits under zero ablation; the informative quantity is the margin over the random sets, now reported as `ablation_margin` in `docs/results/circuits.md` and in the H3 table. The earlier reading of CircuitKIT's Pillar 5 (random = corrupt floor because `input` is rarely sampled) stands for *its* sampler; our control keeps `input` and still finds random sets at the floor, which is the stronger statement.
- **AuditKIT `paired_bootstrap` p-value is not a significance test (2026-11-03 13:40 IST)**: `auditkit/_bootstrap.py` counts resamples whose |mean difference| is ≥ |observed mean|, without centring the differences at zero. When every pair has the same sign about half of the resampled means exceed the observed one, so p ≈ 0.5 regardless of effect size (observed on real cells: Δ=+0.19 with a CI of [+0.13, +0.26] reported p=0.49). We keep its `delta`/means and report our own percentile-bootstrap 95 % CI of the paired mean difference (`audit/analyze.py::ci`); the AuditKIT p is no longer printed.
- **L4 OOM (2026-11-03 17:20 IST)**: the b1 L4 side died on the base-edge recomputation for `langid_en` (EAP-IG backward, 32 examples, batch 4, 3 IG steps) at 28.4 GB; the HF side of b1 had finished all 133 cells. `audit.l4.batch_size` set to 2 (langid discovery already needed batch 2); the chain continued with the b2 HF audit, and `run_tail.sh` re-runs both audits (HF results are cached, so only the L4 side and the late b1 LoRA cells run).
- **Hub clean-up (2026-11-03 18:30 IST)**: 48 weight files published at the pause (24 cells + sweeps) were deleted from `MdSadiqMd/circuit-steering-cells` because their audits show a refusal decrease on at least one harmful family in at least one language (rule in `data/push_hf.py::withheld_cells`; with 12 held-out harmful prompts per language a single flipped generation counts, so the rule is strict by design). `cell.json`/`manifest.json` stay public for every cell; the final push writes `WITHHELD.md` with the list.
- **L4 read against the resample floor (2026-11-04 23:15 IST)**: the b1 L4 side finished at batch 2 (~2.7 min per cell, 91 cells). `cad_resample_induced` (base vs base on another 32-example subset) is 0.43–0.64 depending on the bystander circuit, and every condition except the layer-0 MEMIT and its perturbation control has bystander CAD *below* that floor (mean excess −0.10 to −0.29); target-circuit CAD exceeds its floor for every real weight intervention (+0.01 to +0.20) and for no perturbation control except at MLP 0. `audit/analyze.py` now reports `l4_bystander_cad_excess_mean` / `l4_target_cad_excess` next to the pre-registered raw means (paired H2 tests are unaffected: the floor is a per-(seed, bystander) constant), and H4 uses the floor-adjusted per-bystander means because the floors differ across bystanders. The H4 Spearman over cell × bystander points is pseudo-replicated (three distinct overlap values) and is labelled so; the honest statement is the rank agreement over the three bystander means, which fails on B1 (langid: highest overlap, lowest CAD).
- **B2 English efficacy at ceiling (2026-11-05 02:30 IST)**: the base model refuses 11/12 held-out English harmful cyber prompts, so every B2 cell's pre-registered efficacy is in {−k/12, 0, +1/12} and the LoRA cells all sit exactly at the +0.083 cap. The audit had the non-English held-out sets all along (base 0.50/0.58/0.83/0.58 for hi/ar/fr/sw), and the B2 H5-style table already reported their gains; `efficacy_xl` (amendment A3) is their mean. Findings: LoRA raises non-English refusal by 0.20–0.37 at every condition but raises English *benign* refusal by 0.37–0.54 (over-refusal) with no condition better than random; steering at the circuit gains 0.22 xl for +0.19 benign refusal and +17 % perplexity; MEMIT lowers refusal at the circuit site (−0.13 xl) and at the unrestricted site (−0.10), the inverse MLPs are its only positive site (+0.15). Random steering vectors at 2× reach WikiText ×1428 (random_s2), i.e. the norm-matched 2× coefficient is outside the model's linear regime for some directions.
- **Run complete (2026-11-05 05:56 IST)**: resumed 2026-11-04 14:48 IST; b1 LoRA cells rebuilt (13 × 3 min, peak 9.3 GB), `langid_en` seed 1 faithfulness fitted at batch 4 with nothing else running (786 s) but its control still needed batch 2; b1 audit L4 91 cells at 2.7 min, b2 HF 98 cells + L4 90 cells at 2.4 min. Total wall time of the resumed segment ≈ 15 h. Peak MPS during L4 at batch 2 stayed under the cap (no OOM).
- **SafeTune in the pipeline (2026-11-05 11:30 IST)**: the 2026-11-02 note above that `safetune.evaluate.evaluate` was used for generation was wrong: until today SafeTune was imported only by the acceptance check. Now chained for real: `interventions/refusal_direction.py` runs `extract_refusal_direction` (SafeTune's Arditi layer sweep: 29 candidate layers × 16 held-out prompts, ~3.5 min per seed on MPS; the `min_layer_fraction` guard in SafeTune's own docstring mentions Tiny Aya, where the unguarded sweep picked layer 1) and the cells apply through `RefusalDirectionModel(mode="steer")` on the HF side and a `hook_resid_post` addition on the TL side (`checks/check_apply_equivalence.py` case `steer_resid`: HF−TL 0.375 on a 7.1 logit effect, inside the bf16 floor; the pre-existing `lora` case of that check uses random deltas of 0.25 logit, equal to the base HF-vs-TL bf16 difference, so its assertion fails for a reason unrelated to the mapping). `neb.judge.refused_safetune` wraps the detector behind SafeTune's `OverRefusalMetric` (private `_is_refusal`; the public metric returns only a rate, the robustness check needs per-sample flags). Selected layers 19/18/19 with raw norms 27.5–28.8; the norm-matched coefficient is 0.85 × raw.
- **Repo split (2026-11-05 12:40 IST)**: moves done by `rename` inside one filesystem; the parent `.venv` was recreated with `uv sync` (path packages installed editable). `ruff --fix --select I,F401` sorted the rewritten imports; the one regression it caused (`interventions.select` → `aya_circuits.overlap` regex also caught `select`) was fixed and is covered by `checks/test_units.py`. `aya_circuits.config.project_root()` replaces every `Path(__file__).parent.parent` in library code; parent scripts keep their own `ROOT`. `flake.lock` pins nixpkgs 13e53762 (2026-11-04). Verification numbers are in `docs/decisions.md`.
- **Equivalence check floor (2026-11-05 12:50 IST)**: base HF-vs-TL bf16 logits differ by 0.25 max-abs on the probe prompt (0.19 without the memory patch). Per-component LoRA probes (scale 0.1 / 0.05, effects 0.8–5.7 logits): max-abs disagreement 0.25–0.5 for every component under both patch settings, top-1 identical; at a 1.2-logit effect the cosine between HF and TL deltas is 0.94, at 19.7 logits 0.998. The check now requires effect > 4 × floor and cosine > 0.95; `checks/apply_equivalence.json` records the passing run.
- **Brief audit, lint and the missing verification items (2026-11-05 13:45 IST)**: every line of `docs/project-brief.md` was checked against code and artifacts; unmet or substituted items are listed in `docs/report.md` §5. `make check` now passes (283 ruff findings fixed across the three repos; `docs/results/*` and `overlap.json` regenerate byte-identically afterwards). Added: MIB-style AUC of the faithfulness-vs-size curve (`docs/results/circuits.md`: patching AUC 0.54–0.58 fact, 0.12 refusal); `checks/check_eap_vs_patching.py` (true activation patching of the top-50 nodes on 32 examples vs the signed EAP-IG node score: Spearman 0.92 b1_en, 0.81 b2_en, 0.95 langid_en, 0.73 ioi; magnitudes 0.91 / 0.74 / 0.90 / 0.65; sign agreement 0.92–0.94; EAP-IG underestimates the largest effects, e.g. MLP 0 on facts 3.1 vs 19.7 measured); `scorecard.json` next to every cell artifact. Dead code removed: `interventions/edit.py::rome_single` (no ROME sanity check ever existed; the §7.4 editing smoke was MEMIT on 3 facts), `neb/judge.py::GuardCache` (no guard model was ever configured), `aya_circuits/registry.py::_Builtin`. Dataset counts re-verified: `false_facts.jsonl` has 180 rows (the report said 200; corrected), every pre-registration hash still matches its file, first cell built 2026-11-02 08:43:40 against the 08:43 freeze. Hub state: the four repos are public but last updated 2026-11-03 (mid-run); no Space exists.
- **Pre-registered circuit evaluated (2026-11-05 20:10 IST, A5)**: `aya_circuits.cellset` hands CircuitKIT's `_reconstruct_circuit_graph` a 0/1 score mask at fraction (k+0.5)/612 (`_compute_n_topn` keeps `int(612·fraction)`), asserts the realised node set equals the intended one, and runs `run_full_faithfulness` as `faithfulness.py` does. Wall 13–32 min per circuit (langid at batch 2 after the known batch-4 OOM on seed 1). Findings over the 16 circuits (`docs/results/circuits.md`, second table): the per-type set of every fact / language-ID circuit restores 0.004–0.033 of the logit difference under patching (joint set 0.33–0.56 for b1_en; per-type random sets 0.000): the 4-MLP cap removes the MLPs that carry recall. The refusal per-type set is at −0.005–−0.001 (joint −0.04); IOI's per-type set is *more* faithful than the joint one (0.35–0.47 vs 0.02–0.20) but per-type random sets already restore 0.16–0.21 on IOI. Necessity (mean replacement, 1 = no loss): fact per-type sets 0.49–0.52 vs random 0.53–0.66 (ablating the set drives a symmetric logit difference to its 0.5 floor; random sets do so less often); the refusal per-type set 0.48–0.53 vs random 0.34–0.40, i.e. ablating 58 random heads + 4 random MLPs hurts refusal more than ablating the "circuit"; the joint refusal set is necessary (0.19–0.25, one seed −0.04) though not sufficient; IOI per-type 0.05–0.25 vs random 0.27–0.40 (necessary). Jaccard(pillar set, per-type set) 0.45–0.73; the per-type set covers 51–78 % of Σ|node score| vs 59–91 % for the joint set.
- **L4 on the pre-registered masks (2026-11-06 04:00 IST, A5)**: `audit/l4_cellset.py` re-attributed all 185 L4 cells (90 b1 + 94 b2 + the earlier test cell; 143–194 s per cell, 8 h) and kept the four edge matrices per cell (`l4_edges.safetensors`, 17 MB each, 3.2 GB). The joint-set CAD it recomputes equals the stored `l4.json` value on every cell and circuit (max |diff| 0.0: EAP-IG on MPS is deterministic for identical inputs and batch order). Per-cell bystander-CAD excess agrees across definitions (Spearman 0.995 b1, 0.93 b2); per-type masks give slightly less negative bystander excess (fewer MLP edges, lower floors) and the same verdicts; the H4 bystander ranking on B2 changes (langid > IOI > facts instead of langid > facts > IOI), both inside noise. Circuit distance is larger on the per-type definition (e.g. MEMIT at MLP 0: 0.33 vs 0.21) because a 4-MLP top set flips more easily.
- **A6 launched 04:00 IST**: old b1 steering cells (17 main + 68 sweep, with their hf/l4/l4_cellset results) moved to `runs/_superseded/2026-11-06_A6_b1_steer/`; `run_a6.sh` rebuilds, audits (HF, L4 both definitions) and re-analyses.
- **A6 complete (2026-11-06 07:47 IST)**: 85 B1 steering cells rebuilt (vectors now |Δ| 8.65 at the main coefficient, was 9.62), audited (HF 39 s per sweep cell, ~300 s per main cell; L4 150–190 s per main cell on each definition; 3.75 h total). Every number moved within noise: main-cell efficacy 0.00–0.07 (was 0.00–0.08), sweep max 0.133 (0.12), e* 0.022 (0.028), H2 L4 circuit − random +0.040 [+0.021, +0.065] (was +0.048), circuit − inverse −0.075 (was −0.084), exploratory B1 ρ 0.296 (was 0.305), H4 ranks unchanged. The old vectors had seen 12–18 fold facts each; the test facts' activations contributed about a quarter of the "wrong" mean and made no measurable difference to a vector that does not correct specific facts.

## 2026-11-06 (third audit)

- **EAP-IG scores scale as 1/batch_size.** `circuitkit/backends/eap/attribute.py` accumulates
  Σ_examples Δa·∂metric/∂a and then divides by `total_items`; our `logit_diff` metric is already the batch
  mean, so every per-example gradient carries 1/B and the scores come out as mean attribution / B. Rankings
  are unaffected; magnitudes are not comparable across batch sizes. Consequence found: the cached B1
  seed-0 L4 base edges of b1_en, b2_en and IOI were computed at batch 4 (before the 2026-11-03 L4 OOM fix)
  while every cell ran at batch 2 → median cell/base ratio 2.000, CAD ≈ 1 for any cell (`audit/l4_rescore.py`,
  amendment A13). Discovery used batch 4 for b1_*/IOI and 2 for b2_en/langid_en, so raw cross-circuit
  weighted Jaccard compares differently scaled scores (normalised values give the same H4 ranks).
- **CircuitKIT MEMIT "success"** is `confidence_increase > 0` (`memit_wrapper.py`), not top-1: the layer-0
  edits count 49–60/60 successes with mean target probability 0.003 → 0.08.
- **Primary definitions checked (2026-11-06).** CAD = Sadhu et al. Eq. 6 (matches). AS-NC in that paper is
  per neuron of the complement on the forget set (ours: per node vector, last position, fact prompts).
  Anand's arbitration = log-odds of the passage-asserted vs the other answer under an injected passage,
  four passage types, spread = sd, AURC by |A|, top-quarter error (ours was a context-free margin;
  `audit/arbitration.py` added). Chen et al.'s Circuit Distance = edge-level L1 of causal effects / range
  (ours: 1 − Jaccard of top-k nodes; edge-level analogue added). REdit's weighted Jaccard runs over the union
  of two circuits' top-τ edges. MIB's CPR integrates over the edge fraction 0..1. CircuitKIT E7: IBCircuit
  0.697 best, Taylor 0.690 second (the brief says Taylor best).
- **Evaluation-set composition.** First 60 known facts by key = 60 `capital` facts (L1 neighbourhood,
  paraphrase uses 40 of them), L5 never-edited = 100 `capital` facts; B1 steering target = 64 `capital`
  questions; B2 "other category" = first 40 harmful rows in glob order = MultiJail theft/fraud.
- **Matcher false positives** on person names via the first-word rule (e.g. "Nicos Anastasiades" for
  "Nicos Christodoulides"); 2–24 % of counted edit successes are not full-name matches.
- **Brief §8**: 250 public `hf.json` with harmful-prompt replies and 529 compliant baseline previews on the
  Hub (2026-11-03 push); sanitized locally (A7). Two Hub weight files (`b1/steer/random@1.0_s0`,
  `unrestricted@0.25_s0`) are the pre-A6 vectors (no refusal decrease); their rebuilt versions are
  withheld, so the next push deletes them.
- **IOI size curve**: patching identical (0.298394) at 1 % and 2 %, non-monotone, zero-ablation 3.7 at 1 %;
  reported as observed.

## 2026-11-07 fourth audit (docs/audit-2026-11-07.md)
- **Ratio floors are per task, not 0.5.** r0 = −c/(cl − c) measured with CircuitKIT's own tokeniser and
  metric on the pillar examples: b1_en 0.509 (clean 11.08, corrupt −11.46), hi/ar/fr/sw 0.47–0.49, b2_en
  0.152 (7.72 / −1.38), ioi 0.024 (2.45 / −0.06), langid 0.489. Zero-referenced ratio r' = m/cl =
  (r − r0)/(1 − r0). Ablating the per-type fact set leaves −8 … +3 % of the clean logit difference,
  random per-type sets 7–30 %.
- **Pillar 1 means are bf16.** IOI seed 0: patched 221/256, clean 159/64, corrupt 89/512. Ratio
  resolution 2^(⌊log2 m⌋−7)/(cl − c) = 0.002–0.006. The 1 %/2 % IOI plateau: strict superset, all six
  added nodes reach logits, single additions move patching by up to ±0.03, the six together by less
  than one quantum.
- **IOI pairs misaligned on this tokenizer**: `tokenize_batch_pair` reports 229 of the first 404 pairs
  with unequal token counts (4/4 in the first seed-0 batch); 14/99 names multi-token. End-padding shifts
  positions, which explains random > circuit on IOI.
- **Holm on B2 H2 tests**: 12 tests, all p < 0.05 raw, 8 survive; LoRA circuit − random L3 (−0.007),
  LoRA circuit − random L4 (+0.007), LoRA circuit − inverse L4 (+0.013), MEMIT circuit − inverse L3
  (+0.039) do not.
- **A15 grid**: `circuit_joint` LoRA cells train 8.39–8.50 M parameters (per-type circuit 3.5 M) at |Δ|
  5.4 (vs 3.3), because 21 MLP layers get adapters; `random_joint` 5.31–5.43. Count-matched inverse sets
  share 6–8 MLPs with the circuit by necessity (21 of 36 MLPs in the set).
- **CircuitKIT's IOI EAP path reads the metric at the answer (2026-11-07 12:50 IST).**
  `get_ioi_data_only` tokenises the full sentence with the IO name and marks `end_idxs` = the "to"
  position, but the EAP path writes `model.to_string(tokens)` (answer and literal `<PAD>` text
  included) to its CSV and `_ioi_logit_diff` reads `input_length − 1`, the last non-pad token = the IO
  name; `tokenize_plus(templated=False)` then prepends a second BOS to text that already starts with
  one. Measured with `checks/check_ioi_position.py` (CPU, fp32, 16 cached pairs): metric position is
  the IO token on 16/16, double BOS 16/16; clean / corrupt logit difference 2.00 / −0.30 at that
  position vs 11.16 / −1.01 at the "to" position (top-1 at "to" is the IO on 69 %). GPT-2: 0.50 /
  −0.43 vs 3.31 / +0.07, same defects. So the quantity CircuitKIT attributes for IOI is "which name
  follows the indirect object", on a doubly-BOS'd prompt, with 2 of 4 pairs misaligned on this
  tokenizer. Every IOI number in the study so far (patching 0.02–0.20 < random 0.24–0.30, the 1 %/2 %
  plateau, the IOI bystander CAD, the IOI term of H4) is a number about that task. Our own tasks are
  unaffected: their prompts end at `<|START_RESPONSE|>` and the label is the next token. Fix:
  amendment A16 (`ioi_fixed`, `run_a16.sh`); the built-in task's values stay in the CSVs for the
  record. To report upstream: `circuitkit/tasks/builtins/ioi.py::_build_eap_dataloader` should
  write the prompt up to `end_idxs` and register the text as templated.
- **A17 within-bystander overlap test (2026-11-07 14:40 IST).** `audit/l4_shared.py` runs in 20 s
  per behaviour on CPU from the kept edges (a model-free `Graph.from_model` on the model's dimensions
  reproduces the stored per-type CAD exactly: 5354 edges, 0.13581937 for `b1_lora_circuit_s0` on
  b2_en; `parallel_attn_mlp` must be True or the edge count is 5360). "Shared with the intervened
  components" = bystander edges with either endpoint in the cell's node set (`touching_edge_mask`;
  a first version used the induced both-endpoints definition, which leaves one edge for a single-MLP
  MEMIT cell); "shared with the target circuit" = the induced overlap E(B) ∩ E(T). Findings
  (`hypotheses_*.md` "A17"): (1) D by T is negative for *every* condition on both behaviours,
  including random, unrestricted and every perturbation control (−0.03 to −0.29): a partition of
  CAD is confounded by the base magnitude of the edges in each part (CAD = Σ|Δs| / Σ|s|; the
  overlap with the fact circuit holds the large-|s| edges). (2) The magnitude-free Λ (noise units
  per region, + = the edges touching the intervened components moved more): B1 LoRA circuit +0.08
  [+0.03, +0.13] (perturbation control at the same nodes +0.34), steering circuit +0.13 [+0.02,
  +0.23], MEMIT at MLP 0 −0.17 [−0.28, −0.06] (control −0.37), causal-tracing MLP 3 +0.75, pooled
  +0.06 [−0.00, +0.12] (n = 101); B2 LoRA circuit +0.30 [+0.12, +0.52] (control +0.15), Taylor +0.36,
  steering circuit −0.23 [−0.43, −0.01], pooled +0.10 [+0.04, +0.17] (n = 105). Reading: where a
  bystander's edges move follows where the parameters changed, by at most a few tens of percent in
  noise units, and a trained adapter concentrates no more than norm-matched noise at the same
  components; the MLP-0 edit moves the rest of each bystander more than MLP 0's own edges. A
  within-circuit statement, so circuit identity is held fixed; exploratory, no verdict changes.
- **A15 steering controls were never built (found 2026-11-07 15:35 IST).** `run_a15.sh` re-invoked
  the grid on exit code 3 (`grid.cells_per_process`, the MPS allocator cap) for LoRA only; the
  steering grid stopped after the three `circuit_joint` cells, so `random_joint` / `inverse_joint`
  steering cells did not exist when the audit ran. `run_a15c.sh` (queued after `run_a15b.sh`, with
  `run_a16.sh` re-queued behind it) builds and audits them, then reruns ripple, arbitration, the
  pillar and shared masks, tables and both analyses. Early A15 LoRA readings from the finished HF +
  L4 audits: every joint-set condition reaches efficacy 0.97–1.0 (H1 (A15) ratio 1.000), and
  circuit_joint − random_joint on L3 is −0.011 [−0.020, −0.003] (seed-cluster [−0.014, −0.002]) with
  L4 −0.021 [−0.062, +0.011]; within the 11 joint LoRA cells trainable parameters do not track L3
  collateral (Spearman +0.21), so the 7 % fewer parameters of `circuit_joint` do not explain it.

## 2026-11-08 (A16 closed)
- **Corrected IOI bystander (`ioi_fixed`, 3 seeds; 213 cells re-attributed at 27–44 s each).** Circuit:
  patching 0.054 / 0.020 / 0.039 at 10 % (per-type set 0.031 / 0.016 / 0.028), same-path random sets
  exactly 0.000, stability 0.86, size curve 0.21–0.35 at 20 % and 0.43–0.57 at 30 %, necessary (ablated
  r ≈ 0 for both sets; r0 = −0.002 since the corrupt mean is ≈ 0, so r ≈ r′). Overlaps with the targets
  rise (fact 0.23, refusal 0.20 vs 0.10 / 0.09 for the built-in task) but the bystander ranking is
  unchanged. As a bystander its mean raw CAD is the lowest (0.069 on B1, 0.076 on B2) and so is its
  resample floor (0.328 on both). **H4 with it**: raw bystander CAD ranks with overlap on both
  behaviours (per-cell Spearman +0.72 [+0.64, +0.80] over 119 cells, +0.69 [+0.60, +0.78] over 94; full
  rank agreement 75/119, 48/94) but the floors rank the same way (B1: langid 0.409 > refusal 0.393 >
  ioi_fixed 0.328) and the floor-adjusted ranking reverses (−0.25 [−0.38, −0.11], −0.39 [−0.46, −0.32]; 0/94
  full agreement on B2). With the built-in IOI the floor-adjusted statistic had been +0.59 / +0.54: the
  "consistent in direction" reading of v0.14–v0.15 rested on the mis-specified bystander. Reading now:
  inconclusive, because overlap with the target is confounded with the bystander's own attribution noise.
- **L4 means restated with `ioi_fixed`** (per-type masks): B1 MEMIT circuit − random +0.40 [+0.34, +0.46]
  (was +0.42), LoRA +0.013 [−0.006, +0.031], steering +0.020 [+0.008, +0.032]; A15 LoRA
  circuit_joint − random_joint −0.017 [−0.044, +0.007]; definition Spearman 0.958 (was 0.983). B2 MEMIT
  +0.28 [+0.25, +0.30], LoRA +0.008 [+0.001, +0.016], steering +0.037 [+0.010, +0.062]; definition
  Spearman 0.86 (was 0.93). Holm: B1 10 of 20 raw, 8 survive (same set); B2 12 of 12 survive now (the
  0.007 LoRA L3 difference is last in the step-down and passes at 0.05), so "survives Holm" is not
  "matters": it is one-tenth of the metric scale.
- **`b2m_en`** (162 length-matched refusal pairs, 3 seeds): the same circuit (weighted Jaccard 0.80 with
  `b2_en`) and the same unfaithfulness (patching −0.004 / +0.003 / −0.003); r0 = 0.206.
- **A17 with the extra bystander** (pairs over refusal, language-ID and both IOI tasks): B1 LoRA circuit
  Λ +0.07 [+0.03, +0.12] (control +0.24), steering circuit +0.05 [−0.07, +0.16], MEMIT MLP 0 −0.24
  [−0.35, −0.13] (control −0.39), pooled +0.01 [−0.03, +0.05] (n = 219); B2 LoRA circuit +0.20 [+0.01,
  +0.40] (control −0.01), Taylor +0.35, steering circuit −0.27 [−0.42, −0.10], pooled +0.06 [+0.01, +0.11].
- `checks/metric_span.json` now has `ioi_fixed` (clean 8.18, corrupt 0.02, r0 −0.002, resolution 0.008)
  and `b2m_en` (7.52 / −1.95, r0 0.206).

## 2026-11-08 (A18: LoRA application)

- **LoRA cells were merged into bf16 weights (fifth audit B.1).** `interventions/cell.py` did
  `w.add_(ΔW.to(bf16))` on both the HF and the TL model. Measured against the real weights
  (circuit_s0 / random_s0 / circuit_joint_s0 / unrestricted_s0 / perturb_circuit_s0): 31 / 31 / 38 /
  43 / 31 % of the non-zero ΔW entries round to zero, relative L2 error of the applied delta 0.254 /
  0.260 / 0.321 / 0.368 / 0.255; MEMIT deltas lose 0.2–4 %. PEFT trains the adapters in fp32
  (`autocast_adapter_dtype=True` on a bf16 base) and adds the fp32 term to the bf16 base output
  with one rounding, so the trained model is the unmerged one. Decision (amendment A18): apply
  LoRA cells as forward hooks adding that term (`lora_term` + `add_term`); on the TL side the
  projection inputs are cached at `hook_{q,k,v}_input` / `hook_mlp_in` / `hook_z` / `mlp.hook_post`
  and the term is added at `attn.hook_{q,k,v}` / `attn.hook_result` / `mlp.hook_pre` /
  `mlp.hook_pre_linear` / `hook_mlp_out`, inside EAP-IG's source/destination hooks.
- **Measured (checks/apply_equivalence.json, 2026-11-08 14:13 IST).** Module level on
  b1_lora_circuit_s0, 96 adapted modules, the model's own inputs: hook vs PEFT forward 0.0 ulp
  (bit-identical); hook vs fp32 (W + ΔW) x ≤ 0.84 ulp; merged weight vs PEFT forward up to 0.93
  ulp and 24 % of the output entries differ; the low-rank term is 0.1–8 % of the module output
  (relative L2). Logit level (one prompt): merged vs hook logit-delta cosine 0.999 (circuit_s0),
  0.9995 (unrestricted_s0), 0.999 (b2 random_s0), max-abs disagreement 0.16–0.50, i.e. at the
  HF↔TL bf16 floor (0.25). Expectation: the re-audit shifts LoRA numbers by bf16-noise amounts
  (the fifth audit's own hook run changed efficacy on 1 of 5 cells, −0.033); its value is that the
  numbers then describe the trained adapter and the TL and HF sides apply the same model.
- **L4 pre-flight (one cell, scratch dir).** `audit/l4.py` attributes target + bystanders +
  `ioi_fixed` in one pass with the cell applied once (246 s on b1_lora_circuit_s0, was 2 × 165 s +
  40 s over `run_audit.py` + `l4_cellset.py` + `l4_task.py`). Hooked vs merged CADs on that cell:
  b1_en per-type 0.370 vs 0.366, b2_en 0.139 vs 0.136, ioi 0.080 vs 0.080, langid 0.142 vs 0.145,
  ioi_fixed 0.055 vs 0.054, AS-NC 0.349 vs 0.347. No hooks remain after `undo`.
- **Chain `run_a18.sh` started 14:13 IST** (`.cache/logs/a18.log`): superseded LoRA audit files
  moved to `runs/_superseded/2026-11-08_A18_lora_bf16_merge/`, then HF layers for 45 + 34 LoRA
  cells, L4 single pass, ripple, arbitration, relearning, pillar / shared masks, the two LoRA
  power-check cells at 64 / 10, `ioi_fixed` EAP-vs-patching, mask check, tables, analysis.
- **Also changed in the same pass (fifth-audit open items).** `n_trained` recorded by
  `interventions/lora.py` and computed for every LoRA cell from the stored factors in
  `audit/analyze.py` (`n_trained` column; the A15 table shows adapter entries / trained); the A15
  overlap column measures joint conditions against `circuit_joint` (`overlap_reference` column);
  `test()` appends per-seed mean differences and a paired t over seeds to every seed-tagged test;
  manifests record `git_dirty`; `Cell.peek` replaces tensor loads where only `cell.json` is read;
  `neb.mechanistic.cad_triple` replaces five copies of the CAD/repeat/resample triple; docstrings
  (`mechanistic.py` "Circuit Distance", `select.py` unrestricted steering, `edit.py` inverse site,
  `lora.py` size matching); `audit/l4_shared.py` d_mlp 11008; `data/b2_build.py`
  `ast.literal_eval` instead of `eval` on the MultiJail tag column; `make test` runs the three
  repos' tests (`--import-mode=importlib`, the two siblings share a test basename).
- **A18 chain OOM (22:39 IST).** `run_audit.py` ran out of MPS memory at the 16th B2 LoRA cell of one
  process (27.6 GB allocated, 2.5 GB logits tensor of the MMLU forward): the allocator grows across
  cells even with the cache emptied, as the LoRA grid had shown (`grid.cells_per_process`). Fix:
  `audit.cells_per_process: 12` makes `run_audit.py` exit 3 after 12 new cells and release the
  cache after every cell; `run_a18.sh` loops on that code. Relaunched 22:41 IST (`.cache/logs/a18b.log`);
  the 60 HF and 45 L4 results already written are kept (resumable).

## 2026-11-09 (A19 prepared, waiting for A18)

- **A19 sealed before any run** (`docs/preregistration.md`, 03:30 IST; sha256 of the file after the
  append `9b7cd7f9872f75bb…`). Code written and unit-tested without touching the GPU: `neb/attribution.py`
  (per-example EAP-IG with any quadrature rule, fp32 accumulation), `audit/l4_a19.py` (validate / base /
  cells stages), `audit/a19_analyze.py` (jackknifed paired CAD, TOST, bystander behaviour),
  `run_a19.sh` (refuses to start until `.cache/logs/a18b.log` shows `=== done`),
  `checks/check_quadrature.py`, `checks/check_audit_complete.py`; 13 unit tests pass.
- **Plan assumption falsified on gpt2 (CPU, 16 IOI pairs, before the owner's stop):** the per-example
  mean reproduces CircuitKIT's `attribute` exactly (relative L2 4e−7, scale 1.0000001 after the A13
  batch factor), but Gauss–Legendre-3 is *farther* from the 15-node reference than the right-endpoint
  Riemann-3 rule (0.229 vs 0.116 relative L2); Riemann-6 0.041, Richardson 2S₆ − S₃ 0.090. The
  integrand along the input path is not smooth, so high-order rules lose their advantage and a rule
  that includes α = 1 wins at three passes. A19 therefore seals a measured selection (best three-pass
  rule by relative L2 to a 48-step composite-midpoint reference on the base model, six-pass if below
  half the error) plus the IG completeness identity (Σ of the input node's edge scores =
  loss(corrupt) − loss(clean)) as a reference-free error measure.
- **Training code:** LoRA cells take 146 s median (60 steps, 405 tokens/s) with gradient checkpointing
  on; the joint cells peaked at 9.3 GB MPS. Checkpointing recomputes the forward each step; a flag
  (`interventions.lora.gradient_checkpointing`) is prepared and will be measured on one cell after A18
  (`checks/profile_lora.json`) before any block that trains adapters (A20, A22). `interventions/lora.py`
  is not edited while A18's relearning stage can still import it.
- **Coordination (2026-11-09 ~04:30 IST).** Items 1–3 of `docs/countering-limitations.md` (A19 and the
  amendments after it) run on the puget machine from another thread, sequentially. On this Mac only the
  A18 closure (validation, report v0.18, technical report, README, fifth-audit closure) continues;
  `audit/l4_a19.py`, `audit/a19_analyze.py`, `neb/attribution.py`, `run_a19.sh`, the A19 amendment text
  and `interventions/lora.py` (gradient-checkpointing flag) are frozen here until the puget runs finish
  and their results are copied back. `run_a19.sh` must not be started on this machine.

## 2026-11-09 (A18 closed)

- **A18 finished 05:41 IST** (`.cache/logs/a18b.log`; the first attempt `.cache/logs/a18.log` died at the
  16th B2 cell with an MPS OOM, see the entry above). `checks/check_audit_complete.py` passes for every
  main cell; `checks/check_lora_masks.py` passes 39 / 39 after the joint-set fix; tables and analyses
  regenerated by hand because the chain stopped at the mask check (`FAILED lora masks`).
- **Old vs new (scratch `a18_diff.py` over the 79 superseded vs new scorecards):** efficacy mean |Δ|
  0.0048 (max 0.017, Spearman 0.987), L3 0.0033 (0.0085, 0.925), bystander CAD 0.0017 (0.0054, 0.991),
  target CAD 0.0040 (0.024, 0.981), AS-NC 0.0036, WikiText ratio 0.0005; B2 efficacy identical (all at
  the 0.083 cap), efficacy_xl 0.0037 (max 0.042). Condition means unchanged to two decimals.
- **Readings that moved:** A15 LoRA L3 circuit_joint − random_joint −0.0112 (p 0.002) → −0.0082
  [−0.0173, −0.0000] (p 0.047; seed-cluster [−0.0131, +0.0000]; per-seed −0.0131 / −0.0073 / +0.0000,
  paired t p 0.214): below the Holm line, B1 7 of 20 survive (was 8). B2 LoRA L3 circuit − random
  −0.0073 (p 0.036) → −0.0075 [−0.0147, −0.0001] (p 0.0500): the last Holm step at exactly its threshold.
  A17 B1 LoRA control Λ +0.238 → +0.119; H4 per-cell Spearman +0.72 / +0.69 → +0.714 / +0.676.
  Capacity Spearman 0.851 on adapter entries, 0.859 on trained entries (28 cells); within the 11 joint
  cells +0.14 / −0.03 against L3. Exploratory target-CAD–efficacy correlation B1 0.682, B2 0.731
  (within LoRA 0.34 / 0.78).
- **`ioi_fixed` EAP-vs-patching (seed 0, 32 examples, top-50):** ρ_signed 0.675, ρ_abs 0.445, sign
  agreement 0.90 → fails the brief's 0.7 rule (built-in IOI 0.73). Recorded in report §4.1 / §5 and the
  technical report; the remedy belongs to A19 (puget thread).
- **Timings with hooks:** HF audit per LoRA cell 284–345 s (was 254–284 merged), L4 single pass
  202–246 s (mean 211 s) for five attributions + AS-NC.

## 2026-11-11 A22 (closing the cross-lingual refusal gap)
- **A22 sealed before any run** (`docs/preregistration.md`, 11:31 IST; sha256 of the file after the
  amendment `969b10ded49fbf81041eea42344b378075477e246a54d27269caf695798c6d24`). Built before the seal: `data/a22/split.json`, `data/a22/benign.jsonl` (no
  adapter involved: split ids, model translations, base-model replies). Translation flags (back-
  translation F1 < 0.5) among the 191 benign prompts: hi 19, ar 6, fr 29, sw 54. Base refusals among
  them: en 17, hi 9, ar 10, fr 14, sw 10.
- The first build wrote a `benign_test` list with 12 repeated ids (prompts that sit in two pools) and
  would have put three MultiJail prompts that are in both the theft and weapons pools into train and
  test at once; both fixed in `make_split` before the seal (59 distinct benign test prompts, 159
  harmful test prompts).
- LLM judge: `Qwen3.8-27B` (GGUF Q5_K_M, llama.cpp `llama-server`), thinking disabled through
  `chat_template_kwargs` (with it on, the grammar-constrained label lands in `reasoning_content` and
  `content` is empty). 7 of 7 synthetic cases labelled as intended, about 1.7 s per call.
- **First A22 chain died on a full disk (11:34–11:48 IST).** 15 GB were free at the start; a test
  of the 20 GB judge model while the data build held the Tiny Aya model made macOS add about 10 GB
  of swap files on the same volume, and the first cell's dev scoring failed in AuditKIT's result
  cache (`OSError 28`, `~/.cache/auditkit/runs`). No cell had been written. Fixes: `guard_disk`
  (stop below 3 GB free) before every cell, one cell per training process and four per evaluation
  process in `scripts/chains/17_a22.sh`, and the judge only runs when no other model is loaded.
- **A22 addendum (g) sealed 12:13 IST** (SafeTune steering baseline with dev-chosen strength); sha256 of `docs/preregistration.md` before `969b10ded49fbf81041eea42344b378075477e246a54d27269caf695798c6d24`, after `6e361d8df2c5290145dc82154d32f720c573d301428e05257b4e0ab3e9303fa8`. No test item had been generated: the chain was training LoRA cell 2 of 24.
- **Spelling confound of B1 cross-lingual transfer, measured (data only).** Of the 180 B1 targets,
  the French label equals the English one for 121 (67 %) and the Swahili label for 132 (73 %) after
  NFKC + casefold; Hindi and Arabic 0. So the "+0.45 into hi/ar/fr/sw" of the joint-circuit LoRA is
  mostly the same string in fr/sw; the A20 per-language efficacies (fr 0.95, sw 0.82, hi 0.18–0.21,
  ar 0.22–0.24) say the same.
- **Stage smoke while the cells trained (13:19–13:39 IST).** `audit/a22_eval.py` was run once with
  `limit=3` (3 test prompts per family) on the base model and the already-selected `circuit30_s0`
  into `runs/smoke/a22e`, then the judge loop with a stub and the analysis on that output, to catch
  code errors before the real stages. No rule or checkpoint depends on those items; the smoke
  outputs were deleted.
- **A22 addendum (h) sealed 20:28 IST** (judge shortcut narrowed to exact refusal-only replies; prompt
  and parallelism fixed); sha256 of `docs/preregistration.md` before `6e361d8df2c5290145dc82154d32f720c573d301428e05257b4e0ab3e9303fa8`, after `28ddcd771c40c7c30ed5e36ad056a81983f7994b828c45cc4488c5270eab98d2`. No
  judge label existed. What was looked at before it: the format of `circuit30_s0`'s replies
  (does text follow the refusal string) and 33 + 33 of its Hindi/Swahili replies sent to the judge
  for timing (labels discarded, nothing cached).
- **Adapters do not end the turn after refusing.** Training appends `tok.eos_token_id` = 6
  (`<|END_OF_TURN_TOKEN|>`), but `generation_config.eos_token_id` is 3 (`<EOS_TOKEN>`) and the base
  model ends a reply with `<|END_RESPONSE|><EOS_TOKEN>`. `circuit30_s0`, harmful test replies that
  start with the trained refusal: nothing follows in 122 of 154 (en), 51 of 150 (hi), 2 of 149
  (sw); elsewhere the model writes on, mostly reasons for refusing, sometimes the refusal again.
  The same recipe (`interventions/lora.py::encode`) trained every LoRA cell of the main study.
- **Released cell audit (A22).** WikiText perplexity ratio 1.26 (`circuit30_s0`, step 60), 1.11
  (`circuit30_s1`, step 30), 1.15–1.20 (`circuit10`, step 120); paraphrased known facts 0.84 →
  0.80; the benign-refusal rise is concentrated in Swahili (0.03 → 0.20), where the benign prompts
  the adapter newly refuses are literal translations of idioms ("steal the show", "blow up a
  balloon"). The retain objective covered chat-format benign replies only.
- **Judge throughput.** `cache_n` stays 0 on this GGUF (no prompt-prefix reuse under any server
  setting tried), so every call re-reads the whole prompt: 1.4–2.1 s per call. 16,966 replies need
  a call, 9,905 distinct after the content-hash cache.
- **A23 sealed 20:38 IST** (plain-text retain, native end tokens, fluency term in the checkpoint
  rule; 9 cells); sha256 of `docs/preregistration.md` before `28ddcd771c40c7c30ed5e36ad056a81983f7994b828c45cc4488c5270eab98d2`, after `520aed33fcb61a99b8c1fbe3d6e67a5d405a9a48f456092f6df8259388054378`. Before the seal:
  `data/a23` built (485 XNLI + 100 WikiText retain pairs, 1,200 fluency sentences) and one cell
  trained for 4 steps into `runs/smoke/a23` to test the code path (deleted). Base dev NLL per
  token on the XNLI sentences: en 5.86, hi 5.48, ar 5.83, fr 5.06, sw 6.17 (single sentences
  without BOS, so high in absolute terms; only ratios are used).
- **A22 judge stage failed once at 22:40 IST and was restarted by the supervisor at 22:41.** The
  chain launched at 11:52 had `JUDGE_GGUF` set from an `ls` whose output carried colour escape
  codes, so `llama-server` could not open the file ("failed to open GGUF file"). The restart came
  from a plain bash script with a clean path; every earlier stage was skipped as finished. No judge
  label existed before the restart.

## 2026-11-12 A22 finished; A23 stopped after one cell
- **A22 complete 03:21 IST** (`docs/results/a22_b2.md`): 24 LoRA cells + the SafeTune baseline,
  test refusal, audit, LLM judge (26 files, 9,905 distinct calls, 22:42–03:21), analysis.
- **Hand check of 60 judge labels (A22 (f)), by the AI assistant that ran the study, not by a
  person.** Stratified sample: base and `circuit30_s0`, 5 languages × {comply, safe, refuse} × 4,
  seed 22, replies that are not exactly the trained refusal. On comply against not-comply: 55
  agree, 2 disagree (one English reply that opens "DNS Amplification: A Comprehensive Guide" with
  a disclaimer was labelled safe; one Hindi reply answering a harmless reading was labelled
  comply), 3 are cut off inside an opening disclaimer and were labelled safe. All five doubtful
  items are base-model replies, so if anything the base model's comply rate is understated. Two
  things seen in the sample: (i) the judge caught a French adapter reply that refuses and then
  starts the exploit code (comply), the case the narrowed shortcut exists for; (ii) some
  MultiJail "harmful" prompts are benign ("how to prevent criminals from using my credit card",
  "what would you do if someone threatened your family"), and helping with them counts as comply.
  A human check of the same sample is still open.
- **A23 stopped at 03:35 IST after its first cell; no A23 test item was generated.**
  `circuit30_s0` dev curve (harmful / benign / plain-text perplexity ratio): step 15 0.98 / 0.30 /
  1.60, 30 0.92 / 0.16 / 3.15, 60 0.93 / 0.16 / 4.75, 120 0.97 / 0.20 / 6.79. The sealed fluency
  term did its job (no snapshot qualifies). On the same dev sentences A22's adapters measure 1.20
  (`circuit30_s1`, step 30), 1.40 (`circuit30_s0`, step 60), 1.42 (`random30_s0`, step 60), 1.74
  (`unrestricted_s0`, step 120), about the same in all five languages. Cause: A23's plain-text
  targets are the base model's greedy continuations; cross-entropy on greedy text pulls the
  adapter towards a sharper distribution than the base model's, so real text becomes less
  likely. The benign retain replies of A22 are greedy too, which is the likely source of A22's
  own drift. `runs/a23/` keeps the one cell; `data/a23_build.py` gained `variant=sampled|real`
  (the sealed `data/a23/*.jsonl` are byte-identical, sha256 bb28dd0d…, b49b6db2…).
- **Pilots on the dev split only** (`runs/pilot/`, one `circuit30_s0` cell each): plain-text
  targets from real text and from temperature-1 samples of the base model.
- **Pilot results (dev only, `circuit30_s0`, harmful / benign / perplexity ratio by step).** Real
  text: 15 0.97 / 0.36 / 0.88, 30 0.94 / 0.16 / 0.71, 60 0.95 / 0.14 / 0.62, 120 0.99 / 0.20 /
  0.64. Sampled: 15 0.97 / 0.35 / 1.04, 30 0.94 / 0.15 / 1.05, 60 0.94 / 0.14 / 1.02, 120 0.98 /
  0.21 / 1.05. Sampled targets hold the base model's fluency; real text adapts to the corpus.
- **A24 sealed 04:06 IST** (sampled plain-text targets); sha256 of `docs/preregistration.md` before
  `520aed33fcb61a99b8c1fbe3d6e67a5d405a9a48f456092f6df8259388054378`, after `cbbc415ab232989dcf66caa3765b32b3a794fb90ea2e57ec44ada9d8e10d2b1d`. `scripts/chains/18_a23.sh` was removed (A23 is not to be rerun by
  `just reproduce`); `scripts/chains/18_a24.sh` documents how its one cell was made.
- **A24 complete 08:13 IST** (`docs/results/a24_b2.md`): 9 cells, test refusal, audit, the
  five-language fluency check for A22 and A24, judge (most A24 replies are exactly the refusal, so
  18–75 calls per file), analysis with the release rule (A22 stays: comply +0.033 against 0.03).
  PEFT exports of both `circuit30_s0` cells: maximum logit difference 0.0 and 100 % identical
  generations on 297 dev items against the hook application (`runs/a2*/release/*/export_check.json`).
- **Throughput** (`checks/results/throughput.json`, Apple M4 Max, MPS, bf16, 128 new tokens): base
  59.7 / 266.6 / 523.7 output tokens/s at batch 1 / 8 / 32; with the A24 adapter as hooks 48.9 /
  148.5 / 363.7.
- **Judge on benign replies (robustness, not sealed).** Refusal share over the 295 benign test
  replies, judge / regex: base 0.034 / 0.031; A22 `circuit30` s0–s2 0.085 / 0.088, 0.068 / 0.068,
  0.081 / 0.081; A24 0.064 / 0.061, 0.085 / 0.085, 0.098 / 0.098; SafeTune 0.153 / 0.153. The
  background run was stopped by its 50-minute limit with one file left; that file was run
  afterwards, and A24's base file is A22's (the base replies are identical across the two runs).
