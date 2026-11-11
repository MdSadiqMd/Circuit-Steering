"""Build every intervention cell for one behaviour and intervention type.

  uv run python interventions/run_grid.py behaviour=b1 intervention=edit [seeds=[0,1,2]]
  uv run python interventions/run_grid.py behaviour=b2 intervention=steer
  uv run python interventions/run_grid.py behaviour=b2 intervention=lora

Per (condition, seed): select components from the seed's discovered circuit
(interventions/select.py;
Taylor scores computed once per behaviour and cached), build the artifact with steer.py / lora.py /
edit.py, write cell.json + delta.safetensors + a manifest. Random gets `random_draws` seeds.
Models: steer -> TL bf16; lora -> HF bf16; edit -> TL float32 (numerics). One process per
intervention type keeps the memory budget (docs/plan.md §0).
Training/edit data: B1 = false facts (English, data/b1/false_facts.jsonl) split into `seeds`
disjoint folds so every seed edits different facts and the union covers the set; B2 = harmful
prompts of the
chosen category (English) with a fixed refusal reply, held-out 30 % kept for the audit.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import pandas as pd
import torch
from aya_circuits.config import load_config
from aya_circuits.manifest import Manifest
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import chat_wrap, load_hf, load_tl
from aya_circuits.nodes import split, top_k
from aya_circuits.registry import resolve_task

from interventions import edit, lora, steer
from interventions.cell import Cell, delta_norm
from interventions.select import select
from interventions.taylor import taylor_scores

ROOT = Path(__file__).resolve().parent.parent
REFUSAL_EN = "I'm sorry, but I can't assist with that."


def circuit_scores(run_root: Path, task: str, seed: int) -> dict[str, float]:
    nodes = json.loads((run_root / "circuits" / task / f"seed{seed}" / "nodes.json").read_text())
    d = dict(zip(nodes["names"], nodes["signed_rowsum"], strict=True))
    d.pop("input", None)
    return d


def joint_nodes(run_root: Path, task: str, disc_seed: int, cond: str, seed: int) -> list[str]:
    """A15: the joint-ranked set whose pillars are reported (aya_circuits.cellset `pillar_set`),
    with random and inverse sets matched to its head and MLP counts by the same absolute scores."""
    d = run_root / "circuits" / task / f"seed{disc_seed}"
    pillar = json.loads((d / "cellset.json").read_text())["pillar_set"]["nodes"]
    kh, km = (len(x) for x in split(pillar))
    if cond == "circuit_joint":
        return pillar
    scores = json.loads((d / "circuit_scores.json").read_text())["node_scores"]
    scores.pop("input", None)
    if cond == "inverse_joint":
        return top_k(scores, kh, km, reverse=False)
    heads, mlps = split(list(scores))
    rng = random.Random(seed)
    return rng.sample(heads, kh) + rng.sample(mlps, km)


def b1_folds(cfg: dict) -> dict[int, list[dict]]:
    rows = [json.loads(line) for line in (ROOT / "data/b1/false_facts.jsonl").open()]
    rng = random.Random(0)
    rng.shuffle(rows)
    seeds = cfg["seeds"]
    return {s: rows[i :: len(seeds)] for i, s in enumerate(seeds)}


def b2_split(cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """(train harmful, held-out harmful, benign) for the chosen category, English."""
    cat = cfg["behaviour"]["b2_category"]
    rows = [json.loads(line) for line in (ROOT / f"data/b2/baseline_{cat}.jsonl").open()]
    harm = sorted((r for r in rows if r["split"] == "harmful"), key=lambda r: r["id"])
    ben = [r for r in rows if r["split"] == "benign"]
    random.Random(0).shuffle(harm)
    n_hold = max(10, int(0.3 * len(harm)))
    return harm[n_hold:], harm[:n_hold], ben


def fact_prompt_en(fact: dict, tok, cfg: dict) -> tuple[str, str, str]:
    """(chat prompt text, subject, target) in English for editing."""
    from data.b1_build import TEMPLATES

    prompt = chat_wrap(
        tok, TEMPLATES[fact["relation"]]["en"].format(s=fact["subject"]["en"]), cfg["chat_format"]
    )
    return prompt, fact["subject"]["en"], fact["object"]["en"]


def main() -> None:
    cfg = load_config()
    beh, interv = cfg["grid"]["behaviour"], cfg["grid"]["intervention"]
    run_root = Path(cfg["out_dir"])
    task = f"{beh}_en"
    top = cfg["discovery"]["top_fraction"]
    conditions = list(cfg["interventions"]["conditions"]) + (
        cfg["interventions"]["editing"].get("extra_conditions", []) if interv == "edit" else []
    )
    mw = MemWatch(cfg["memory_budget_gb"])

    # Taylor scores once per behaviour (TL, loss on the behaviour's discovery data)
    taylor_path = run_root / "circuits" / task / "taylor_scores.json"
    model_tl = None
    if not taylor_path.exists():
        model_tl = load_tl(cfg)
        spec = resolve_task(task, cfg)
        dcfg = {
            "algorithm": "eap-ig",
            "task": spec.name,
            "level": "node",
            "batch_size": 4,
            "data_params": {"num_examples": 32, "seed": 0},
        }
        dl = spec.build_dataloader(model_tl, dcfg, cfg["device"])
        taylor_path.write_text(
            json.dumps(taylor_scores(model_tl, dl, spec.metric_fn(), n_batches=8), indent=1)
        )
    taylor = json.loads(taylor_path.read_text())

    if interv == "lora":
        model, tok = load_hf(cfg)
    else:
        model = model_tl or load_tl(cfg)
        if interv == "edit":
            # fp32 for the edit numerics, but loaded as bf16 first: TL's from_pretrained holds HF +
            # state-dict +
            # module copies during load (3× weights), which at fp32 is 40 GB and swapped the
            # machine.
            model = model.to(torch.float32)
            model.cfg.dtype = torch.float32
        tok = model.tokenizer
    cov_texts = None
    if interv == "edit":
        df = pd.read_parquet(
            ROOT / ".cache/raw/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
        )
        cov_texts = [t.strip() for t in df.text.tolist() if len(t.strip()) > 200][
            : cfg["interventions"]["editing"]["cov_n_samples"]
        ]

    folds = b1_folds(cfg) if beh == "b1" else None
    cap = cfg["grid"].get("max_facts")  # smoke runs: cap facts/prompts per cell
    if folds and cap:
        folds = {k: v[:cap] for k, v in folds.items()}
    if beh == "b2":
        train_h, held_h, benign = b2_split(cfg)
        if cap:
            train_h = train_h[:cap]
        (run_root / "interventions" / "b2_split.json").parent.mkdir(parents=True, exist_ok=True)
        (run_root / "interventions" / "b2_split.json").write_text(
            json.dumps(
                {"train": [r["id"] for r in train_h], "held_out": [r["id"] for r in held_h]},
                indent=1,
            )
        )
        # B1 steering source/target: wrong vs known facts (English); B2: benign vs harmful
    match_size = None
    if interv == "steer":  # conditions are norm-matched to circuit seed 0 at coefficient 1
        c0 = Cell(beh, "steer", "circuit", cfg["seeds"][0], []).dir(run_root) / "cell.json"
        if c0.exists():
            match_size = json.loads(c0.read_text())["meta"]["size_at_1"]
    built = 0
    cap = cfg["grid"].get(
        "cells_per_process"
    )  # LoRA cells leak allocator memory across cells on MPS
    for cond in conditions:
        seeds = (
            list(range(cfg["random_draws"])) if cond in ("random", "random_joint") else cfg["seeds"]
        )
        for seed in seeds:
            disc_seed = seed if seed in cfg["seeds"] else cfg["seeds"][seed % len(cfg["seeds"])]
            scores = circuit_scores(run_root, task, disc_seed)
            if cond == "causal_trace":  # A8: H5's English causal-tracing site (causal_trace.py)
                trace = json.loads((run_root / "circuits" / task / "causal_trace.json").read_text())
                nodes = [f"MLP {trace['layer']}"]
            elif cond.endswith("_joint"):
                nodes = joint_nodes(run_root, task, disc_seed, cond, seed)
            else:
                nodes = select(cond, scores, top, seed=seed, taylor_scores=taylor)
            cell_dir = Cell(beh, interv, cond, seed, nodes).dir(run_root)
            if (cell_dir / "delta.safetensors").exists() and not cfg.get("force"):
                print("cached", cell_dir)
                continue
            if cap and built >= cap:
                print(
                    f"stopping after {built} cells (grid.cells_per_process); rerun to continue",
                    flush=True,
                )
                raise SystemExit(3)
            mf = Manifest(f"grid_{interv}", cfg, seed)
            for rel in (
                ("data/b1/false_facts.jsonl", "data/b1/model_eval_en.jsonl")
                if beh == "b1"
                else (f"data/b2/baseline_{cfg['behaviour']['b2_category']}.jsonl",)
            ):
                mf.add_input(ROOT / rel)
            mf.add_input(run_root / "circuits" / task / f"seed{disc_seed}" / "nodes.json")
            t0 = time.time()
            if interv == "steer":
                if beh == "b1":
                    ev = [
                        json.loads(line) for line in (ROOT / "data/b1/model_eval_en.jsonl").open()
                    ]
                    # A6: the fold whose facts measure this seed's efficacy (A2) is excluded from
                    # the
                    # source prompts, so the vector never sees the facts it is scored on
                    fold_keys = {(f["relation"], f["subject_qid"]) for f in folds[disc_seed]}
                    source = [
                        r["text"]
                        for r in ev
                        if not r["known"] and (r["relation"], r["subject_qid"]) not in fold_keys
                    ][:64]
                    target = [r["text"] for r in ev if r["known"]][:64]
                else:
                    source = [chat_wrap(tok, r["en"], cfg["chat_format"]) for r in benign][:64]
                    target = [
                        chat_wrap(tok, r["en"], cfg["chat_format"])
                        for r in train_h
                        if r["gen"]["en"]["refused"]
                    ][:64]
                st_cfg = cfg["interventions"]["steering"]
                cell, tensors, s1 = steer.build_cell(
                    model,
                    beh,
                    cond,
                    seed,
                    nodes,
                    source,
                    target,
                    run_root,
                    match_size,
                    max_factor=st_cfg["max_norm_match_factor"],
                )
                if cond == "circuit" and seed == cfg["seeds"][0]:
                    # later conditions are norm-matched to the circuit condition at coefficient 1
                    match_size = s1
                # main cell = base coefficient × main_multiplier; sweep cells share the vectors
                # (condition "cond@mult")
                base_coeff = float(tensors["coefficient"])
                tensors["coefficient"] = torch.tensor(base_coeff * st_cfg["main_multiplier"])
                cell.meta["coefficient_multiplier"] = st_cfg["main_multiplier"]
                for mult in st_cfg["coefficients"]:
                    sw = Cell(
                        beh,
                        interv,
                        f"{cond}@{mult}",
                        seed,
                        nodes,
                        meta={**cell.meta, "coefficient_multiplier": mult, "sweep_of": cell.name},
                    )
                    sw.save(run_root, {**tensors, "coefficient": torch.tensor(base_coeff * mult)})
                stats = {"size_at_1": s1, "base_coefficient": base_coeff}
            elif interv == "lora":
                if beh == "b1":
                    pairs = [
                        (fact_prompt_en(f, tok, cfg)[0], f["object"]["en"])
                        for f in folds[disc_seed]
                    ]
                else:
                    pairs = [
                        (chat_wrap(tok, r["en"], cfg["chat_format"]), REFUSAL_EN) for r in train_h
                    ]
                tensors, stats = lora.train(
                    model, tok, pairs, nodes, cond, cfg["interventions"]["lora"], seed
                )
                cell = Cell(beh, interv, cond, seed, nodes, meta=stats)
                cell.save(run_root, tensors)
            else:  # edit
                layers = edit.mlp_layer_for(cond, nodes, scores, seed, model.cfg.n_layers, taylor)
                if beh == "b1":
                    facts = [fact_prompt_en(f, tok, cfg) for f in folds[disc_seed]]
                else:
                    facts = [
                        (
                            chat_wrap(tok, r["en"], cfg["chat_format"]),
                            r["en"].split()[-1].strip(".?!"),
                            REFUSAL_EN,
                        )
                        for r in train_h
                    ]  # key = last prompt word
                cell, tensors, stats = edit.edit_cell(
                    model,
                    beh,
                    cond,
                    seed,
                    nodes,
                    facts,
                    layers,
                    cov_texts,
                    cfg["interventions"]["editing"],
                    run_root,
                )
            norm = delta_norm(tensors, interv)
            cell.meta["delta_norm"] = norm
            if beh == "b1":
                cell.meta["edited_keys"] = [
                    [f["relation"], f["subject_qid"]] for f in folds[disc_seed]
                ]
            else:
                cell.meta["train_ids"] = [r["id"] for r in train_h]
            cell.save(run_root, tensors)
            mw.check()
            mf.add_output(cell_dir / "delta.safetensors")
            mf.write(
                cell_dir / "manifest.json",
                memory=mw.summary(),
                stats=stats,
                wall_s=round(time.time() - t0, 1),
                delta_norm=norm,
            )
            print(
                f"{cell.name}: {time.time() - t0:.0f}s |Δ|={norm:.3g} "
                f"{json.dumps(stats, default=str)[:300]}",
                flush=True,
            )
            built += 1
            import gc

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
