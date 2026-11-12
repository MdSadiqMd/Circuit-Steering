"""Amendment A20 (plan item 1): B1 steering and LoRA cells on the certified circuit and controls.

  uv run python interventions/run_a20.py out_dir=runs/a20 grid.intervention=steer|lora

Conditions per discovery seed d (node sets from <out_dir>/circuits/b1_en/seed<d>/certified.json and
the seed's CircuitKIT absolute node scores, `circuit.pt`):
  cert            the certified set (aya_circuits.certify)
  cert_random     per-type random: the certified head and MLP counts drawn uniformly (the
                  pre-registered random rule), 2 draws per seed (secondary)
  cert_strat      entry-matched random: in every layer as many heads as the certified set has there,
                  drawn uniformly within the layer, and the MLP count drawn uniformly over layers;
                  LoRA adapters then touch the same attention layers, so `n_trained` matches up to
                  the GQA groups the heads fall in (lora.py), 5 draws per seed (primary)
  cert_inverse    the bottom per-type counts by the same absolute scores
  unrestricted    every node (rebuilt here so the whole block runs on one device)
Random draws carry cell seeds d + 3j (j = 0, 1, ...), so `disc_seed_of` maps them back to d. All
else is the pre-registered B1 recipe of interventions/run_grid.py: the seed's false-fact fold, LoRA
rank 8 / 60 steps / lr 2e-4, steering by difference of means at the last position with sources that
exclude the fold (A6), norm-matched to the main study's `circuit` seed-0 size at coefficient 1 and
applied at 0.5x (as A15); no sweep cells.
"""

from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.manifest import Manifest
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_hf, load_tl
from aya_circuits.nodes import split, top_k

from interventions import lora, steer
from interventions.cell import Cell, delta_norm
from interventions.run_grid import b1_folds, fact_prompt_en

ROOT = Path(__file__).resolve().parent.parent
DRAWS = {"cert_strat": 5, "cert_random": 2}  # per discovery seed
MAIN_RUNS = ROOT / "runs"  # the main study root: only the steering norm target is read from it


def strat_sample(nodes: list[str], all_nodes: list[str], rng: random.Random) -> list[str]:
    """Same head count per layer as `nodes`, heads uniform within each layer; MLPs count-matched."""
    heads, mlps = split(nodes)
    all_heads, all_mlps = split(all_nodes)
    per_layer: dict[int, int] = {}
    for h in heads:
        per_layer[int(h[1:].split(".")[0])] = per_layer.get(int(h[1:].split(".")[0]), 0) + 1
    out = []
    for layer in sorted(per_layer):
        pool = [h for h in all_heads if int(h[1:].split(".")[0]) == layer]
        out += rng.sample(pool, per_layer[layer])
    return out + rng.sample(all_mlps, len(mlps))


def condition_nodes(cond: str, seed: int, cert: list[str], scores: dict[str, float]) -> list[str]:
    all_nodes = list(scores)
    kh, km = (len(x) for x in split(cert))
    if cond == "cert":
        return cert
    if cond == "cert_inverse":
        return top_k(scores, kh, km, reverse=False)
    if cond == "unrestricted":
        return all_nodes
    if cond == "cert_random":
        heads, mlps = split(all_nodes)
        rng = random.Random(seed)
        return rng.sample(heads, kh) + rng.sample(mlps, km)
    if cond == "cert_strat":
        return strat_sample(cert, all_nodes, random.Random(10_000 + seed))
    raise ValueError(cond)


def plan(cfg: dict) -> list[tuple[str, int, int]]:
    """(condition, cell seed, discovery seed) for every A20 cell."""
    seeds = cfg["seeds"]
    out = []
    for cond in ("cert", "cert_random", "cert_strat", "cert_inverse", "unrestricted"):
        for d in seeds:
            if cond in ("cert_random", "cert_strat"):
                out += [(cond, d + len(seeds) * j, d) for j in range(DRAWS[cond])]
            else:
                out.append((cond, d, d))
    return out


def main() -> None:
    cfg = load_config()
    interv, beh, task = cfg["grid"]["intervention"], "b1", "b1_en"
    assert interv in ("steer", "lora"), interv
    run_root = Path(cfg["out_dir"])
    assert run_root.resolve() != MAIN_RUNS.resolve(), "A20 cells must not go into the main root"
    mw = MemWatch(cfg["memory_budget_gb"])
    if interv == "lora":
        model, tok = load_hf(cfg)
    else:
        model = load_tl(cfg)
        tok = model.tokenizer
    folds = b1_folds(cfg)
    match_size = json.loads(
        (MAIN_RUNS / "interventions/b1/steer/circuit_s0/cell.json").read_text()
    )["meta"]["size_at_1"]
    ev = [json.loads(line) for line in (ROOT / "data/b1/model_eval_en.jsonl").open()]
    st_cfg = cfg["interventions"]["steering"]
    only = set(cfg.get("only") or [])  # e.g. only=[cert_s0] (the reproducibility rebuild)
    for cond, seed, d in plan(cfg):
        if only and f"{cond}_s{seed}" not in only:
            continue
        circ = run_root / "circuits" / task / f"seed{d}"
        cert = json.loads((circ / "certified.json").read_text())["certified"]
        if cert is None:
            print(f"seed {d}: no certified circuit, skipping its cells", flush=True)
            continue
        scores = ck.load_scores(str(circ / "circuit.pt")).scores
        nodes = condition_nodes(cond, seed, cert["nodes"], scores)
        cell_dir = Cell(beh, interv, cond, seed, nodes).dir(run_root)
        if (cell_dir / "delta.safetensors").exists() and not cfg.get("force"):
            print("cached", cell_dir, flush=True)
            continue
        mf, t0 = Manifest(f"a20_{interv}", cfg, seed), time.time()
        for p in (
            ROOT / "data/b1/false_facts.jsonl",
            ROOT / "data/b1/model_eval_en.jsonl",
            circ / "certified.json",
            circ / "circuit.pt",
        ):
            mf.add_input(p)
        fold_keys = [[f["relation"], f["subject_qid"]] for f in folds[d]]
        if interv == "steer":
            fold = {tuple(k) for k in fold_keys}
            source = [
                r["text"]
                for r in ev
                if not r["known"] and (r["relation"], r["subject_qid"]) not in fold
            ][:64]
            target = [r["text"] for r in ev if r["known"]][:64]
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
            base_coeff = float(tensors["coefficient"])
            tensors["coefficient"] = torch.tensor(base_coeff * st_cfg["main_multiplier"])
            cell.meta["coefficient_multiplier"] = st_cfg["main_multiplier"]
            stats = {"size_at_1": s1, "base_coefficient": base_coeff, "match_size": match_size}
        else:
            pairs = [(fact_prompt_en(f, tok, cfg)[0], f["object"]["en"]) for f in folds[d]]
            tensors, stats = lora.train(
                model, tok, pairs, nodes, cond, cfg["interventions"]["lora"], seed
            )
            cell = Cell(beh, interv, cond, seed, nodes, meta=stats)
        cell.meta.update(
            delta_norm=delta_norm(tensors, interv),
            edited_keys=fold_keys,
            disc_seed=d,
            amendment="A20",
            certified_k=cert["k"],
        )
        cell.save(run_root, tensors)
        mw.check()
        mf.add_output(cell_dir / "delta.safetensors")
        mf.write(
            cell_dir / "manifest.json",
            memory=mw.summary(),
            stats=stats,
            wall_s=round(time.time() - t0, 1),
            delta_norm=cell.meta["delta_norm"],
        )
        print(
            f"{cell.name}: {time.time() - t0:.0f}s |Δ|={cell.meta['delta_norm']:.3g} "
            f"{json.dumps(stats, default=str)[:200]}",
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        sc = {f"A{layer}.{h}": float(layer * 16 + h) for layer in range(3) for h in range(16)}
        sc |= {f"MLP {layer}": float(layer) for layer in range(3)}
        cert = ["A0.1", "A0.2", "A2.5", "MLP 1"]
        s = condition_nodes("cert_strat", 4, cert, sc)
        assert sorted(int(n[1:].split(".")[0]) for n in split(s)[0]) == [0, 0, 2]
        assert len(split(s)[1]) == 1 and s == condition_nodes("cert_strat", 4, cert, sc)
        r = condition_nodes("cert_random", 4, cert, sc)
        assert len(split(r)[0]) == 3 and len(split(r)[1]) == 1
        assert condition_nodes("cert_inverse", 0, cert, sc) == ["A0.0", "A0.1", "A0.2", "MLP 0"]
        p = plan({"seeds": [0, 1, 2]})
        assert len(p) == 30 and all(s % 3 == d for _, s, d in p)
        print("run_a20 ok")
    else:
        main()
