"""L4 power check: is the mechanistic layer blind, or under-sampled?

Every L4 number in the study uses 32 examples and 3 IG steps, and its resample floor (base vs base
on another example subset) is 0.35–0.58, above the CAD of every LoRA and steering cell. This
recomputes the base edges (main, same-data repeat, alternate subset) and a few cells' edges at
`n_examples` / `ig_steps` given on the command line (default 64 / 10) for the target and the three
bystanders, and reports CAD on both mask definitions next to the stored 32 / 3 values, so the
floor's dependence on the attribution budget is measured rather than assumed. Writes
runs/audit/<beh>/power/l4_power.json.
Run: uv run python audit/l4_power.py [n_examples=64 ig_steps=10 cells=lora/circuit_s0,...]
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_tl
from aya_circuits.nodes import circuit_nodes
from aya_circuits.registry import resolve_task
from neb import mechanistic as M
from safetensors.torch import load_file, save_file

from audit.l4 import BYSTANDERS, l4_cfg
from interventions.cell import Cell, apply_tl

DEFAULT_CELLS = "lora/circuit_s0,steer/circuit_s0,lora/unrestricted_s0,edit/taylor_s0"


def main() -> None:
    cfg = load_config()
    beh = cfg["grid"]["behaviour"]
    runs = Path(cfg["out_dir"])
    out_dir = runs / "audit" / beh / "power"
    out_dir.mkdir(parents=True, exist_ok=True)
    big = copy.deepcopy(cfg)
    big["audit"]["l4"]["n_examples"] = int(cfg.get("n_examples", 64))
    big["audit"]["l4"]["ig_steps"] = int(cfg.get("ig_steps", 10))
    steps = big["audit"]["l4"]["ig_steps"]
    tasks = [f"{beh}_en"] + BYSTANDERS[beh]
    top = cfg["discovery"]["top_fraction"]
    cells = [
        runs / "interventions" / beh / c for c in str(cfg.get("cells", DEFAULT_CELLS)).split(",")
    ]
    disc_seed = 0
    mw = MemWatch(cfg["memory_budget_gb"])
    model = load_tl(cfg)
    base = M.BaseCircuits(runs, tasks, disc_seed, top)
    bedges, graphs = {}, {}
    for c in tasks:
        p = out_dir / f"base_edges_{c}_s{disc_seed}.safetensors"
        spec = resolve_task(c, cfg)
        if not p.exists():
            t0 = time.time()
            e0, g = M.edge_scores(model, spec, l4_cfg(big, spec, disc_seed), cfg["device"], steps)
            e1, _ = M.edge_scores(
                model, spec, l4_cfg(big, spec, disc_seed + 1000), cfg["device"], steps
            )
            e2, _ = M.edge_scores(model, spec, l4_cfg(big, spec, disc_seed), cfg["device"], steps)
            save_file({"edges": e0, "edges_alt_seed": e1, "edges_repeat": e2}, str(p))
            print(f"base {c}: {time.time() - t0:.0f}s", flush=True)
            mw.check()
        bedges[c] = load_file(str(p))
        if c not in graphs:
            _, graphs[c] = M.edge_scores(
                model,
                spec,
                l4_cfg(big, spec, disc_seed) | {"data_params": {"num_examples": 2, "seed": 0}},
                cfg["device"],
                1,
            )
    res_p = out_dir / "l4_power.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}
    res["settings"] = {"n_examples": big["audit"]["l4"]["n_examples"], "ig_steps": steps}
    res["floors"] = {}
    for c in tasks:
        g, e = graphs[c], bedges[c]
        masks = {
            "cell": M.induced_edge_mask(g, circuit_nodes(base.data[c]["signed_nodes"], top)),
            "joint": M.induced_edge_mask(g, base.data[c]["top_nodes"]),
        }
        res["floors"][c] = {
            k: {
                "repeat": M.cad(e["edges"], e["edges_repeat"], m)["cad"],
                "resample": M.cad(e["edges"], e["edges_alt_seed"], m)["cad"],
                "resample_spearman": M.cad(e["edges"], e["edges_alt_seed"], m)["spearman"],
            }
            for k, m in masks.items()
        }
    print("floors", json.dumps(res["floors"]), flush=True)
    for d in cells:
        cell, tensors = Cell.load(d)
        if cell.name in res and "cells" in res.get(cell.name, {}):
            continue
        stored_p = runs / "audit" / beh / cell.name / "l4_cellset.json"
        stored = json.loads(stored_p.read_text()) if stored_p.exists() else {}
        t0 = time.time()
        undo = apply_tl(model, cell, tensors)
        try:
            out = {}
            for c in tasks:
                spec = resolve_task(c, cfg)
                new, g = M.edge_scores(
                    model, spec, l4_cfg(big, spec, disc_seed), cfg["device"], steps
                )
                e = bedges[c]
                out[c] = {}
                for k, nodes in (
                    ("cell", circuit_nodes(base.data[c]["signed_nodes"], top)),
                    ("joint", base.data[c]["top_nodes"]),
                ):
                    m = M.induced_edge_mask(g, nodes)
                    out[c][k] = {
                        "cad": M.cad(e["edges"], new, m)["cad"],
                        "cad_excess": M.cad(e["edges"], new, m)["cad"]
                        - M.cad(e["edges"], e["edges_alt_seed"], m)["cad"],
                        "stored_32_3": (
                            stored.get(c, {}).get(f"{k}_set", {}).get("cad_induced") or {}
                        ).get("cad"),
                        "stored_32_3_resample": (
                            stored.get(c, {}).get(f"{k}_set", {}).get("cad_resample_induced") or {}
                        ).get("cad"),
                    }
        finally:
            undo()
        res[cell.name] = {"cells": out, "wall_s": round(time.time() - t0)}
        res_p.write_text(json.dumps(res, indent=1))
        print(cell.name, json.dumps(out), flush=True)
        mw.check()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    res_p.write_text(json.dumps(res, indent=1))
    print("wrote", res_p)


if __name__ == "__main__":
    main()
