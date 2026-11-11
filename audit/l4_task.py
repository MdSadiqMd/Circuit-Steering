"""L4 for one bystander task on every audited cell, from a fresh per-cell attribution.

Amendment A16: the built-in `ioi` bystander was discovered on a mis-specified task (metric at the
answer token, double BOS, misaligned pairs; checks/check_ioi_position.py), so `ioi_fixed`
(data/ioi_fixed.py) replaces it. For every cell that has `l4_edges.safetensors` this applies the
cell, attributes `task` under the audit's own settings (same example subset, seed, IG steps,
batch size as `audit/run_audit.py`), keeps the edge matrix in `l4_edges_<task>.safetensors` and
writes `l4_<task>.json` with, on three node-set definitions (per-type `circuit_nodes`, joint top-k
by |signed row sum|, CircuitKIT pillar set from `cellset.json`):
  cad_induced / cad_repeat_induced / cad_resample_induced   on the induced edge mask
  circuit_distance                                          node turnover (per-type / joint top-k)
Base, same-data repeat and other-subset edges per discovery seed come from
`audit.run_audit.base_edges` (cached under runs/audit/<beh>/base/). `audit/analyze.py` reads the
file as the `ioi` bystander once every L4 cell has it.
Cells audited after amendment A18 get the tasks in `audit.l4.extra_tasks` from the main pass
(audit/l4.py); this script backfills a task on older cells.
Run: uv run python audit/l4_task.py grid.behaviour=b1 task=ioi_fixed [max_cells=]
"""

from __future__ import annotations

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
from safetensors.torch import save_file

from audit.l4 import base_edges, l4_cfg, per_type_distance
from interventions.cell import Cell, apply_tl


def main() -> None:
    cfg = load_config()
    beh, task = cfg["grid"]["behaviour"], str(cfg["task"])
    runs = Path(cfg["out_dir"])
    out_root = runs / "audit" / beh
    top = cfg["discovery"]["top_fraction"]
    spec = resolve_task(task, cfg)
    cells = sorted(
        p.parent
        for p in (runs / "interventions" / beh).glob("*/*/cell.json")
        if (out_root / Cell.peek(p.parent).name / "l4_edges.safetensors").exists()
        and not (out_root / Cell.peek(p.parent).name / f"l4_{task}.json").exists()
    )
    if cfg.get("max_cells"):
        cells = cells[: int(cfg["max_cells"])]
    print(f"{len(cells)} cells for {beh} / {task}", flush=True)
    if not cells:
        return
    mw = MemWatch(cfg["memory_budget_gb"])
    model = load_tl(cfg)
    base_cache: dict[int, dict] = {}
    for d in cells:
        cell, tensors = Cell.load(d)
        disc_seed = (
            cell.seed if cell.seed in cfg["seeds"] else cfg["seeds"][cell.seed % len(cfg["seeds"])]
        )
        if disc_seed not in base_cache:
            base = M.BaseCircuits(runs, [task], disc_seed, top)
            pillar = json.loads(
                (runs / "circuits" / task / f"seed{disc_seed}" / "cellset.json").read_text()
            )["pillar_set"]["nodes"]
            base_cache[disc_seed] = {
                "bd": base.data[task],
                "e": base_edges(model, cfg, beh, [task], disc_seed, out_root)[task],
                "nodes": {
                    "cell_set": circuit_nodes(base.data[task]["signed_nodes"], top),
                    "joint_set": base.data[task]["top_nodes"],
                    "pillar_set": pillar,
                },
            }
        bc = base_cache[disc_seed]
        t0 = time.time()
        undo = apply_tl(model, cell, tensors)
        try:
            new, graph = M.edge_scores(
                model,
                spec,
                l4_cfg(cfg, spec, disc_seed),
                cfg["device"],
                cfg["audit"]["l4"]["ig_steps"],
            )
        finally:
            undo()
        e, bd = bc["e"], bc["bd"]
        names = M.node_names(graph)
        base_nodes = dict(zip(names, e["edges"].sum(dim=1).tolist(), strict=True))
        new_nodes = dict(zip(names, new.sum(dim=1).tolist(), strict=True))
        base_nodes.pop("input", None)
        new_nodes.pop("input", None)
        res = {"disc_seed": disc_seed, "task": task}
        for k, nodes in bc["nodes"].items():
            res[k] = M.cad_triple(e, new, M.induced_edge_mask(graph, nodes))
        res["cell_set"]["circuit_distance"] = per_type_distance(base_nodes, new_nodes, top)
        res["joint_set"]["circuit_distance"] = M.circuit_distance(base_nodes, new_nodes, bd["k"])[
            "circuit_distance"
        ]
        save_file({task: new}, str(out_root / cell.name / f"l4_edges_{task}.safetensors"))
        (out_root / cell.name / f"l4_{task}.json").write_text(json.dumps(res, default=str))
        print(
            f"{cell.name} l4_{task} {time.time() - t0:.0f}s cad "
            f"cell={res['cell_set']['cad_induced']['cad']:.3f} "
            f"(floor {res['cell_set']['cad_resample_induced']['cad']:.3f}) "
            f"joint={res['joint_set']['cad_induced']['cad']:.3f} "
            f"pillar={res['pillar_set']['cad_induced']['cad']:.3f}",
            flush=True,
        )
        mw.check()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
