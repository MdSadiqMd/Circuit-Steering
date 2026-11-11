"""L4 on the pre-registered circuit definition, from a fresh per-cell re-attribution that is kept.

Before amendment A18 `audit/run_audit.py` scored L4 on the joint-ranked node sets (top 61 of 612 by
|signed row sum|, docs/audit-2026-11-05.md B1) and stored only the resulting numbers; cells audited
since then get both definitions from one pass (audit/l4.py) and never reach this script. For the
older cells this pass re-applies every cell that has an `l4.json` but no `l4_cellset.json`,
re-attributes the target task and the three bystanders under the identical audit settings (same
example subset, seed, IG steps, batch size), saves the four edge-score matrices
(`l4_edges.safetensors`, {task: [n_forward, n_backward]}) so any mask can be re-scored offline, and
writes `l4_cellset.json` with, per task:
  cell_set:   cad_induced / cad_repeat_induced / cad_resample_induced on the edge set induced by the
              per-type circuit (`aya_circuits.nodes.circuit_nodes`), circuit distance with per-type
              top-k
  joint_set:  the same three CADs on the joint-ranked mask, a reproduction check against l4.json
and AS-NC on the target task with the cell set as "inside" (and the joint set, for the same check).
Run: uv run python audit/l4_cellset.py grid.behaviour=b1 [cells=steer,lora,edit]
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

from audit.l4 import BYSTANDERS, ROOT, base_edges, l4_cfg, per_type_distance
from interventions.cell import Cell, apply_tl


def main() -> None:
    cfg = load_config()
    beh = cfg["grid"]["behaviour"]
    want = set(str(cfg.get("cells", "steer,lora,edit")).split(","))
    runs = Path(cfg["out_dir"])
    out_root = runs / "audit" / beh
    top = cfg["discovery"]["top_fraction"]
    a = cfg["audit"]["l4"]
    tasks = [f"{beh}_en"] + BYSTANDERS[beh]
    cells = sorted(
        p.parent
        for p in (runs / "interventions" / beh).glob("*/*/cell.json")
        if p.parent.parent.name in want
        and (out_root / Cell.peek(p.parent).name / "l4.json").exists()
        and not (out_root / Cell.peek(p.parent).name / "l4_cellset.json").exists()
    )
    if cfg.get("max_cells"):
        cells = cells[: int(cfg["max_cells"])]
    print(f"{len(cells)} cells for {beh}", flush=True)
    if not cells:
        return
    mw = MemWatch(cfg["memory_budget_gb"])
    model = load_tl(cfg)
    ev = [json.loads(line) for line in (ROOT / "data/b1/model_eval_en.jsonl").open()][:32]
    texts = [r["text"] for r in ev]
    base_cache: dict[int, dict] = {}
    for d in cells:
        cell, tensors = Cell.load(d)
        disc_seed = (
            cell.seed if cell.seed in cfg["seeds"] else cfg["seeds"][cell.seed % len(cfg["seeds"])]
        )
        if disc_seed not in base_cache:
            base = M.BaseCircuits(runs, tasks, disc_seed, top)
            bedges = base_edges(model, cfg, beh, tasks, disc_seed, out_root)
            all_nodes = list(base.data[tasks[0]]["signed_nodes"])
            base_cache[disc_seed] = {
                "base": base,
                "bedges": bedges,
                "cell_nodes": {c: circuit_nodes(base.data[c]["signed_nodes"], top) for c in tasks},
                "acts": M.last_position_acts(model, texts, all_nodes),
                "all_nodes": all_nodes,
            }
        bc = base_cache[disc_seed]
        stored = json.loads((out_root / cell.name / "l4.json").read_text())
        t0 = time.time()
        undo = apply_tl(model, cell, tensors)
        try:
            edges, res = {}, {"disc_seed": disc_seed}
            for c in tasks:
                spec = resolve_task(c, cfg)
                new, graph = M.edge_scores(
                    model, spec, l4_cfg(cfg, spec, disc_seed), cfg["device"], a["ig_steps"]
                )
                edges[c] = new
                bd, e = bc["base"].data[c], bc["bedges"][c]
                m_cell = M.induced_edge_mask(graph, bc["cell_nodes"][c])
                m_joint = M.induced_edge_mask(graph, bd["top_nodes"])
                names = M.node_names(graph)
                base_nodes = dict(zip(names, e["edges"].sum(dim=1).tolist(), strict=True))
                new_nodes = dict(zip(names, new.sum(dim=1).tolist(), strict=True))
                base_nodes.pop("input", None)
                new_nodes.pop("input", None)
                res[c] = {
                    "cell_set": M.cad_triple(e, new, m_cell)
                    | {"circuit_distance": per_type_distance(base_nodes, new_nodes, top)},
                    "joint_set": M.cad_triple(e, new, m_joint)
                    | {
                        "circuit_distance": M.circuit_distance(base_nodes, new_nodes, bd["k"])[
                            "circuit_distance"
                        ],
                        "stored_cad_induced": stored[c]["cad_induced"]["cad"],
                    },
                }
            new_acts = M.last_position_acts(model, texts, bc["all_nodes"])
        finally:
            undo()
        res["as_nc"] = {
            "cell_set": M.as_nc(bc["acts"], new_acts, bc["cell_nodes"][tasks[0]]),
            "joint_set": M.as_nc(bc["acts"], new_acts, bc["base"].data[tasks[0]]["top_nodes"]),
            "stored_joint_as_nc": stored.get("as_nc", {}).get("as_nc"),
        }
        res["repro_max_abs_diff"] = max(
            abs(
                res[c]["joint_set"]["cad_induced"]["cad"]
                - res[c]["joint_set"]["stored_cad_induced"]
            )
            for c in tasks
        )
        save_file(edges, str(out_root / cell.name / "l4_edges.safetensors"))
        (out_root / cell.name / "l4_cellset.json").write_text(json.dumps(res, default=str))
        t = res[tasks[0]]
        print(
            f"{cell.name} l4_cellset {time.time() - t0:.0f}s target cad "
            f"cell={t['cell_set']['cad_induced']['cad']:.3f} "
            f"(floor {t['cell_set']['cad_resample_induced']['cad']:.3f}) "
            f"joint={t['joint_set']['cad_induced']['cad']:.3f} "
            f"repro_diff={res['repro_max_abs_diff']:.2e}",
            flush=True,
        )
        mw.check()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
