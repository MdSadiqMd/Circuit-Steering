"""CAD on the CircuitKIT pillar set (the joint-ranked set whose faithfulness §4.1 reports and the
set amendment A15's `circuit_joint` cells intervene on), for every audited cell, from the kept edge
matrices. No attribution is rerun: `l4_edges.safetensors` holds each cell's post-intervention edge
scores and `runs/audit/<beh>/base/l4_edges_*` the base, repeat and resample edges, so a third mask
is one offline pass. Writes runs/audit/<beh>/<cell>/l4_pillar.json: per task cad / repeat /
resample on the induced edge set of `cellset.json` `pillar_set`; the pillar-set AS-NC is not
recomputed (activations are not kept). Run: uv run python audit/l4_pillarmask.py grid.behaviour=b1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
from aya_circuits.config import load_config
from aya_circuits.model import load_tl
from aya_circuits.registry import resolve_task
from neb import mechanistic as M
from safetensors.torch import load_file

from audit.l4 import BYSTANDERS, l4_cfg
from interventions.cell import Cell


def main() -> None:
    cfg = load_config()
    beh = cfg["grid"]["behaviour"]
    runs = Path(cfg["out_dir"])
    out_root = runs / "audit" / beh
    tasks = [f"{beh}_en"] + BYSTANDERS[beh]
    model = load_tl(cfg)
    graphs = {}
    for c in tasks:  # the graph object only (two examples, one IG step) for the edge index
        spec = resolve_task(c, cfg)
        small = l4_cfg(cfg, spec, 0) | {"data_params": {"num_examples": 2, "seed": 0}}
        _, graphs[c] = M.edge_scores(model, spec, small, cfg["device"], 1)
    masks = {}
    for seed in cfg["seeds"]:
        for c in tasks:
            pillar = json.loads(
                (runs / "circuits" / c / f"seed{seed}" / "cellset.json").read_text()
            )
            masks[seed, c] = M.induced_edge_mask(graphs[c], pillar["pillar_set"]["nodes"])
    n = 0
    for cj in sorted((runs / "interventions" / beh).glob("*/*/cell.json")):
        cell = Cell.peek(cj.parent)
        d = out_root / cell.name
        if (d / "l4_pillar.json").exists() or not (d / "l4_edges.safetensors").exists():
            continue
        disc_seed = (
            cell.seed if cell.seed in cfg["seeds"] else cfg["seeds"][cell.seed % len(cfg["seeds"])]
        )
        edges = load_file(str(d / "l4_edges.safetensors"))
        res = {"disc_seed": disc_seed}
        for c in tasks:
            e = load_file(str(out_root / "base" / f"l4_edges_{c}_s{disc_seed}.safetensors"))
            res[c] = M.cad_triple(e, edges[c], masks[disc_seed, c])
        (d / "l4_pillar.json").write_text(json.dumps(res, default=str))
        n += 1
    print(f"wrote l4_pillar.json for {n} cells of {beh}")


if __name__ == "__main__":
    main()
