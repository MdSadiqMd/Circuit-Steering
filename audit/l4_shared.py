"""H4 within one bystander circuit (amendment A17): do the bystander's edges that are shared with
the intervened components move more than its other edges?

The pre-registered H4 test compares three bystander circuits, so overlap is confounded with
circuit identity (report §4.5). This partitions each bystander's own induced edge set E(B) into
E(B) ∩ E(X) and E(B) \\ E(X) for a node set X, and scores CAD, the same-data repeat floor and the
resample floor on each part from the kept edge matrices (`l4_edges*.safetensors`, base files under
runs/audit/<beh>/base/). Three choices of X per cell:
  cell      the components the cell intervened on (steering / LoRA nodes; MEMIT: the edited MLP
            layers); here "shared" = edges with *either* endpoint in X (an adapter on a head
            changes that head's inputs and output); skipped for unrestricted steering / LoRA
  target    the pre-registered per-type target circuit (`aya_circuits.nodes.circuit_nodes`)
  pillar    CircuitKIT's joint-ranked target set (`cellset.json` `pillar_set`)
D = (CAD_shared − floor_shared) − (CAD_unshared − floor_unshared) > 0 means the overlap region of
the same circuit, under the same attribution run, moved more. No model is loaded: the edge index
comes from `Graph.from_model` on the model's dimensions (verified to reproduce the stored CAD).
Writes runs/audit/<beh>/<cell>/l4_shared.json.
Run: uv run python audit/l4_shared.py grid.behaviour=b1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
from aya_circuits.config import load_config
from aya_circuits.nodes import circuit_nodes
from circuitkit.backends.eap.graph import Graph
from neb import mechanistic as M
from safetensors.torch import load_file

from audit.l4 import BYSTANDERS
from interventions.cell import Cell

GRAPH_CFG = {
    "n_layers": 36,
    "n_heads": 16,
    "d_model": 2048,
    "d_mlp": 11008,
    "parallel_attn_mlp": True,
}
MIN_EDGES = 10


def intervened_nodes(cell: Cell) -> list[str] | None:
    if cell.intervention == "edit":
        return [f"MLP {layer}" for layer in cell.meta["layers"]]
    if not cell.nodes or len(cell.nodes) >= 612:
        return None
    return list(cell.nodes)


def part(base: dict, new, mask) -> dict:
    if int(mask.sum()) < MIN_EDGES:
        return {"n_edges": int(mask.sum())}
    return {
        "n_edges": int(mask.sum()),
        "cad": M.cad(base["edges"], new, mask)["cad"],
        "repeat": M.cad(base["edges"], base["edges_repeat"], mask)["cad"],
        "resample": M.cad(base["edges"], base["edges_alt_seed"], mask)["cad"],
    }


def main() -> None:
    cfg = load_config()
    beh = cfg["grid"]["behaviour"]
    runs = Path(cfg["out_dir"])
    out_root = runs / "audit" / beh
    top = cfg["discovery"]["top_fraction"]
    graph = Graph.from_model(GRAPH_CFG)
    target = f"{beh}_en"
    tasks = BYSTANDERS[beh] + ["ioi_fixed"]
    per_seed: dict[int, dict] = {}
    n = 0
    for cj in sorted((runs / "interventions" / beh).glob("*/*/cell.json")):
        cell = Cell.peek(cj.parent)
        d = out_root / cell.name
        if (d / "l4_shared.json").exists() or not (d / "l4_edges.safetensors").exists():
            continue
        seed = (
            cell.seed if cell.seed in cfg["seeds"] else cfg["seeds"][cell.seed % len(cfg["seeds"])]
        )
        if seed not in per_seed:
            tnodes = json.loads(
                (runs / "circuits" / target / f"seed{seed}" / "nodes.json").read_text()
            )
            signed = dict(zip(tnodes["names"], tnodes["signed_rowsum"], strict=True))
            signed.pop("input", None)
            cs = runs / "circuits" / target / f"seed{seed}" / "cellset.json"
            per_seed[seed] = {
                "target": circuit_nodes(signed, top),
                "pillar": json.loads(cs.read_text())["pillar_set"]["nodes"],
                "bystanders": {},
            }
            for b in tasks:
                bn = runs / "circuits" / b / f"seed{seed}" / "nodes.json"
                bp = out_root / "base" / f"l4_edges_{b}_s{seed}.safetensors"
                if not bn.exists() or not bp.exists():
                    continue
                nodes = json.loads(bn.read_text())
                bsig = dict(zip(nodes["names"], nodes["signed_rowsum"], strict=True))
                bsig.pop("input", None)
                per_seed[seed]["bystanders"][b] = {
                    "mask": M.induced_edge_mask(graph, circuit_nodes(bsig, top)),
                    "base": load_file(str(bp)),
                }
        ps = per_seed[seed]
        edges = load_file(str(d / "l4_edges.safetensors"))
        fx = d / "l4_edges_ioi_fixed.safetensors"
        if fx.exists():
            edges |= load_file(str(fx))
        assert next(iter(edges.values())).shape == tuple(graph.real_edge_mask.shape)
        sets = {"cell": intervened_nodes(cell), "target": ps["target"], "pillar": ps["pillar"]}
        res = {"disc_seed": seed, "n_cell_nodes": len(sets["cell"] or [])}
        for b, bd in ps["bystanders"].items():
            if b not in edges:
                continue
            res[b] = {}
            for name, nodes in sets.items():
                if nodes is None:
                    continue
                shared = bd["mask"] & (
                    M.touching_edge_mask(graph, nodes)
                    if name == "cell"
                    else M.induced_edge_mask(graph, nodes)
                )
                res[b][name] = {
                    "shared": part(bd["base"], edges[b], shared),
                    "unshared": part(bd["base"], edges[b], bd["mask"] & ~shared),
                }
        (d / "l4_shared.json").write_text(json.dumps(res))
        n += 1
    print(f"wrote l4_shared.json for {n} cells of {beh}")


if __name__ == "__main__":
    main()
