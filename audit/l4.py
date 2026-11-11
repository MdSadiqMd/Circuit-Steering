"""L4 (mechanistic) core shared by the audit stages: the attribution settings, the base edges with
their two floors, and one per-cell attribution pass that writes every L4 artifact at once.

Per cell (`attribute_cell`): apply the cell to the TL model, re-attribute the target task, the three
bystanders and `audit.l4.extra_tasks` (A16: `ioi_fixed`) under the audit settings (same example
subset, seed, IG steps and batch size as the base edges), keep the edge matrices, undo. Written next
to the cell under runs/audit/<beh>/<cell>/:
  l4.json                 joint-ranked set (top 61 of 612 by |signed row sum|): CAD on the induced
                          and top-1 % edge masks with both floors, node turnover, AS-NC
  l4_cellset.json         the pre-registered per-type set (`cell_set`) next to the joint set
  l4_<task>.json          every extra task on the per-type, joint and CircuitKIT pillar sets
  l4_edges*.safetensors   the edge matrices, so any other mask is an offline pass
                          (audit/l4_pillarmask.py, audit/l4_shared.py)
Before amendment A18 `audit/run_audit.py` and `audit/l4_cellset.py` attributed every cell twice
(`repro_max_abs_diff` recorded the agreement); cells written here carry `single_pass`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from aya_circuits.nodes import circuit_nodes
from aya_circuits.registry import resolve_task
from neb import mechanistic as M
from safetensors.torch import load_file, save_file

from interventions.cell import Cell, apply_tl

ROOT = Path(__file__).resolve().parent.parent
BYSTANDERS = {"b1": ["b2_en", "ioi", "langid_en"], "b2": ["b1_en", "ioi", "langid_en"]}


def l4_cfg(cfg: dict, spec, seed: int) -> dict:
    a = cfg["audit"]["l4"]
    return {
        "algorithm": "eap-ig",
        "task": spec.name,
        "level": "node",
        "batch_size": a["batch_size"],
        "data_params": {"num_examples": a["n_examples"], "seed": seed},
        "seed": seed,
        "cache_dir": cfg["cache_dir"],
    }


def base_edges(
    model, cfg: dict, beh: str, circuits: list[str], seed: int, out_root: Path
) -> dict[str, dict]:
    """Base-model signed edge scores for every circuit under the *audit's* settings (same examples,
    same IG steps as the intervened recomputation), plus a same-data recomputation (measurement
    floor) and a second-subset recomputation (sampling floor). Comparing against the discovery
    run's edges instead gave CAD ~1 for a no-op cell (smoke run). Cached under <out_root>/base/."""
    out = {}
    for c in circuits:
        p = out_root / "base" / f"l4_edges_{c}_s{seed}.safetensors"
        if not p.exists():
            spec = resolve_task(c, cfg)
            steps = cfg["audit"]["l4"]["ig_steps"]
            e0, _ = M.edge_scores(model, spec, l4_cfg(cfg, spec, seed), cfg["device"], steps)
            e1, _ = M.edge_scores(model, spec, l4_cfg(cfg, spec, seed + 1000), cfg["device"], steps)
            e2, _ = M.edge_scores(model, spec, l4_cfg(cfg, spec, seed), cfg["device"], steps)
            p.parent.mkdir(parents=True, exist_ok=True)
            save_file({"edges": e0, "edges_alt_seed": e1, "edges_repeat": e2}, str(p))
        out[c] = load_file(str(p))
    return out


def per_type_distance(base_nodes: dict, new_nodes: dict, top: float) -> float:
    """Node turnover of the per-type top sets (the pre-registered circuit definition)."""
    from aya_circuits.nodes import sizes, top_k
    from aya_circuits.overlap import jaccard

    kh, km = sizes(base_nodes, top)
    return 1 - jaccard(top_k(base_nodes, kh, km), top_k(new_nodes, kh, km))


def node_sums(graph, edges: torch.Tensor) -> dict[str, float]:
    d = dict(zip(M.node_names(graph), edges.sum(dim=1).tolist(), strict=True))
    d.pop("input", None)
    return d


def disc_seed_of(cell: Cell, cfg: dict) -> int:
    return cell.seed if cell.seed in cfg["seeds"] else cfg["seeds"][cell.seed % len(cfg["seeds"])]


class Base:
    """Everything one discovery seed's cells share: base edges with floors, node sets under the
    three definitions, and the base model's last-position activations (for AS-NC)."""

    def __init__(self, model, cfg: dict, beh: str, seed: int, out_root: Path):
        runs, top = Path(cfg["out_dir"]), cfg["discovery"]["top_fraction"]
        self.seed, self.top = seed, top
        self.tasks = [f"{beh}_en"] + BYSTANDERS[beh]
        self.extras = list(cfg["audit"]["l4"].get("extra_tasks") or [])
        every = self.tasks + self.extras
        self.circuits = M.BaseCircuits(runs, every, seed, top)
        self.edges = base_edges(model, cfg, beh, every, seed, out_root)
        self.cell_nodes = {
            c: circuit_nodes(self.circuits.data[c]["signed_nodes"], top) for c in every
        }
        self.pillar = {
            c: json.loads((runs / "circuits" / c / f"seed{seed}" / "cellset.json").read_text())[
                "pillar_set"
            ]["nodes"]
            for c in self.extras
        }
        ev = [json.loads(line) for line in (ROOT / "data/b1/model_eval_en.jsonl").open()][:32]
        self.texts = [r["text"] for r in ev]
        self.all_nodes = list(self.circuits.data[self.tasks[0]]["signed_nodes"])
        self.acts = M.last_position_acts(model, self.texts, self.all_nodes)


def attribute_cell(model, cfg: dict, cell: Cell, tensors: dict, base: Base, out: Path) -> dict:
    """One attribution pass with the cell applied; writes l4.json, l4_cellset.json, the extra-task
    files and the edge matrices under `out`; returns the l4.json content."""
    steps, target = cfg["audit"]["l4"]["ig_steps"], base.tasks[0]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    undo = apply_tl(model, cell, tensors)
    try:
        edges, graph = {}, None
        for c in base.tasks + base.extras:
            spec = resolve_task(c, cfg)
            edges[c], graph = M.edge_scores(
                model, spec, l4_cfg(cfg, spec, base.seed), cfg["device"], steps
            )
        new_acts = M.last_position_acts(model, base.texts, base.all_nodes)
    finally:
        undo()
    l4: dict = {}
    l4c: dict = {"disc_seed": base.seed}
    for c in base.tasks:
        bd, e, new = base.circuits.data[c], base.edges[c], edges[c]
        bn, nn = node_sums(graph, e["edges"]), node_sums(graph, new)
        m_joint = M.induced_edge_mask(graph, bd["top_nodes"])
        m_top = M.top_edges_mask(e["edges"], graph, 0.01)
        joint = M.cad_triple(e, new, m_joint)
        dist = M.circuit_distance(bn, nn, bd["k"])
        l4[c] = joint | {
            "cad_top1pct_edges": M.cad(e["edges"], new, m_top),
            "cad_repeat_top1pct_edges": M.cad(e["edges"], e["edges_repeat"], m_top),
            "cad_resample_top1pct_edges": M.cad(e["edges"], e["edges_alt_seed"], m_top),
            "circuit_distance": dist,
        }
        l4c[c] = {
            "cell_set": M.cad_triple(e, new, M.induced_edge_mask(graph, base.cell_nodes[c]))
            | {"circuit_distance": per_type_distance(bn, nn, base.top)},
            "joint_set": joint
            | {"circuit_distance": dist["circuit_distance"], "stored_cad_induced": None},
        }
    joint_as_nc = M.as_nc(base.acts, new_acts, base.circuits.data[target]["top_nodes"])
    l4["as_nc_target_prompts"] = {"n_prompts": len(base.texts)}
    l4["as_nc"] = joint_as_nc
    l4c["as_nc"] = {
        "cell_set": M.as_nc(base.acts, new_acts, base.cell_nodes[target]),
        "joint_set": joint_as_nc,
        "stored_joint_as_nc": None,
    }
    l4c["repro_max_abs_diff"] = None
    l4c["single_pass"] = l4["single_pass"] = "one attribution pass (audit/l4.py, A18)"
    for c in base.extras:
        bd, e, new = base.circuits.data[c], base.edges[c], edges[c]
        bn, nn = node_sums(graph, e["edges"]), node_sums(graph, new)
        res = {"disc_seed": base.seed, "task": c}
        for k, nodes in (
            ("cell_set", base.cell_nodes[c]),
            ("joint_set", bd["top_nodes"]),
            ("pillar_set", base.pillar[c]),
        ):
            res[k] = M.cad_triple(e, new, M.induced_edge_mask(graph, nodes))
        res["cell_set"]["circuit_distance"] = per_type_distance(bn, nn, base.top)
        res["joint_set"]["circuit_distance"] = M.circuit_distance(bn, nn, bd["k"])[
            "circuit_distance"
        ]
        save_file({c: new}, str(out / f"l4_edges_{c}.safetensors"))
        (out / f"l4_{c}.json").write_text(json.dumps(res, default=str))
    save_file({c: edges[c] for c in base.tasks}, str(out / "l4_edges.safetensors"))
    (out / "l4.json").write_text(json.dumps(l4, default=str))
    (out / "l4_cellset.json").write_text(json.dumps(l4c, default=str))
    t = l4c[target]["cell_set"]
    print(
        f"{cell.name} l4 {time.time() - t0:.0f}s target cad cell={t['cad_induced']['cad']:.3f} "
        f"(floor {t['cad_resample_induced']['cad']:.3f}) "
        f"joint={l4[target]['cad_induced']['cad']:.3f}",
        flush=True,
    )
    return l4
