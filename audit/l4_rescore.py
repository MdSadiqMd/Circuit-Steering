"""Fix for the L4 base-edge scale mismatch found in the third audit (docs/audit-2026-11-06.md).

CircuitKIT's `attribute` divides the accumulated scores by the number of examples while our metric
is already a batch mean (aya_circuits.tasks.logit_diff, mean=True), so EAP-IG edge scores scale as
1/batch_size. The cached base edges runs/audit/b1/base/l4_edges_{b1_en,b2_en,ioi}_s0 were computed
at batch 4 before the L4 OOM fix (docs/notes.md 2026-11-03 17:20) and every cell at batch 2, so
those cells' edges were exactly 2× the base and every CAD on those circuits read ≈ 1.

This script (1) moves the stale base files to runs/_superseded/2026-11-06_l4_base_bs4/, (2)
recomputes them at the audit batch size through the same `base_edges` path, (3) rescores every
affected cell offline from its kept edges (l4_edges.safetensors): both circuit definitions in
l4_cellset.json and the joint-set fields of l4.json (CAD induced / repeat / resample, top-1 % edge
CAD, circuit distance); AS-NC is activation-based and unchanged. Old cell JSONs are kept next to the
old base files. Finally it checks, for every behaviour, seed and circuit, that the median ratio of
cell to base edges is ≈ 1.
Run: uv run python audit/l4_rescore.py grid.behaviour=b1 disc_seed=0 "tasks=[b1_en,b2_en,ioi]"
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.model import load_tl
from aya_circuits.nodes import circuit_nodes
from circuitkit.backends.eap.graph import Graph
from neb import mechanistic as M
from safetensors.torch import load_file

from audit.l4 import BYSTANDERS, base_edges, per_type_distance

ROOT = Path(__file__).resolve().parent.parent
SUP = "_superseded/2026-11-06_l4_base_bs4"


def scale_report(runs: Path) -> list[str]:
    """Median |cell edge / base edge| per (behaviour, disc seed, circuit) over every L4 cell."""
    lines = []
    for beh in ("b1", "b2"):
        acc: dict[tuple[int, str], list[float]] = {}
        for p in sorted((runs / "audit" / beh).glob("*/l4_edges.safetensors")):
            seed = json.loads((p.parent / "l4_cellset.json").read_text())["disc_seed"]
            new = load_file(str(p))
            for t, n in new.items():
                e = load_file(
                    str(runs / "audit" / beh / "base" / f"l4_edges_{t}_s{seed}.safetensors")
                )
                m = e["edges"].abs() > 1e-4
                acc.setdefault((seed, t), []).append(float((n[m] / e["edges"][m]).median()))
        for (seed, t), v in sorted(acc.items()):
            lines.append(
                f"{beh} s{seed} {t}: median cell/base ratio {torch.tensor(v).median():.3f}"
            )
    return lines


def main() -> None:
    cfg = load_config()
    beh, seed = cfg["grid"]["behaviour"], int(cfg["disc_seed"])
    tasks = list(cfg["tasks"])
    runs = ROOT / cfg["out_dir"]
    out_root = runs / "audit" / beh
    sup = runs / SUP / beh
    all_tasks = [f"{beh}_en"] + BYSTANDERS[beh]
    top = cfg["discovery"]["top_fraction"]
    for t in tasks:
        p = out_root / "base" / f"l4_edges_{t}_s{seed}.safetensors"
        if p.exists() and not (sup / "base" / p.name).exists():
            (sup / "base").mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), sup / "base" / p.name)
    model = load_tl(cfg)
    bedges = base_edges(model, cfg, beh, all_tasks, seed, out_root)
    for t in tasks:
        old = load_file(str(sup / "base" / f"l4_edges_{t}_s{seed}.safetensors"))["edges"]
        ratio = float(bedges[t]["edges"].abs().sum() / old.abs().sum())
        print(f"{t}: sum|new base| / sum|old base| = {ratio:.3f}")
    graph = Graph.from_model(model, node_scores=True)
    base = M.BaseCircuits(runs, all_tasks, seed, top)
    names = M.node_names(graph)
    n_cells = 0
    for cj in sorted(out_root.glob("*/l4_cellset.json")):
        l4c = json.loads(cj.read_text())
        if l4c.get("disc_seed") != seed or not (cj.parent / "l4_edges.safetensors").exists():
            continue
        l4p = cj.parent / "l4.json"
        keep = sup / cj.parent.name
        if not keep.exists():
            keep.mkdir(parents=True)
            shutil.copy2(cj, keep / cj.name)
            if l4p.exists():
                shutil.copy2(l4p, keep / l4p.name)
        l4 = json.loads(l4p.read_text()) if l4p.exists() else {}
        new_edges = load_file(str(cj.parent / "l4_edges.safetensors"))
        for t in tasks:
            e, new = bedges[t], new_edges[t]
            bd = base.data[t]
            m_cell = M.induced_edge_mask(graph, circuit_nodes(bd["signed_nodes"], top))
            m_joint = M.induced_edge_mask(graph, bd["top_nodes"])
            m_top = M.top_edges_mask(e["edges"], graph, 0.01)
            bn = dict(zip(names, e["edges"].sum(dim=1).tolist(), strict=True))
            nn = dict(zip(names, new.sum(dim=1).tolist(), strict=True))
            bn.pop("input", None)
            nn.pop("input", None)
            joint = {
                "cad_induced": M.cad(e["edges"], new, m_joint),
                "cad_repeat_induced": M.cad(e["edges"], e["edges_repeat"], m_joint),
                "cad_resample_induced": M.cad(e["edges"], e["edges_alt_seed"], m_joint),
            }
            l4c[t]["cell_set"].update(
                cad_induced=M.cad(e["edges"], new, m_cell),
                cad_repeat_induced=M.cad(e["edges"], e["edges_repeat"], m_cell),
                cad_resample_induced=M.cad(e["edges"], e["edges_alt_seed"], m_cell),
                circuit_distance=per_type_distance(bn, nn, top),
            )
            dist = M.circuit_distance(bn, nn, bd["k"])
            l4c[t]["joint_set"].update(
                joint, circuit_distance=dist["circuit_distance"], stored_cad_induced=None
            )
            if t in l4:
                l4[t].update(
                    joint,
                    cad_top1pct_edges=M.cad(e["edges"], new, m_top),
                    cad_repeat_top1pct_edges=M.cad(e["edges"], e["edges_repeat"], m_top),
                    cad_resample_top1pct_edges=M.cad(e["edges"], e["edges_alt_seed"], m_top),
                    circuit_distance=dist,
                )
                l4c[t]["joint_set"]["stored_cad_induced"] = l4[t]["cad_induced"]["cad"]
        l4c["repro_max_abs_diff"] = (
            max(
                abs(
                    l4c[t]["joint_set"]["cad_induced"]["cad"]
                    - (l4c[t]["joint_set"]["stored_cad_induced"] or 0)
                )
                for t in all_tasks
                if l4c[t]["joint_set"].get("stored_cad_induced") is not None
            )
            if l4
            else None
        )
        l4c["rescored"] = (
            "2026-11-06: base edges of "
            + ",".join(tasks)
            + f" seed {seed} recomputed at the audit "
            "batch size (were batch 4: half scale); CAD and circuit distance rescored from "
            "kept edges"
        )
        cj.write_text(json.dumps(l4c, default=str))
        if l4:
            l4["rescored"] = l4c["rescored"]
            l4p.write_text(json.dumps(l4, default=str))
        n_cells += 1
    print(f"rescored {n_cells} cells")
    print("\n".join(scale_report(runs)))


if __name__ == "__main__":
    main()
