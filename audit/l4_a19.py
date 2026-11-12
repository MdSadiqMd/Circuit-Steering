"""Amendment A19: per-example EAP-IG with a measured quadrature rule on the base model and the H2
cells of both behaviours, plus the attribution-free bystander measure
(docs/countering-limitations.md item 3; the sealed rules are in docs/preregistration.md A19).

Every attribution goes through `neb.attribution.eap_ig_inputs` (CircuitKIT's hooks and index layout,
fp32 accumulation, scores kept per example). Three resumable stages:
  stage=validate  base model, discovery seed 0, the four A19 tasks, every rule in `RULES` (3-pass,
                  6-pass, 48-step references): relative L2 to the midpoint-48 reference, the mean
                  relative completeness gap (Σ of the input node's edge scores must equal
                  loss(corrupt) − loss(clean) for an exact integral), and the agreement of the
                  Riemann-3 per-example mean with CircuitKIT's `attribute` (× batch size, A13).
                  Writes checks/results/quadrature_tinyaya.json and the sealed choice to
                  runs/audit/a19/rule.json.
  stage=base      per discovery seed: per-example scores and clean / corrupt loss vectors of the
                  four tasks under the chosen rule -> runs/audit/a19/base/<task>_s<seed>.safetensors
  stage=cells     every main cell of the A19 conditions (grid.behaviour, optional cells= filter):
                  the cell applied with interventions.cell.apply_tl, per-example scores on the
                  union of the three circuit masks and the base's top-1 % edges, the full mean and
                  the loss vectors -> runs/audit/<beh>/<cell>/a19.safetensors + a19.json. Exits 3
                  after `audit.cells_per_process` new cells (MPS allocator growth); chains loop.
Tasks: the behaviour's target, its two non-IOI bystanders and `ioi_fixed` (the built-in `ioi` is
dropped, A16). Examples: the audit's 32-example subsets per task and seed (same seed, same rows).
Run: uv run python audit/l4_a19.py stage=validate | stage=base | stage=cells grid.behaviour=b1
"""

from __future__ import annotations

import gc
import json
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_tl
from aya_circuits.nodes import circuit_nodes
from aya_circuits.registry import resolve_task
from aya_circuits.tasks import logit_diff
from circuitkit.backends.eap.attribute import attribute
from circuitkit.backends.eap.graph import Graph
from neb import mechanistic as M
from neb.attribution import RULES, SIX_PASS, THREE_PASS, eap_ig_inputs
from safetensors.torch import load_file, save_file
from scipy.stats import spearmanr

from audit.l4 import BYSTANDERS, ROOT, disc_seed_of, l4_cfg
from interventions.cell import Cell, apply_tl

CONDITIONS = {
    "circuit",
    "random",
    "inverse",
    "taylor",
    "unrestricted",
    "circuit_joint",
    "random_joint",
    "inverse_joint",
}
REFERENCE = "midpoint48"
LOSS_VEC = partial(logit_diff, loss=True, mean=False)


def tasks_for(beh: str) -> list[str]:
    return [f"{beh}_en"] + [b for b in BYSTANDERS[beh] if b != "ioi"] + ["ioi_fixed"]


def a19_root(cfg: dict) -> Path:
    return Path(cfg["out_dir"]) / "audit" / "a19"


def attribute_task(model, cfg: dict, task: str, seed: int, rule: str):
    spec = resolve_task(task, cfg)
    nodes, weights = RULES[rule]
    dl = spec.build_dataloader(model, l4_cfg(cfg, spec, seed), cfg["device"])
    return eap_ig_inputs(
        model, Graph.from_model(model, node_scores=True), dl, LOSS_VEC, nodes, weights
    )


def completeness(scores: torch.Tensor, clean_loss, corrupt_loss, input_idx: int) -> dict:
    """Per-example IG completeness: Σ_dst s_j(input→dst) − (loss_corrupt_j − loss_clean_j)."""
    gap = scores[:, input_idx, :].sum(1) - (corrupt_loss - clean_loss)
    return {
        "mean_abs_gap": float(gap.abs().mean()),
        "rel_gap": float(gap.abs().mean() / (corrupt_loss - clean_loss).abs().mean()),
    }


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def validate(model, cfg: dict) -> None:
    """Stage 0: measure every rule on the base model (seed 0) and seal the choice."""
    out_p = ROOT / "checks/results/quadrature_tinyaya.json"
    rule_p = a19_root(cfg) / "rule.json"
    if rule_p.exists() and not cfg.get("force"):
        print("rule already sealed:", json.loads(rule_p.read_text())["rule"])
        return
    seed = cfg["seeds"][0]
    tasks = sorted(set(tasks_for("b1")) | set(tasks_for("b2")))
    graph = Graph.from_model(model, node_scores=True)
    real, input_idx = graph.real_edge_mask.cpu(), graph.forward_index(graph.nodes["input"])
    res: dict = {"seed": seed, "tasks": {}, "reference": REFERENCE}
    for task in tasks:
        t0 = time.time()
        spec = resolve_task(task, cfg)
        g = Graph.from_model(model, node_scores=True)
        attribute(
            model,
            g,
            spec.build_dataloader(model, l4_cfg(cfg, spec, seed), cfg["device"]),
            spec.metric_fn(),
            method="EAP-IG-inputs",
            ig_steps=3,
            quiet=True,
        )
        lib = g.scores.detach().float().cpu() * cfg["audit"]["l4"]["batch_size"]
        means, comp = {}, {}
        for rule in RULES:
            s, lc, lk = attribute_task(model, cfg, task, seed, rule)
            means[rule] = s.mean(0)
            comp[rule] = completeness(s, lc, lk, input_idx)
        ref = means[REFERENCE][real]
        res["tasks"][task] = {
            "riemann3_vs_library": {
                "rel_l2": rel_l2(means["riemann3"][real], lib[real]),
                "spearman": float(spearmanr(means["riemann3"][real], lib[real]).statistic),
            },
            "reference_spread": {
                k: rel_l2(means[k][real], ref) for k in ("gl15", "riemann48") if k != REFERENCE
            },
            "rules": {
                k: {
                    "rel_l2_vs_reference": rel_l2(means[k][real], ref),
                    "spearman_vs_reference": float(spearmanr(means[k][real], ref).statistic),
                    **comp[k],
                }
                for k in RULES
            },
            "wall_s": round(time.time() - t0),
        }
        print(task, json.dumps(res["tasks"][task]), flush=True)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def mean_err(rule: str) -> float:
        return sum(res["tasks"][t]["rules"][rule]["rel_l2_vs_reference"] for t in tasks) / len(
            tasks
        )

    best3 = min(THREE_PASS, key=mean_err)
    best6 = min(SIX_PASS, key=mean_err)
    chosen = best6 if mean_err(best6) < 0.5 * mean_err(best3) else best3
    res["selection"] = {
        "best_three_pass": best3,
        "best_six_pass": best6,
        "mean_rel_l2": {k: mean_err(k) for k in (*THREE_PASS, *SIX_PASS)},
        "rule": chosen,
        "criterion": "A19: best 3-pass rule by mean relative L2 to midpoint-48 over the four "
        "tasks; the best 6-pass rule instead if its error is below half of that",
    }
    out_p.write_text(json.dumps(res, indent=1))
    rule_p.parent.mkdir(parents=True, exist_ok=True)
    nodes, weights = RULES[chosen]
    rule_p.write_text(
        json.dumps({"rule": chosen, "nodes": nodes, "weights": weights, "sealed_from": str(out_p)})
    )
    print("sealed rule:", chosen, json.dumps(res["selection"]["mean_rel_l2"]))


def sealed_rule(cfg: dict) -> str:
    return json.loads((a19_root(cfg) / "rule.json").read_text())["rule"]


def base(model, cfg: dict) -> None:
    """Stage 1: per-example base scores and loss vectors for every seed and A19 task."""
    rule = sealed_rule(cfg)
    out = a19_root(cfg) / "base"
    out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(set(tasks_for("b1")) | set(tasks_for("b2")))
    for seed in cfg["seeds"]:
        for task in tasks:
            p = out / f"{task}_s{seed}.safetensors"
            if p.exists():
                continue
            t0 = time.time()
            s, lc, lk = attribute_task(model, cfg, task, seed, rule)
            save_file(
                {"scores": s, "clean_loss": lc, "corrupt_loss": lk}, str(p), metadata={"rule": rule}
            )
            print(f"base {task} s{seed} {time.time() - t0:.0f}s", flush=True)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


def union_mask(
    graph: Graph, runs: Path, task: str, seed: int, top: float, base_mean
) -> torch.Tensor:
    """Per-type ∪ joint ∪ pillar induced edges ∪ the base's top-1 % edges, as a boolean [F, B]."""
    nodes = json.loads((runs / "circuits" / task / f"seed{seed}" / "nodes.json").read_text())
    signed = dict(zip(nodes["names"], nodes["signed_rowsum"], strict=True))
    signed.pop("input", None)
    k = max(1, round(len(signed) * top))
    joint = sorted(signed, key=lambda n: abs(signed[n]), reverse=True)[:k]
    pillar = json.loads((runs / "circuits" / task / f"seed{seed}" / "cellset.json").read_text())[
        "pillar_set"
    ]["nodes"]
    m = M.induced_edge_mask(graph, circuit_nodes(signed, top))
    m |= M.induced_edge_mask(graph, joint)
    m |= M.induced_edge_mask(graph, pillar)
    m |= M.top_edges_mask(base_mean, graph, 0.01)
    return m


def cells(model, cfg: dict) -> None:
    """Stage 2: the A19 cells of one behaviour."""
    beh, rule = cfg["grid"]["behaviour"], sealed_rule(cfg)
    runs, top = Path(cfg["out_dir"]), cfg["discovery"]["top_fraction"]
    want = set(str(cfg.get("cells", "steer,lora,edit")).split(","))
    tasks = tasks_for(beh)
    todo = []
    for cj in sorted((runs / "interventions" / beh).glob("*/*/cell.json")):
        c = Cell.peek(cj.parent)
        if (
            c.intervention in want
            and c.condition in CONDITIONS
            and "@" not in c.condition
            and not (runs / "audit" / beh / c.name / "a19.json").exists()
        ):
            todo.append(cj.parent)
    if cfg.get("max_cells"):
        todo = todo[: int(cfg["max_cells"])]
    print(f"{len(todo)} cells for {beh} (rule {rule})", flush=True)
    graph = Graph.from_model(model, node_scores=True)
    mw, cap = MemWatch(cfg["memory_budget_gb"]), cfg["audit"].get("cells_per_process")
    masks: dict[tuple[str, int], torch.Tensor] = {}
    bases: dict[tuple[str, int], dict] = {}
    for built, d in enumerate(todo, 1):
        cell, tensors = Cell.load(d)
        seed = disc_seed_of(cell, cfg)
        for task in tasks:
            if (task, seed) not in bases:
                t = load_file(str(a19_root(cfg) / "base" / f"{task}_s{seed}.safetensors"))
                # the mean and the loss vectors are all this stage needs; the 138 MB per-example
                # tensor is left to the analysis
                bases[task, seed] = {"mean": t["scores"].mean(0), "clean_loss": t["clean_loss"]}
                masks[task, seed] = union_mask(
                    graph, runs, task, seed, top, bases[task, seed]["mean"]
                )
        t0 = time.time()
        out, meta = {}, {"disc_seed": seed, "rule": rule, "tasks": {}}
        undo = apply_tl(model, cell, tensors)
        try:
            for task in tasks:
                s, lc, lk = attribute_task(model, cfg, task, seed, rule)
                b, m = bases[task, seed], masks[task, seed]
                idx = m.flatten().nonzero().flatten()
                out[task] = s.mean(0)
                out[f"{task}_perex"] = s.reshape(s.shape[0], -1)[:, idx].contiguous()
                out[f"{task}_idx"] = idx
                out[f"{task}_clean_loss"], out[f"{task}_corrupt_loss"] = lc, lk
                signed = json.loads(
                    (runs / "circuits" / task / f"seed{seed}" / "nodes.json").read_text()
                )
                signed = dict(zip(signed["names"], signed["signed_rowsum"], strict=True))
                signed.pop("input", None)
                m_cell = M.induced_edge_mask(graph, circuit_nodes(signed, top))
                meta["tasks"][task] = {
                    "n_examples": int(s.shape[0]),
                    "n_union_edges": int(idx.numel()),
                    "cad_cell_set": M.cad(b["mean"], out[task], m_cell)["cad"],
                    "clean_loss_delta": float((lc - b["clean_loss"]).mean()),
                    "clean_acc_delta": float(
                        (lc < 0).float().mean() - (b["clean_loss"] < 0).float().mean()
                    ),
                }
        finally:
            undo()
        cell_dir = runs / "audit" / beh / cell.name
        cell_dir.mkdir(parents=True, exist_ok=True)
        save_file(out, str(cell_dir / "a19.safetensors"), metadata={"rule": rule})
        (cell_dir / "a19.json").write_text(json.dumps(meta))
        t = meta["tasks"][tasks[0]]
        print(
            f"{cell.name} a19 {time.time() - t0:.0f}s target cad {t['cad_cell_set']:.3f} "
            f"Δloss {t['clean_loss_delta']:+.3f}",
            flush=True,
        )
        mw.check()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if cap and built >= cap:
            print(
                f"stopping after {built} cells (audit.cells_per_process); rerun to continue",
                flush=True,
            )
            raise SystemExit(3)


def main() -> None:
    cfg = load_config()
    stage = str(cfg.get("stage", "cells"))
    model = load_tl(cfg)
    {"validate": validate, "base": base, "cells": cells}[stage](model, cfg)


if __name__ == "__main__":
    main()
