"""Validation of neb.attribution (amendment A19) on a small model, CPU fp32, no MPS budget touched.

1. Equivalence: with the right-endpoint Riemann rule the per-example mean equals CircuitKIT's
   `attribute(method="EAP-IG-inputs")` score times the batch size (A13 scaling), up to accumulation
   precision; the clean loss vector's mean equals the metric's batch mean.
2. Quadrature: relative L2 distance of every 3- and 6-pass rule (right-endpoint Riemann, composite
   midpoint, Gauss–Legendre, Richardson 2·S₆ − S₃) to a 48-step composite-midpoint reference on the
   same examples, with the spread between the fine rules (GL-15, Riemann-48, midpoint-48) as the
   reference's own uncertainty, so the integration error of a cheap rule is measured, not assumed.
   On gpt2 IOI Gauss–Legendre-3 is *worse* than Riemann-3 (the integrand is not smooth along the
   path), which is why amendment A19 seals a measured selection instead of a rule.
Writes checks/results/quadrature.json.
Run: uv run python checks/check_quadrature.py [model=gpt2 n=16]
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.model import load_tl
from aya_circuits.registry import resolve_task
from aya_circuits.tasks import logit_diff, register_csv_task
from circuitkit.backends.eap.attribute import attribute
from circuitkit.backends.eap.graph import Graph
from neb.attribution import RULES, SIX_PASS, THREE_PASS, eap_ig_inputs
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def gpt2_pairs(tok, n: int) -> Path:
    """IOI-style pairs for gpt2 (ABBA clean, ABC corrupt, single-token names, equal length), as a
    CSV in the task format (clean, corrupted, correct_idx, incorrect_idx); no BOS in the text."""
    import csv
    import itertools

    def ids(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)  # TL's tokenizer prepends BOS otherwise

    names = [
        x
        for x in ("Mary", "John", "Tom", "Anne", "Paul", "Kate", "Mark", "Lucy", "James", "Sara")
        if len(ids(" " + x)) == 1
    ]
    places = ["store", "park", "office", "school"]
    objects = ["drink", "book", "ring", "kiss"]
    rows = []
    for (a, b), c, place, obj in zip(
        itertools.permutations(names, 2),
        itertools.cycle(names[2:]),
        itertools.cycle(places),
        itertools.cycle(objects),
        strict=False,
    ):
        if c in (a, b):
            continue
        clean = f"When {a} and {b} went to the {place}, {b} gave a {obj} to"
        corrupt = f"When {a} and {b} went to the {place}, {c} gave a {obj} to"
        if len(ids(clean)) != len(ids(corrupt)):
            continue
        rows.append(
            {
                "clean": clean,
                "corrupted": corrupt,
                "correct_idx": ids(" " + a)[0],
                "incorrect_idx": ids(" " + b)[0],
            }
        )
        if len(rows) == max(n, 8) * 2:
            break
    p = ROOT / ".cache" / "smoke" / "ioi_small_gpt2.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"])
        w.writeheader()
        w.writerows(rows)
    return p


def main() -> None:
    cfg = load_config()
    name, n, bs = str(cfg.get("model", "gpt2")), int(cfg.get("n", 16)), int(cfg.get("bs", 4))
    cfg = cfg | {"model": name, "device": str(cfg.get("device", "cpu")), "dtype": "float32"}
    model = load_tl(cfg)
    model.cfg.default_prepend_bos = False
    task = str(cfg.get("task", "ioi_fixed"))
    spec = (
        register_csv_task("ioi_small", gpt2_pairs(model.tokenizer, n))
        if name.startswith("gpt2")
        else resolve_task(task, cfg)
    )
    dcfg = {
        "algorithm": "eap-ig",
        "task": spec.name,
        "level": "node",
        "batch_size": bs,
        "data_params": {"num_examples": n, "seed": 0},
        "seed": 0,
        "cache_dir": cfg["cache_dir"],
    }
    loss_vec = partial(logit_diff, loss=True, mean=False)
    out = {"model": name, "n_examples": n, "batch_size": bs}
    # 1. the library's score with the batch-mean metric (Riemann, 3 steps)
    graph = Graph.from_model(model, node_scores=True)
    attribute(
        model,
        graph,
        spec.build_dataloader(model, dcfg, cfg["device"]),
        spec.metric_fn(),
        method="EAP-IG-inputs",
        ig_steps=3,
        quiet=True,
    )
    lib = graph.scores.detach().float().cpu() * bs
    ref_name = str(cfg.get("reference", "midpoint48"))
    rules = {k: RULES[k] for k in (*THREE_PASS, *SIX_PASS, "gl15", "riemann48", "midpoint48")}
    means, perex, losses = {}, {}, {}
    for key, (nodes, weights) in rules.items():
        s, lc, lk = eap_ig_inputs(
            model,
            Graph.from_model(model, node_scores=True),
            spec.build_dataloader(model, dcfg, cfg["device"]),
            loss_vec,
            nodes,
            weights,
        )
        means[key], perex[key], losses[key] = s.mean(0), s, (lc, lk)
        if key == "riemann3":
            out["clean_loss_mean"], out["corrupt_loss_mean"] = float(lc.mean()), float(lk.mean())
    real = graph.real_edge_mask.cpu()
    out["riemann3_vs_library"] = {
        "rel_l2": rel_l2(means["riemann3"][real], lib[real]),
        "spearman": float(spearmanr(means["riemann3"][real], lib[real]).statistic),
        "max_abs": float((means["riemann3"][real] - lib[real]).abs().max()),
        "scale": float(means["riemann3"][real].norm() / lib[real].norm()),
    }
    ref = means[ref_name][real]
    out["reference"] = ref_name
    out[
        "reference_spread"
    ] = {  # how far the fine rules are from each other: the reference's own error
        k: rel_l2(means[k][real], ref) for k in ("gl15", "riemann48", "midpoint48") if k != ref_name
    }
    out["vs_reference"] = {
        k: {
            "rel_l2": rel_l2(means[k][real], ref),
            "spearman": float(spearmanr(means[k][real], ref).statistic),
        }
        for k in (*THREE_PASS, *SIX_PASS)
    }
    rich = 2 * means["riemann6"][real] - means["riemann3"][real]
    out["vs_reference"]["richardson_6_3"] = {
        "rel_l2": rel_l2(rich, ref),
        "spearman": float(spearmanr(rich, ref).statistic),
    }
    out["best_three_pass"] = min(THREE_PASS, key=lambda k: out["vs_reference"][k]["rel_l2"])
    out["best_six_pass"] = min(SIX_PASS, key=lambda k: out["vs_reference"][k]["rel_l2"])
    # IG completeness per rule: Σ_dst s_j(input→dst) must equal loss_corrupt_j − loss_clean_j for an
    # exact integral, so the relative gap is a reference-free measure of the quadrature error
    input_idx = graph.forward_index(graph.nodes["input"])
    lc, lk = losses["riemann3"]
    out["completeness_rel_gap"] = {
        k: float(
            (perex[k][:, input_idx, :].sum(1) - (lk - lc)).abs().mean() / (lk - lc).abs().mean()
        )
        for k in rules
    }
    # per-example scores are a decomposition of the mean (exact by construction) and vary across
    # examples (otherwise the jackknife would be vacuous)
    pe = perex["gl3"]
    out["per_example"] = {
        "mean_reconstruction_max_abs": float((pe.mean(0) - means["gl3"]).abs().max()),
        "between_example_rel_sd": float(pe[:, real].std(0).mean() / pe[:, real].abs().mean()),
    }
    (ROOT / "checks/results/quadrature.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    assert out["riemann3_vs_library"]["spearman"] > 0.99, (
        "per-example mean disagrees with CircuitKIT"
    )
    assert 0.9 < out["riemann3_vs_library"]["scale"] < 1.1, "scale differs from library × batch"
    assert out["per_example"]["mean_reconstruction_max_abs"] < 1e-5
    print("quadrature check OK")


if __name__ == "__main__":
    main()
