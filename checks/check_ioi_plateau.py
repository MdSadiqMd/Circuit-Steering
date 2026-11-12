"""Why the IOI size curve is flat between 1 % and 2 % (audit-2026-11-06 C16).

For the IOI circuit of each seed: the node sets CircuitKIT keeps at 1 % and 2 %, which of the added
nodes have an in-graph path to `logits`, the clean-vs-corrupt output difference of every added node
at the last position on the curve's own examples, and the patching ratio of the 1 % set, the 2 % set
and the 1 % set plus each added node one at a time (same `evaluate_at` path as the curve).
Writes checks/results/ioi_plateau.json. Run: uv run python checks/check_ioi_plateau.py [seeds=[0,1]]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # before transformer_lens
import torch
from aya_circuits.cellset import fraction_for, mask, realised, summary
from aya_circuits.config import load_config
from aya_circuits.discover import node_names
from aya_circuits.faithfulness import discovery_cfg, evaluate_at
from aya_circuits.model import load_tl
from aya_circuits.registry import resolve_task
from circuitkit.api import _qkv_flags_enabled, _reconstruct_circuit_graph
from neb.mechanistic import last_position_acts

ROOT = Path(__file__).resolve().parent.parent


def graph_at(model, spec, scores, frac, cfg, seed):
    with _qkv_flags_enabled(model):
        return _reconstruct_circuit_graph(
            model,
            {"node_scores": scores},
            discovery_cfg(cfg, spec, seed, "zero", None),
            {"target_sparsity": 1 - frac, "scope": "both"},
            cfg["device"],
        )


def reaches_logits(graph) -> set[str]:
    """Nodes with an in-graph edge path to `logits`."""
    names = node_names(graph)
    in_graph = graph.in_graph.cpu()
    bwd = {}
    for n in names[1:]:
        if n.startswith("MLP "):
            bwd[graph._backward_index(graph.cfg, f"m{n[4:]}")] = n
        else:
            layer, h = n[1:].split(".")
            for qkv in "qkv":
                bwd[graph._backward_index(graph.cfg, f"a{layer}.h{h}", qkv=qkv)] = n
    logits_b = in_graph.shape[1] - 1
    reach, changed = set(), True
    while changed:
        changed = False
        for f, n in enumerate(names):
            if n in reach:
                continue
            cols = in_graph[f].nonzero().flatten().tolist()
            if any(b == logits_b or bwd.get(b) in reach for b in cols):
                reach.add(n)
                changed = True
    return reach


def main() -> None:
    cfg = load_config()
    seeds = cfg.get("seeds_check", [0, 1])
    model = load_tl(cfg)
    spec = resolve_task("ioi", cfg)
    n_curve = cfg["faithfulness"]["curve_n_examples"]
    out = {}
    for seed in seeds:
        d = Path(cfg["out_dir"]) / "circuits" / "ioi" / f"seed{seed}"
        scores = ck.load_scores(str(d / "circuit.pt")).scores
        all_nodes = list(scores)
        sets, graphs = {}, {}
        for frac in (0.01, 0.02):
            g = graph_at(model, spec, scores, frac, cfg, seed)
            names = node_names(g)
            kept = [n for n, k in zip(names, g.nodes_in_graph.tolist(), strict=True) if k]
            sets[frac], graphs[frac] = kept, g
        n1 = [n for n in sets[0.01] if n != "input"]
        n2 = [n for n in sets[0.02] if n != "input"]
        added = [n for n in n2 if n not in n1]
        reach2 = reaches_logits(graphs[0.02])
        # clean vs corrupt output of each added node at the last position (curve examples)
        dl = spec.build_dataloader(
            model, discovery_cfg(cfg, spec, seed, "zero", n_curve), cfg["device"]
        )
        clean, corrupt = [], []
        for batch in dl:
            c, k = batch[0], batch[1]
            clean += list(c) if isinstance(c, (list, tuple)) else model.to_string(c)
            corrupt += list(k) if isinstance(k, (list, tuple)) else model.to_string(k)
        ac, ak = last_position_acts(model, clean, added), last_position_acts(model, corrupt, added)
        diff = {
            n: {
                "rel_diff_last_pos": float(
                    ((ac[n] - ak[n]).norm(dim=-1) / ac[n].norm(dim=-1).clamp_min(1e-6)).mean()
                ),
                "max_abs_diff": float((ac[n] - ak[n]).abs().max()),
            }
            for n in added
        }

        def ev(keep: list[str], all_nodes: list[str] = all_nodes, seed: int = seed) -> dict:
            got = realised(
                model, spec, mask(all_nodes, set(keep)), fraction_for(len(keep)), cfg, seed
            )
            assert got == set(keep), (sorted(got - set(keep)), sorted(set(keep) - got))
            return summary(
                evaluate_at(
                    model,
                    spec,
                    mask(all_nodes, set(keep)),
                    fraction_for(len(keep)),
                    cfg,
                    seed,
                    ["patching"],
                    "zero",
                    1,
                    n_examples=n_curve,
                )
            )

        stored = json.loads((d / "faithfulness.json").read_text())["curve"]
        res = {
            "nodes_1pct": n1,
            "nodes_2pct": n2,
            "added_at_2pct": added,
            "added_with_path_to_logits": [n for n in added if n in reach2],
            "nodes_2pct_with_path_to_logits": sorted(n for n in n2 if n in reach2),
            "edges_1pct": int(graphs[0.01].in_graph.sum()),
            "edges_2pct": int(graphs[0.02].in_graph.sum()),
            "clean_vs_corrupt_last_position": diff,
            "stored_patching": {
                f: stored[f]["zero"]["metadata"]["patching_raw_ratio"] for f in ("0.01", "0.02")
            },
            "patching_1pct_set": ev(n1),
            "patching_2pct_set": ev(n2),
            "patching_1pct_plus_one": {n: ev([*n1, n]) for n in added},
        }
        out[f"seed{seed}"] = res
        print(json.dumps({k: v for k, v in res.items() if "nodes" not in k}, indent=1), flush=True)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    (ROOT / "checks/results/ioi_plateau.json").write_text(json.dumps(out, indent=1))
    print("wrote checks/results/ioi_plateau.json")


if __name__ == "__main__":
    main()
