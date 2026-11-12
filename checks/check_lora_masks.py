"""The LoRA cells' restriction equals CircuitKIT's selective-finetuning selection (brief §5:
CircuitKIT supplies "selective LoRA masks"; interventions/lora.py builds the masks itself).

For every LoRA cell except `unrestricted`:
  1. The selection implied by the cell's node list (q rows of each head, k/v rows of its GQA group,
     o columns, MLP layers) and, for the pre-registered per-type cells, CircuitKIT's
     `select_components` (behind `ck.selective_finetune`, node level, scope both) on 0/1 scores of
     the node set, which applies the same per-type top-10 % rule; for `circuit` cells the latter
     also runs on the discovery's signed scores at top_frac 0.10 and must select the cell's nodes.
     The A15 joint sets (40 heads + 21 MLPs) are not a per-type 10 %, so only the node-list
     selection applies to them.
  2. The trained factors in delta.safetensors are read back: nonzero rows of B (q/k/v), nonzero
     columns of A (o_proj) and the MLP modules that carry adapters.
  3. They must match exactly. Writes checks/results/lora_masks.json; exit 1 on any mismatch. CPU
     only.
Run: uv run python checks/check_lora_masks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuitkit.applications.selective_finetuning.selector import select_components
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
N_LAYERS, N_Q, N_KV, HEAD_DIM = 36, 16, 4, 128  # Cohere2 tiny-aya-global config (stack_check.json)


def ck_selection(head_scores: dict, mlp_scores: dict, top_frac: float) -> dict:
    sel = select_components(
        head_scores, mlp_scores, {"level": "node"}, top_frac, "both", N_LAYERS, N_Q, N_KV, HEAD_DIM
    )
    attn = {int(k[5:]): {p: set(v) for p, v in d.items()} for k, d in sel.attn.items()}
    return {"attn": attn, "mlp": {int(k[4:]) for k in sel.mlp}}


def indicator(nodes: list[str]) -> tuple[dict, dict]:
    keep = set(nodes)
    heads = {(lay, h): float(f"A{lay}.{h}" in keep) for lay in range(N_LAYERS) for h in range(N_Q)}
    mlps = {lay: float(f"MLP {lay}" in keep) for lay in range(N_LAYERS)}
    return heads, mlps


def from_nodes(nodes: list[str]) -> dict:
    """The selection the node list implies (interventions/lora.py `attach`): q rows and o columns
    of each head, k/v rows of the head's GQA group (4 query heads per KV head), whole MLP layers."""
    attn: dict[int, dict[str, set]] = {}
    mlp = set()
    group = N_Q // N_KV
    for n in nodes:
        if n.startswith("MLP "):
            mlp.add(int(n[4:]))
            continue
        layer, h = (int(x) for x in n[1:].split("."))
        d = attn.setdefault(layer, {"q": set(), "k": set(), "v": set(), "o": set()})
        rows = set(range(h * HEAD_DIM, (h + 1) * HEAD_DIM))
        kv = set(range((h // group) * HEAD_DIM, (h // group + 1) * HEAD_DIM))
        d["q"] |= rows
        d["o"] |= rows
        d["k"] |= kv
        d["v"] |= kv
    return {"attn": attn, "mlp": mlp}


def from_factors(t: dict) -> dict:
    attn: dict[int, dict[str, set]] = {}
    mlp = set()
    for k, v in t.items():
        if not k.startswith(("A:", "B:")):
            continue
        mod = k[2:]
        layer = int(mod.split(".")[2])
        proj = mod.rsplit(".", 1)[1]
        if proj in ("gate_proj", "up_proj", "down_proj"):
            mlp.add(layer)
        elif k.startswith("B:") and proj in ("q_proj", "k_proj", "v_proj"):
            attn.setdefault(layer, {})[proj[0]] = set(v.abs().sum(1).nonzero().flatten().tolist())
        elif k.startswith("A:") and proj == "o_proj":
            attn.setdefault(layer, {})["o"] = set(v.abs().sum(0).nonzero().flatten().tolist())
    return {"attn": attn, "mlp": mlp}


def main() -> None:
    out, bad = {}, []
    for cj in sorted((ROOT / "runs/interventions").glob("*/lora/*/cell.json")):
        cell = json.loads(cj.read_text())
        if cell["condition"] == "unrestricted" or cell["condition"].startswith("perturb_"):
            continue
        name = f"{cell['behaviour']}_lora_{cell['condition']}_s{cell['seed']}"
        got = from_factors(load_file(str(cj.parent / "delta.safetensors")))
        ok = from_nodes(cell["nodes"]) == got
        rec = {"match": ok, "n_attn_layers": len(got["attn"]), "n_mlp_layers": len(got["mlp"])}
        if not cell["condition"].endswith("_joint"):
            expected = ck_selection(*indicator(cell["nodes"]), 0.10)
            rec["ck_selection_matches"] = expected == got
            ok = ok and rec["ck_selection_matches"]
        if cell["condition"] == "circuit":
            seed = cell["seed"]
            nodes = json.loads(
                (ROOT / f"runs/circuits/{cell['behaviour']}_en/seed{seed}/nodes.json").read_text()
            )
            signed = dict(zip(nodes["names"], nodes["signed_rowsum"], strict=True))
            hs = {(lay, h): signed[f"A{lay}.{h}"] for lay in range(N_LAYERS) for h in range(N_Q)}
            ms = {lay: signed[f"MLP {lay}"] for lay in range(N_LAYERS)}
            rec["ck_selects_cell_nodes"] = ck_selection(hs, ms, 0.10) == expected
            ok = ok and rec["ck_selects_cell_nodes"]
        out[name] = rec
        if not ok:
            bad.append(name)
        print(name, rec, flush=True)
    (ROOT / "checks/results/lora_masks.json").write_text(
        json.dumps({"n_cells": len(out), "mismatches": bad, "cells": out}, indent=1)
    )
    print(f"{len(out) - len(bad)}/{len(out)} LoRA cells match CircuitKIT's selection")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
