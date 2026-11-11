"""Brief §12.5: validate every steering direction against patching its full component.

A direction can move the output through a pathway the component itself does not use (Makelov et
al., ICLR 2024, subspace illusion). For each main steering cell (no sweep, no `safetune_rd`), on
source-side prompts and at the cell's nodes, at the last prompt position:
  full       the node's activation replaced by its mean over target-side prompts (component patch)
  direction  only the component along v̂ (the cell's normalised vector) replaced by the target
             mean's component; the orthogonal part is kept (subspace patch)
  steer      the cell itself (coefficient·v added at every position), for reference
Metric: B2 = logit(refusal onset "I'm") − logit(comply onset "In") on the 28 benign near-miss
prompts; B1 = logit(gold first token) − logit(base top-1) on the seed's held-out wrong facts (A6
keeps them out of the vector's sources). Effects are mean metric changes against the unpatched
model. A subspace illusion shows up as a direction effect larger than, or opposite in sign to,
the full-component effect. Writes checks/results/steer_direction.json.
Run: uv run python checks/check_steer_direction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap, load_tl
from circuitkit.applications.steering import ActivationSteering

from interventions.cell import Cell, apply_tl
from interventions.run_grid import b1_folds, b2_split
from interventions.steer import last_position_means

ROOT = Path(__file__).resolve().parent.parent
REFUSE, COMPLY = 15817, 1056  # data/b2/eap_en.csv correct_idx / incorrect_idx ("I'm" / "In")


@torch.no_grad()
def metric(model, texts, a_idx, b_idx, hooks=(), bs: int = 8) -> torch.Tensor:
    """logit[a] − logit[b] at the last position; a_idx/b_idx: one id or one per prompt."""
    out = []
    for i in range(0, len(texts), bs):
        toks = model.to_tokens(texts[i : i + bs], prepend_bos=False, padding_side="left")
        logits = model.run_with_hooks(toks, fwd_hooks=list(hooks))[:, -1].float()
        n = toks.shape[0]
        a = torch.as_tensor(a_idx[i : i + n] if isinstance(a_idx, list) else [a_idx] * n)
        b = torch.as_tensor(b_idx[i : i + n] if isinstance(b_idx, list) else [b_idx] * n)
        r = torch.arange(n)
        out.append((logits[r, a.to(logits.device)] - logits[r, b.to(logits.device)]).cpu())
    return torch.cat(out)


def patch_hooks(model, nodes, target_mean, vecs, mode: str):
    """Hooks patching each node at the last position (left padding: -1 is the prompt end)."""
    helper = ActivationSteering(model, dict.fromkeys(nodes, 1.0))
    hooks = []
    for n in nodes:
        name = helper._get_hook_point_from_node(n)
        head = None if n.startswith("MLP ") else helper._node_head_index(n)
        tgt = target_mean[n]
        u = vecs[n].float() / vecs[n].float().norm().clamp_min(1e-8)

        def hook(act, hook, tgt=tgt, u=u, head=head):
            x = act[:, -1] if head is None else act[:, -1, head]
            t, uu = tgt.to(x.device, torch.float32), u.to(x.device, torch.float32)
            xf = x.float()
            new = t.expand_as(xf) if mode == "full" else xf + ((t @ uu) - (xf @ uu))[:, None] * uu
            if head is None:
                act[:, -1] = new.to(act.dtype)
            else:
                act[:, -1, head] = new.to(act.dtype)
            return act

        hooks.append((name, hook))
    return hooks


def main() -> None:
    cfg = load_config()
    runs = ROOT / cfg["out_dir"]
    model = load_tl(cfg)
    tok = model.tokenizer
    ev = [json.loads(line) for line in (ROOT / "data/b1/model_eval_en.jsonl").open()]
    known = [r["text"] for r in ev if r["known"]][:64]
    folds = b1_folds(cfg)
    train_h, _, benign = b2_split(cfg)
    b2_src = [chat_wrap(tok, r["en"], cfg["chat_format"]) for r in benign][:64]
    b2_tgt = [
        chat_wrap(tok, r["en"], cfg["chat_format"]) for r in train_h if r["gen"]["en"]["refused"]
    ][:64]
    out = {}
    for beh in ("b1", "b2"):
        for cj in sorted((runs / "interventions" / beh / "steer").glob("*/cell.json")):
            cell, tensors = Cell.load(cj.parent)
            if "@" in cell.condition or not cell.nodes:
                continue
            seed = cell.seed if cell.seed in cfg["seeds"] else cfg["seeds"][cell.seed % 3]
            if beh == "b1":
                keys = {(f["relation"], f["subject_qid"]) for f in folds[seed]}
                rows = [r for r in ev if (r["relation"], r["subject_qid"]) in keys]
                src, tgt = [r["text"] for r in rows], known
                a, b = [r["label"] for r in rows], [r["top1_id"] for r in rows]
            else:
                src, tgt, a, b = b2_src, b2_tgt, REFUSE, COMPLY
            vecs = {k[4:]: v for k, v in tensors.items() if k.startswith("vec:")}
            tmean = last_position_means(model, tgt, cell.nodes)
            base = metric(model, src, a, b)
            res = {"n_prompts": len(src), "n_nodes": len(cell.nodes)}
            for mode in ("full", "direction"):
                m = metric(model, src, a, b, patch_hooks(model, cell.nodes, tmean, vecs, mode))
                res[f"{mode}_effect"] = float((m - base).mean())
            undo = apply_tl(model, cell, tensors)
            try:
                res["steer_effect"] = float((metric(model, src, a, b) - base).mean())
            finally:
                undo()
            f, d = res["full_effect"], res["direction_effect"]
            res["direction_over_full"] = d / f if abs(f) > 1e-6 else None
            # illusion signature: the subspace patch moves the metric the other way, or further
            res["illusion_flag"] = bool((f * d < 0) or abs(d) > 1.1 * abs(f))
            out[cell.name] = res
            print(cell.name, json.dumps(res), flush=True)
    flagged = [k for k, v in out.items() if v["illusion_flag"]]
    (ROOT / "checks/results/steer_direction.json").write_text(
        json.dumps({"flagged": flagged, "cells": out}, indent=1)
    )
    print(f"{len(flagged)}/{len(out)} steering cells flagged")


if __name__ == "__main__":
    main()
