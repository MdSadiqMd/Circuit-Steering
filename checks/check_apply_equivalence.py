"""Verifies that applying the same cell to the HF model and to the TransformerLens model changes the
logits identically (weight layout mapping in interventions/cell.py and steering hook placement).
Random deltas per cell type; the two bf16 models already differ by ~0.25 logit max-abs over the
262k-entry
output with no delta, so the criterion is the cosine between the two logit-delta vectors (a wrong
layout gives ≈ 0) plus an effect well above that floor, and the max-abs disagreement is reported.
Amendment A18 (LoRA applied as low-rank hooks): the same comparison on real LoRA cells, the logit
difference between the hook and the bf16-merged weights it replaces, and a module-level check that
the hook reproduces PEFT's training-time forward (bf16 base output + fp32 term, one rounding)
exactly while the merged weight does not.
Run: CIRCUITKIT_TL_MEMORY_PATCH=1 uv run python checks/check_apply_equivalence.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401
import torch
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap, load_hf, load_tl

from interventions.cell import (
    Cell,
    _get_module,
    add_term,
    apply_hf,
    apply_tl,
    lora_deltas,
    lora_factors,
    lora_term,
)


@torch.no_grad()
def logits_hf(model, tok, text):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    return model(**enc).logits[0, -1].float().cpu()


@torch.no_grad()
def logits_tl(model, text):
    return model(model.to_tokens(text, prepend_bos=False))[0, -1].float().cpu()


def compare(lh, lt, base_h, base_t) -> dict:
    dh, dt = lh - base_h, lt - base_t
    return {
        "delta_hf": float(dh.abs().max()),
        "delta_tl": float(dt.abs().max()),
        "hf_vs_tl": float((dh - dt).abs().max()),
        "cosine": float(torch.nn.functional.cosine_similarity(dh, dt, dim=0)),
    }


def main() -> None:
    cfg = load_config()
    hf, tok = load_hf(cfg)
    tl = load_tl(cfg)
    text = chat_wrap(
        tok,
        "Which country is the following city in: Lyon? Answer with only the country name.",
        cfg["chat_format"],
    )
    base_h, base_t = logits_hf(hf, tok, text), logits_tl(tl, text)
    print("base hf-tl max diff", float((base_h - base_t).abs().max()))
    g = torch.Generator().manual_seed(0)
    d, dm, dh, nh = 2048, 11008, 128, 16
    report = {}
    # LoRA-style deltas on layer 12: q (head 3 rows), k (kv group 0), o (head 3 cols), down_proj
    layer = 12
    A_q = torch.randn(4, d, generator=g) * 0.3
    B_q = torch.zeros(nh * dh, 4)
    B_q[3 * dh : 4 * dh] = torch.randn(dh, 4, generator=g) * 0.3
    A_o = torch.zeros(4, nh * dh)
    A_o[:, 3 * dh : 4 * dh] = torch.randn(4, dh, generator=g) * 0.3
    B_o = torch.randn(d, 4, generator=g) * 0.3
    A_dn = torch.randn(4, dm, generator=g) * 0.15
    B_dn = torch.randn(d, 4, generator=g) * 0.15
    tensors = {
        "scaling": torch.tensor(2.0),
        f"A:model.layers.{layer}.self_attn.q_proj": A_q,
        f"B:model.layers.{layer}.self_attn.q_proj": B_q,
        f"A:model.layers.{layer}.self_attn.o_proj": A_o,
        f"B:model.layers.{layer}.self_attn.o_proj": B_o,
        f"A:model.layers.{layer}.mlp.down_proj": A_dn,
        f"B:model.layers.{layer}.mlp.down_proj": B_dn,
    }
    cell = Cell("b1", "lora", "circuit", 0, ["A12.3", "MLP 12"])
    u = apply_hf(hf, cell, tensors)
    lh = logits_hf(hf, tok, text)
    u()
    u = apply_tl(tl, cell, tensors)
    lt = logits_tl(tl, text)
    u()
    report["lora"] = compare(lh, lt, base_h, base_t)
    assert float((logits_hf(hf, tok, text) - base_h).abs().max()) < 1e-3, "undo failed (hf)"
    assert float((logits_tl(tl, text) - base_t).abs().max()) < 1e-3, "undo failed (tl)"
    # LoRA deltas on the projections the case above leaves out: k and v of KV group 1 (GQA layout,
    # 4 query heads per KV head) and gate/up of the MLP, layer 7
    layer = 7
    tensors = {"scaling": torch.tensor(2.0)}
    for proj, rows in (("k_proj", 4 * dh), ("v_proj", 4 * dh)):
        B = torch.zeros(rows, 4)
        B[dh : 2 * dh] = torch.randn(dh, 4, generator=g) * 0.3
        tensors[f"A:model.layers.{layer}.self_attn.{proj}"] = torch.randn(4, d, generator=g) * 0.3
        tensors[f"B:model.layers.{layer}.self_attn.{proj}"] = B
    for proj in ("gate_proj", "up_proj"):
        tensors[f"A:model.layers.{layer}.mlp.{proj}"] = torch.randn(4, d, generator=g) * 0.15
        tensors[f"B:model.layers.{layer}.mlp.{proj}"] = torch.randn(dm, 4, generator=g) * 0.15
    cell = Cell("b1", "lora", "circuit", 0, ["A7.5", "MLP 7"])
    u = apply_hf(hf, cell, tensors)
    lh = logits_hf(hf, tok, text)
    u()
    u = apply_tl(tl, cell, tensors)
    lt = logits_tl(tl, text)
    u()
    report["lora_kv_gate_up"] = compare(lh, lt, base_h, base_t)
    # edit-style rank-2 delta on W_out of layer 20
    U = torch.randn(dm, 2, generator=g) * 0.3
    V = torch.randn(2, d, generator=g) * 0.3
    cell = Cell("b1", "edit", "circuit", 0, ["MLP 20"])
    tensors = {"U:20": U, "V:20": V}
    u = apply_hf(hf, cell, tensors)
    lh = logits_hf(hf, tok, text)
    u()
    u = apply_tl(tl, cell, tensors)
    lt = logits_tl(tl, text)
    u()
    report["edit"] = compare(lh, lt, base_h, base_t)
    # steering: head A8.5 and MLP 15
    vecs = {
        "vec:A8.5": torch.randn(d, generator=g) * 0.5,
        "vec:MLP 15": torch.randn(d, generator=g) * 0.5,
        "coefficient": torch.tensor(1.0),
    }
    cell = Cell("b2", "steer", "circuit", 0, ["A8.5", "MLP 15"])
    u = apply_hf(hf, cell, vecs)
    lh = logits_hf(hf, tok, text)
    u()
    u = apply_tl(tl, cell, vecs)
    lt = logits_tl(tl, text)
    u()
    report["steer"] = compare(lh, lt, base_h, base_t)
    # SafeTune refusal-direction cell: residual-stream addition after layer 10 (HF: SafeTune hook;
    # TL: hook_resid_post)
    vecs = {"resid:10": torch.randn(d, generator=g) * 0.5, "coefficient": torch.tensor(1.0)}
    cell = Cell("b2", "steer", "safetune_rd", 0, [])
    u = apply_hf(hf, cell, vecs)
    lh = logits_hf(hf, tok, text)
    u()
    u = apply_tl(tl, cell, vecs)
    lt = logits_tl(tl, text)
    u()
    report["steer_resid"] = compare(lh, lt, base_h, base_t)
    # A18: real LoRA cells through both hook implementations, and the module-level exactness of the
    # hook against fp32 (W + ΔW) x, next to the bf16-merged weight it replaces
    for cell_dir in ("b1/lora/circuit_s0", "b1/lora/unrestricted_s0", "b2/lora/random_s0"):
        d = Path(cfg["out_dir"]) / "interventions" / cell_dir
        if not (d / "delta.safetensors").exists():
            continue
        cell, tensors = Cell.load(d)
        u = apply_hf(hf, cell, tensors)
        lh = logits_hf(hf, tok, text)
        u()
        u = apply_tl(tl, cell, tensors)
        lt = logits_tl(tl, text)
        u()
        report[f"real_{cell.name}"] = compare(lh, lt, base_h, base_t)
        assert float((logits_hf(hf, tok, text) - base_h).abs().max()) < 1e-3, "undo failed (hf)"
        assert float((logits_tl(tl, text) - base_t).abs().max()) < 1e-3, "undo failed (tl)"
        lm = logits_merged(hf, tok, text, tensors)
        report[f"real_{cell.name}"]["merged_vs_hook"] = compare(lm, lh, base_h, base_h) | {
            "note": "delta_hf = bf16-merged weights (pre-A18), delta_tl = hook"
        }
        if cell_dir == "b1/lora/circuit_s0":
            report["module_exactness"] = module_exactness(hf, tok, text, tensors)
    print(json.dumps(report, indent=1))
    Path("checks/results/apply_equivalence.json").write_text(json.dumps(report, indent=1))
    floor = float((base_h - base_t).abs().max())
    for k, v in report.items():
        if k == "module_exactness":
            continue
        assert v["delta_hf"] > 4 * floor, (
            f"{k}: effect {v['delta_hf']} not above the bf16 floor {floor}"
        )
        assert v["cosine"] > 0.95, f"{k}: hf and tl deltas disagree (cosine {v['cosine']:.3f})"
    m = report.get("module_exactness")
    if m:
        assert m["max_hook_vs_peft_ulps"] == 0.0 and m["max_hook_vs_fp32_ulps"] <= 1.0, m
    print("apply equivalence OK")


@torch.no_grad()
def logits_merged(hf, tok, text: str, tensors: dict) -> torch.Tensor:
    """The pre-A18 application: ΔW added to the bf16 weights in place (restored afterwards)."""
    saved = []
    for path, d in lora_deltas(tensors).items():
        w = _get_module(hf, path).weight
        saved.append((w, w.detach().to("cpu", copy=True)))
        w.add_(d.to(w.device, w.dtype))
    try:
        return logits_hf(hf, tok, text)
    finally:
        for w, orig in saved:
            w.copy_(orig.to(w.device))


@torch.no_grad()
def module_exactness(hf, tok, text: str, tensors: dict) -> dict:
    """Every adapted module of the cell on its own input x from one forward pass: the hook's output
    against PEFT's forward (bf16 W x + fp32 term, rounded once: the model that was trained) and
    against fp32 (W + ΔW) x, in units of one bf16 ulp of the largest output; the bf16-merged weight
    under the same measures, plus the share of its output entries that differ from PEFT's."""
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(hf.device)
    inputs: dict[str, torch.Tensor] = {}
    handles = [
        _get_module(hf, path).register_forward_hook(
            lambda m, inp, out, path=path: inputs.__setitem__(path, inp[0].detach())
        )
        for path in lora_factors(tensors)
    ]
    hf(**enc)
    for h in handles:
        h.remove()
    s, lin = float(tensors["scaling"]), torch.nn.functional.linear
    per = {}
    for path, (A, B) in lora_factors(tensors).items():
        w = _get_module(hf, path).weight
        x, A, B = inputs[path], A.to(w.device), B.to(w.device)
        dw = lora_deltas({"scaling": tensors["scaling"], f"A:{path}": A, f"B:{path}": B})[path]
        fp32 = lin(x.float(), w.float() + dw)
        peft = add_term(lin(x, w), lora_term(x, A, B, s)).float()
        hook = add_term(lin(x, w), lora_term(x, A, B, s)).float()  # what apply_hf's hook returns
        merged = lin(x, (w.float() + dw).to(w.dtype)).float()
        ulp = float(fp32.abs().max()) * 2.0**-7  # bf16: 8-bit significand
        per[path] = {
            "hook_vs_peft_ulps": float((hook - peft).abs().max()) / ulp,
            "hook_vs_fp32_ulps": float((hook - fp32).abs().max()) / ulp,
            "merged_vs_fp32_ulps": float((merged - fp32).abs().max()) / ulp,
            "merged_vs_peft_ulps": float((merged - peft).abs().max()) / ulp,
            "merged_entries_differ": float((merged != peft).float().mean()),
            "term_rel_l2": float(lora_term(x, A, B, s).norm() / fp32.norm()),
        }
    return {
        "n_modules": len(per),
        "max_hook_vs_peft_ulps": max(v["hook_vs_peft_ulps"] for v in per.values()),
        "max_hook_vs_fp32_ulps": max(v["hook_vs_fp32_ulps"] for v in per.values()),
        "max_merged_vs_fp32_ulps": max(v["merged_vs_fp32_ulps"] for v in per.values()),
        "max_merged_vs_peft_ulps": max(v["merged_vs_peft_ulps"] for v in per.values()),
        "mean_merged_entries_differ": sum(v["merged_entries_differ"] for v in per.values())
        / len(per),
        "term_rel_l2_range": [
            min(v["term_rel_l2"] for v in per.values()),
            max(v["term_rel_l2"] for v in per.values()),
        ],
    }


if __name__ == "__main__":
    main()
