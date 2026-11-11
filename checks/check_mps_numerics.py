"""MPS numerics check: TransformerLens warns that the MPS backend may be silently wrong
(TransformerLens issue 1178). We compare, on 8 prompts, (a) TL bf16 on mps vs HF bf16 on mps,
(b) TL bf16 on mps vs TL float32 on cpu, (c) HF float32 on cpu as the reference: top-1 agreement,
top-5 overlap and max |Δlogit|. Writes checks/results/mps_numerics.json. Run after GPU jobs finish
(loads a 13 GB fp32 copy on CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401
import torch
from aya_circuits.config import load_config
from aya_circuits.model import load_hf, load_tl


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    def top5(x):
        return set(torch.topk(x, 5).indices.tolist())

    return {
        "top1_agree": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
        "top5_overlap": sum(len(top5(x) & top5(y)) / 5 for x, y in zip(a, b, strict=False))
        / len(a),
        "max_abs_diff": float((a - b).abs().max()),
        "mean_abs_diff": float((a - b).abs().mean()),
    }


@torch.no_grad()
def main() -> None:
    cfg = load_config()
    lines = Path("data/b1/prompts_en.jsonl").read_text().splitlines()[:8]
    rows = [json.loads(line) for line in lines]
    texts = [r["text"] for r in rows]
    out = {}
    hf, tok = load_hf(cfg)
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc_mps = {k: v.to(hf.device) for k, v in enc.items()}
    l_hf_mps = hf(**enc_mps).logits[:, -1].float().cpu()
    del hf
    torch.mps.empty_cache()
    tl = load_tl(cfg)
    l_tl_mps = torch.stack(
        [tl(tl.to_tokens(t, prepend_bos=False))[0, -1].float().cpu() for t in texts]
    )
    del tl
    torch.mps.empty_cache()
    hf32, _ = load_hf({**cfg, "device": "cpu"}, dtype="float32")
    l_hf_cpu32 = hf32(**enc).logits[:, -1].float()
    del hf32
    tl32 = load_tl({**cfg, "device": "cpu", "dtype": "float32"})
    l_tl_cpu32 = torch.stack(
        [tl32(tl32.to_tokens(t, prepend_bos=False))[0, -1].float() for t in texts]
    )
    out["tl_mps_vs_hf_mps"] = stats(l_tl_mps, l_hf_mps)
    out["tl_mps_vs_tl_cpu32"] = stats(l_tl_mps, l_tl_cpu32)
    out["hf_mps_vs_hf_cpu32"] = stats(l_hf_mps, l_hf_cpu32)
    out["tl_cpu32_vs_hf_cpu32"] = stats(l_tl_cpu32, l_hf_cpu32)
    out["gold_top1_cpu32"] = float(
        sum(int(l_hf_cpu32[i].argmax()) == rows[i]["label"] for i in range(len(rows))) / len(rows)
    )
    Path("checks/results/mps_numerics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
