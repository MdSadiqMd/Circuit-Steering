"""Reproducibility and cross-device checks for A20 / A21 (puget), written to checks/:
  lora   the A20 cert_s0 LoRA cell trained twice (runs/a20 vs runs/a20_repro): max |ΔW| difference
         over every adapted module, relative to max |ΔW|
  memit  the A21 site10_s0 edit built twice (runs/a21 vs runs/a21_repro): relative L2 of ΔW
  mac    A21 site00_s<d> (puget) vs the main study's MEMIT `circuit_s<d>` (layer 0, same fold and
         settings, built on the Mac): relative L2 of ΔW and both norms
Run: uv run python checks/check_a20_a21_repro.py lora|memit|mac
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from interventions.cell import Cell, edit_deltas, lora_deltas

ROOT = Path(__file__).resolve().parent.parent


def rel(a, b) -> float:
    return float((a - b).norm() / b.norm())


def main(which: str) -> None:
    out = {}
    if which == "lora":
        a = lora_deltas(Cell.load(ROOT / "runs/a20/interventions/b1/lora/cert_s0")[1])
        b = lora_deltas(Cell.load(ROOT / "runs/a20_repro/interventions/b1/lora/cert_s0")[1])
        assert a.keys() == b.keys()
        diff = max(float((a[k] - b[k]).abs().max()) for k in a)
        scale = max(float(a[k].abs().max()) for k in a)
        out = {
            "max_abs_diff": diff,
            "max_abs_delta": scale,
            "rel_l2_max": max(rel(a[k], b[k]) for k in a),
        }
    elif which == "memit":
        a = edit_deltas(Cell.load(ROOT / "runs/a21/interventions/b1/edit/site10_s0")[1])
        b = edit_deltas(Cell.load(ROOT / "runs/a21_repro/interventions/b1/edit/site10_s0")[1])
        out = {"rel_l2": rel(a[10], b[10]), "norms": [float(a[10].norm()), float(b[10].norm())]}
    else:
        for d in (0, 1, 2):
            a = edit_deltas(Cell.load(ROOT / f"runs/a21/interventions/b1/edit/site00_s{d}")[1])[0]
            b = edit_deltas(Cell.load(ROOT / f"runs/interventions/b1/edit/circuit_s{d}")[1])[0]
            out[f"s{d}"] = {
                "rel_l2": rel(a, b),
                "norm_puget": float(a.norm()),
                "norm_mac": float(b.norm()),
            }
    p = ROOT / f"checks/results/a20_a21_repro_{which}.json"
    p.write_text(json.dumps(out, indent=1))
    print(which, json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
