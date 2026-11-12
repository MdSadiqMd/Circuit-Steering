"""A19 validity check: one cell attributed twice gives the same per-example scores and jackknife SE.
The cell is re-attributed by audit/l4_a19.py stage=cells into a scratch root that links the main
root's circuits, A19 base files and this one cell; this script compares the two a19.safetensors.
Run: uv run python checks/check_a19_repro.py <scratch root> [cell, default b1_lora_circuit_s0]
Writes checks/results/a19_repro.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401
from neb.attribution import jackknife_se
from safetensors.torch import load_file

from audit.a19_analyze import cad_series
from audit.l4_a19 import tasks_for

ROOT = Path(__file__).resolve().parent.parent


def main(scratch: Path, name: str) -> None:
    beh, seed = name.split("_")[0], int(name.rsplit("_s", 1)[1]) % 3
    a = load_file(str(ROOT / "runs/audit" / beh / name / "a19.safetensors"))
    b = load_file(str(scratch / "audit" / beh / name / "a19.safetensors"))
    out = {"cell": name}
    for task in tasks_for(beh):
        base = load_file(str(ROOT / "runs/audit/a19/base" / f"{task}_s{seed}.safetensors"))
        idx = a[f"{task}_idx"]
        bp = base["scores"].reshape(base["scores"].shape[0], -1)[:, idx]
        (fa, la), (fb, lb) = cad_series(bp, a[f"{task}_perex"]), cad_series(bp, b[f"{task}_perex"])
        out[task] = {
            "same_edges": bool((idx == b[f"{task}_idx"]).all()),
            "perex_max_abs_diff": float((a[f"{task}_perex"] - b[f"{task}_perex"]).abs().max()),
            "perex_max_abs": float(a[f"{task}_perex"].abs().max()),
            "cad": [fa, fb],
            "jackknife_se": [jackknife_se(la), jackknife_se(lb)],
        }
    (ROOT / "checks/results/a19_repro.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "b1_lora_circuit_s0")
