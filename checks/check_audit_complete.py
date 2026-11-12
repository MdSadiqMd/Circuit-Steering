"""Is the audit of every main cell complete and finite? For each behaviour: every main cell
(no sweep) has hf.json, l4.json, l4_cellset.json, l4_edges.safetensors, l4_ioi_fixed.json,
l4_pillar.json, l4_shared.json, arbitration.json; B1 cells ripple.json; B2 non-control cells
relearn.json; every numeric leaf of the JSON files is finite; the scorecards exist and agree with
`docs/results/<beh>_cells.csv`. Optional `cells=lora` restricts the intervention. Exit 1 on any gap.
Run: uv run python checks/check_audit_complete.py [cells=lora]
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config

from interventions.cell import Cell

ROOT = Path(__file__).resolve().parent.parent
COMMON = (
    "hf.json",
    "l4.json",
    "l4_cellset.json",
    "l4_edges.safetensors",
    "l4_ioi_fixed.json",
    "l4_pillar.json",
    "l4_shared.json",
    "arbitration.json",
)


def non_finite(obj, path: str = "") -> list[str]:
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in non_finite(v, f"{path}.{k}")]
    if isinstance(obj, list):
        return [x for i, v in enumerate(obj) for x in non_finite(v, f"{path}[{i}]")]
    if isinstance(obj, float) and not math.isfinite(obj):
        return [path]
    return []


def main() -> None:
    cfg = load_config()
    runs = Path(cfg["out_dir"])
    want = set(str(cfg.get("cells", "steer,lora,edit")).split(","))
    gaps: list[str] = []
    for beh in ("b1", "b2"):
        csv_cells = {
            r["cell"] for r in csv.DictReader((ROOT / f"docs/results/{beh}_cells.csv").open())
        }
        for cj in sorted((runs / "interventions" / beh).glob("*/*/cell.json")):
            c = Cell.peek(cj.parent)
            if c.intervention not in want or "@" in c.condition:
                continue
            d = runs / "audit" / beh / c.name
            need = list(COMMON)
            if beh == "b1":
                need.append("ripple.json")
            if beh == "b2" and not c.condition.startswith("perturb_"):
                need.append("relearn.json")
            for f in need:
                if not (d / f).exists():
                    gaps.append(f"{c.name}: missing {f}")
                elif f.endswith(".json"):
                    bad = non_finite(json.loads((d / f).read_text()))
                    if bad:
                        gaps.append(f"{c.name}: non-finite {f} {bad[:3]}")
            if not (cj.parent / "scorecard.json").exists():
                gaps.append(f"{c.name}: missing scorecard.json")
            if c.name not in csv_cells:
                gaps.append(f"{c.name}: not in docs/results/{beh}_cells.csv")
    print("\n".join(gaps) if gaps else "audit complete and finite for every main cell")
    raise SystemExit(1 if gaps else 0)


if __name__ == "__main__":
    main()
