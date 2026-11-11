"""L1 one-hop ripple evaluation of the B1 cells (data/b1/ripple_en.jsonl, data/b1_ripple.py).

For the base model (every item) and every B1 main cell (items of the cell's own fold: `edited_keys`
for weight cells, the seed's fold for steering, as in audit/run_audit.py), apply the cell to the HF
model, score the items with the fact matcher through AuditKIT (neb.behavioural.eval_facts) and
write runs/audit/b1/<cell>/ripple.json. audit/analyze.py turns it into `l1_ripple_gain` (paired mean
of cell − base on the same items). Resumable.
Run: uv run python audit/ripple.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config
from aya_circuits.model import load_hf
from neb import behavioural as B

from audit.run_audit import run_to_dict
from interventions.cell import Cell, apply_hf
from interventions.run_grid import b1_folds

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = load_config()
    runs = ROOT / cfg["out_dir"]
    items = [json.loads(line) for line in (ROOT / "data/b1/ripple_en.jsonl").open()]
    out_root = runs / "audit/b1"
    model, tok = load_hf(cfg)
    base_p = out_root / "base/ripple.json"
    if not base_p.exists():
        base_p.write_text(json.dumps(run_to_dict(B.eval_facts(model, tok, items, "en", "ripple"))))
    folds = b1_folds(cfg)
    for cj in sorted((runs / "interventions/b1").glob("*/*/cell.json")):
        cell = Cell.peek(cj.parent)
        res_p = out_root / cell.name / "ripple.json"
        if "@" in cell.condition or res_p.exists() or not (out_root / cell.name).exists():
            continue
        tensors = Cell.load(cj.parent)[1]
        keys = cell.meta.get("edited_keys") or [
            [f["relation"], f["subject_qid"]]
            for f in folds[cell.seed if cell.seed in cfg["seeds"] else cfg["seeds"][cell.seed % 3]]
        ]
        keys = {tuple(k) for k in keys}
        rows = [r for r in items if (r["fact_relation"], r["fact_subject_qid"]) in keys]
        t0 = time.time()
        undo = apply_hf(model, cell, tensors)
        try:
            res = run_to_dict(B.eval_facts(model, tok, rows, "en", "ripple"))
        finally:
            undo()
        res_p.write_text(json.dumps(res))
        print(f"{cell.name} ripple n={len(rows)} acc={res['headline']} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
