"""Brief §11: a 20-step torch.profiler pass over LoRA training, and a determinism check.

Profiles `interventions.lora.train` for 20 steps on the b1 circuit cell's data (seed 0 fold,
circuit nodes) with the CPU-side profiler (MPS kernels are asynchronous; the op table shows
dispatch cost and the synchronising ops). Then runs 5 steps twice with
`torch.use_deterministic_algorithms(True)` and compares the loss traces and the trained factors bit
for bit, recording whether SDPA on MPS accepts the flag. Writes checks/results/profile_lora.json.
No cell artifact is written.
Run: uv run python checks/profile_lora.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from aya_circuits.config import load_config
from aya_circuits.model import load_hf
from torch.profiler import ProfilerActivity, profile

from interventions import lora
from interventions.run_grid import b1_folds, circuit_scores, fact_prompt_en
from interventions.select import select

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = load_config()
    model, tok = load_hf(cfg)
    nodes = select("circuit", circuit_scores(ROOT / cfg["out_dir"], "b1_en", 0), 0.10)
    pairs = [(fact_prompt_en(f, tok, cfg)[0], f["object"]["en"]) for f in b1_folds(cfg)[0]]
    lcfg = {**cfg["interventions"]["lora"], "steps": 20}
    out: dict = {"cell_like": "b1 lora circuit seed 0", "steps": 20}
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        _, stats = lora.train(model, tok, pairs, nodes, "circuit", lcfg, 0)
    out["wall_s"] = round(time.time() - t0, 1)
    out["train_stats"] = stats
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    out["top_ops_self_cpu"] = table.splitlines()
    print(table)

    det: dict = {}
    try:
        torch.use_deterministic_algorithms(True)
        runs = []
        for _ in range(2):
            f, s = lora.train(model, tok, pairs, nodes, "circuit", {**lcfg, "steps": 5}, 0)
            runs.append((f, s["loss_first5"]))
        det["accepted"] = True
        det["loss_traces"] = [r[1] for r in runs]
        det["losses_identical"] = runs[0][1] == runs[1][1]
        det["factors_identical"] = all(
            torch.equal(runs[0][0][k], runs[1][0][k]) for k in runs[0][0]
        )
    except Exception as e:
        det["accepted"] = False
        det["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    finally:
        torch.use_deterministic_algorithms(False)
    out["deterministic_algorithms"] = det
    (ROOT / "checks/results/profile_lora.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "top_ops_self_cpu"}, indent=1))


if __name__ == "__main__":
    main()
