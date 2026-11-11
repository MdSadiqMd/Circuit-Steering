"""SafeTune refusal-direction steering baseline for B2 (brief §5: `safetune` supplies "the
refusal-direction steering baseline"). Condition `safetune_rd`, one cell per seed.

Per seed: SafeTune's `extract_refusal_direction` (Arditi et al. 2024: difference of residual-stream
means at the last prompt token, harmful − benign, with SafeTune's layer-selection sweep: minimum
bypass refusal under ablation subject to the KL and layer constraints) on a seeded 5/6 subsample of
the same English prompts the circuit steering cells use (harmful prompts the model refuses vs benign
near-misses). The raw direction at the selected layer is stored as `resid:{layer}` and added at
every position to the residual stream after that layer (HF: SafeTune
`RefusalDirectionModel(mode="steer")`;
TL: `hook_resid_post`; equivalence checked in checks/check_apply_equivalence.py). Norm-matched like
every other steering condition: coefficient = size_at_1(circuit, seed 0) / |d| capped at
max_norm_match_factor, times main_multiplier. No sweep cells (amendment A3: the B2 matched-efficacy
table is uninformative).
Run: uv run python interventions/refusal_direction.py
"""

from __future__ import annotations

import importlib.metadata as md
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from aya_circuits.config import load_config
from aya_circuits.manifest import Manifest
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import chat_wrap, load_hf
from safetune.interventions.steer.refusal_direction import (
    RefusalDirectionConfig,
    extract_refusal_direction,
)

from interventions.cell import Cell, delta_norm
from interventions.run_grid import b2_split

ROOT = Path(__file__).resolve().parent.parent
COND = "safetune_rd"
LOG = logging.getLogger("safetune.interventions.steer.refusal_direction")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def main() -> None:
    cfg = load_config()
    run_root = Path(cfg["out_dir"])
    st_cfg = cfg["interventions"]["steering"]
    match = json.loads((run_root / "interventions/b2/steer/circuit_s0/cell.json").read_text())[
        "meta"
    ]["size_at_1"]
    train_h, _, benign = b2_split(cfg)
    harmful_all = [r["en"] for r in train_h if r["gen"]["en"]["refused"]][:64]
    benign_all = [r["en"] for r in benign][:64]
    model, tok = load_hf(cfg)
    probe = [{"role": "user", "content": harmful_all[0]}]
    # SafeTune formats prompts with tokenizer.apply_chat_template; load_hf set the pipeline's format
    assert tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True) == chat_wrap(
        tok, harmful_all[0], cfg["chat_format"]
    )
    mw = MemWatch(cfg["memory_budget_gb"])
    LOG.setLevel(logging.INFO)
    for seed in cfg["seeds"]:
        cell_dir = Cell("b2", "steer", COND, seed, []).dir(run_root)
        if (cell_dir / "delta.safetensors").exists() and not cfg.get("force"):
            print("cached", cell_dir)
            continue
        rng = random.Random(seed)
        harmful = rng.sample(harmful_all, len(harmful_all) * 5 // 6)
        benign_s = rng.sample(benign_all, len(benign_all) * 5 // 6)
        cap = _Capture()
        LOG.addHandler(cap)
        mf, t0 = Manifest("grid_steer", cfg, seed), time.time()
        mf.add_input(ROOT / f"data/b2/baseline_{cfg['behaviour']['b2_category']}.jsonl")
        d, layer, _ = extract_refusal_direction(
            model,
            tok,
            harmful,
            benign_s,
            RefusalDirectionConfig(pool_method="last_token", normalize=False),
        )
        LOG.removeHandler(cap)
        d = d.detach().float().cpu()
        s1 = float(d.norm())
        coeff = min(st_cfg["max_norm_match_factor"], match / max(s1, 1e-8))
        stats = {
            "size_at_1": s1,
            "base_coefficient": coeff,
            "layer": int(layer),
            "n_source": len(benign_s),
            "n_target": len(harmful),
            "safetune_version": md.version("safetune"),
            "safetune_log": cap.lines,
        }
        cell = Cell(
            "b2",
            "steer",
            COND,
            seed,
            [],
            meta={
                **stats,
                "coefficient_multiplier": st_cfg["main_multiplier"],
                "train_ids": [r["id"] for r in train_h],
            },
        )
        tensors = {
            f"resid:{layer}": d,
            "coefficient": torch.tensor(coeff * st_cfg["main_multiplier"]),
        }
        norm = delta_norm(tensors, "steer")
        cell.meta["delta_norm"] = norm
        cell.save(run_root, tensors)
        mw.check()
        mf.add_output(cell_dir / "delta.safetensors")
        mf.write(
            cell_dir / "manifest.json",
            memory=mw.summary(),
            stats=stats,
            wall_s=round(time.time() - t0, 1),
            delta_norm=norm,
        )
        print(
            f"{cell.name}: {time.time() - t0:.0f}s layer={layer} |d|={s1:.3g} "
            f"coeff={coeff:.3g} |Δ|={norm:.3g}",
            flush=True,
        )
        for line in cap.lines:
            print("  safetune:", line)


if __name__ == "__main__":
    main()
