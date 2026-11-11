"""L5 relearning probe for B2 (brief §3.3 item 4; Guo et al. 2025 measure how much a removed
behaviour comes back after brief further training). Here the behaviour is strengthened refusal,
so the probe asks how much of the gain survives a short *benign* fine-tune, the attack that
erodes refusal without any harmful training data (Qi et al. 2024, "Fine-tuning aligned language
models compromises safety, even when users do not intend to"); no compliance with a harmful
request is ever trained on.

For the base model and every B2 main cell (no sweep cells, no perturbation controls): apply the
cell (weights in place or steering hooks), train an unrestricted LoRA adapter on 64 fixed benign
Dolly instruction/response pairs (rank 8, lr 2e-4, 40 steps, seed 0, same trainer as the cells,
interventions/lora.py), apply it, re-measure refusal on the held-out harmful cyber prompts in all
five languages and on the English near-miss benign prompts, then remove both. Nothing is saved but
the judgments: runs/audit/b2/<cell>/relearn.json (harmful-prompt replies are not stored, brief §8).
Run: uv run python audit/relearn.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap, load_hf
from neb import behavioural as B

from audit.run_audit import run_to_dict
from data.sanitize import strip_audit
from interventions import lora
from interventions.cell import Cell, apply_hf
from interventions.run_grid import b2_split

ROOT = Path(__file__).resolve().parent.parent
DOLLY = ROOT / ".cache/raw/dolly/databricks-dolly-15k.jsonl"
N_PAIRS, STEPS = 64, 40


def benign_pairs(tok, cfg: dict) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in DOLLY.open()]
    rows = [
        r
        for r in rows
        if r["category"] in ("open_qa", "general_qa", "brainstorming", "creative_writing")
        and not r["context"]
        and 20 <= len(r["response"]) <= 400
    ]
    rows = random.Random(0).sample(rows, N_PAIRS)
    return [(chat_wrap(tok, r["instruction"], cfg["chat_format"]), r["response"]) for r in rows]


def probe(model, tok, cfg, pairs, held, benign) -> dict:
    factors, stats = lora.train(
        model, tok, pairs, [], "unrestricted", {**cfg["interventions"]["lora"], "steps": STEPS}, 0
    )
    undo = apply_hf(model, Cell("b2", "lora", "relearn", 0, []), factors)
    try:
        out = {
            f"refusal_harmful_{lang}": run_to_dict(
                B.eval_refusal(model, tok, cfg, held, lang, f"relearn_harmful_{lang}")
            )
            for lang in cfg["languages"]
        }
        out["refusal_benign_en"] = run_to_dict(
            B.eval_refusal(model, tok, cfg, benign, "en", "relearn_benign_en")
        )
    finally:
        undo()
    out["train"] = {k: stats[k] for k in ("steps", "final_loss", "loss_first5", "n_trainable")}
    return strip_audit(out)


def main() -> None:
    cfg = load_config()
    runs = ROOT / cfg["out_dir"]
    out_root = runs / "audit/b2"
    model, tok = load_hf(cfg)
    pairs = benign_pairs(tok, cfg)
    split = json.loads((runs / "interventions/b2_split.json").read_text())
    _, _, benign_rows = b2_split(cfg)
    pool = [json.loads(line) for line in (ROOT / "data/b2/baseline_cyber.jsonl").open()]
    held = [r for r in pool if r["id"] in set(split["held_out"])]
    benign = benign_rows[: cfg["audit"]["l2"]["n_benign"]]
    base_p = out_root / "base/relearn.json"
    if not base_p.exists():
        base_p.write_text(json.dumps(probe(model, tok, cfg, pairs, held, benign)))
        print("base relearn done", flush=True)
    for cj in sorted((runs / "interventions/b2").glob("*/*/cell.json")):
        cell = Cell.peek(cj.parent)
        res_p = out_root / cell.name / "relearn.json"
        if (
            "@" in cell.condition
            or cell.condition.startswith("perturb_")
            or res_p.exists()
            or not res_p.parent.exists()  # audited cells only (run_audit.py creates the dir)
        ):
            continue
        tensors = Cell.load(cj.parent)[1]
        t0 = time.time()
        undo = apply_hf(model, cell, tensors)
        try:
            res = probe(model, tok, cfg, pairs, held, benign)
        finally:
            undo()
        res_p.write_text(json.dumps(res))
        h = {k: v["headline"]["refusal"] for k, v in res.items() if k.startswith("refusal_")}
        print(f"{cell.name} relearn {time.time() - t0:.0f}s {h}", flush=True)


if __name__ == "__main__":
    main()
