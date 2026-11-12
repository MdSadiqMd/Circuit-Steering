"""Generation throughput of Tiny Aya on this machine, for the cost comparison in the write-up.

  uv run python checks/bench_throughput.py [cell=runs/a22/interventions/b2/lora/circuit30_s0]

Greedy generation of 128 new tokens for 64 benign prompts (the A22 benign train prompts, English)
at batch sizes 1, 8 and 32, base model and, if `cell` is given, the model with that adapter applied
as hooks. Reports generated tokens per second (wall clock, tokens after the prompt, EOS-terminated
rows counted up to their EOS). Writes checks/results/throughput.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
N_PROMPTS, NEW_TOKENS, BATCHES = 64, 128, (1, 8, 32)


def main() -> None:
    import torch
    from aya_circuits.config import load_config
    from aya_circuits.model import chat_wrap, load_hf

    from interventions.cell import Cell, apply_hf

    cfg = load_config()
    model, tok = load_hf(cfg)
    rows = [json.loads(line) for line in (ROOT / "data/a22/benign.jsonl").open()][:N_PROMPTS]
    prompts = [chat_wrap(tok, r["en"], cfg["chat_format"]) for r in rows]
    tok.padding_side = "left"

    @torch.no_grad()
    def bench(bs: int) -> dict:
        n_new, n_in = 0, 0
        use = prompts if bs > 1 else prompts[:16]
        t0 = time.time()
        for i in range(0, len(use), bs):
            enc = tok(use[i : i + bs], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(model.device)
            g = model.generate(
                **enc, max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id
            )
            new = g[:, enc["input_ids"].shape[1] :]
            n_new += int((new != tok.pad_token_id).sum())
            n_in += int(enc["attention_mask"].sum())
        wall = time.time() - t0
        return {
            "batch_size": bs,
            "prompts": len(use),
            "input_tokens": n_in,
            "output_tokens": n_new,
            "wall_s": round(wall, 2),
            "output_tokens_per_s": round(n_new / wall, 1),
        }

    bench(8)  # warm-up
    out = {
        "model": cfg["model"],
        "dtype": cfg["dtype"],
        "device": cfg["device"],
        "machine": cfg.get("machine", "Apple M4 Max, MPS"),
        "new_tokens": NEW_TOKENS,
        "base": [bench(bs) for bs in BATCHES],
    }
    if cfg.get("cell"):
        cell, tensors = Cell.load(cfg["cell"])
        undo = apply_hf(model, cell, tensors)
        try:
            out["adapter"] = {"cell": cell.name, "runs": [bench(bs) for bs in BATCHES]}
        finally:
            undo()
    (ROOT / "checks/results/throughput.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
