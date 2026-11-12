"""Plain-text fluency of the base model and of every cell of a run, in five languages.

  uv run python audit/a23_fluency.py out_dir=runs/a23     # also works on runs/a22

The audit's own fluency number is English WikiText perplexity. A22's adapters raised it by
11–26 % while facts, MMLU and IFEval barely moved, so A23 adds a second, multilingual reading on
text that no adapter was trained on: the held-out XNLI test premises of data/a23/fluency.jsonl
(200 parallel sentences per language, human translations). Text is fed as the audit feeds
WikiText: no chat template, no BOS; the first token is not scored.

Writes <out_dir>/fluency/<cell>.json = {lang: [summed NLL, scored tokens]} for the base and every
cell under <out_dir>/interventions/b2/. `ratio` turns two such records into perplexity ratios.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
FLUENCY = ROOT / "data/a23/fluency.jsonl"


def rows(split: str) -> list[dict]:
    return [r for r in map(json.loads, FLUENCY.open()) if r["split"] == split]


def by_lang(model, tok, texts: list[dict], bs: int = 8, max_len: int = 96) -> dict[str, list]:
    """{lang: [summed next-token NLL, scored tokens]} over `texts` ({"lang", "text"} rows)."""
    import torch

    tok.padding_side = "right"
    out: dict[str, list] = {}
    for lang in sorted({r["lang"] for r in texts}):
        batch_texts = [r["text"] for r in texts if r["lang"] == lang]
        tot, n = 0.0, 0
        for i in range(0, len(batch_texts), bs):
            enc = tok(
                batch_texts[i : i + bs],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
            ).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), enc["input_ids"][:, 1:], reduction="none"
            )
            mask = enc["attention_mask"][:, 1:]
            tot += float((nll * mask).sum())
            n += int(mask.sum())
        out[lang] = [tot, n]
    return out


def ratio(cell: dict[str, list], base: dict[str, list]) -> dict[str, float]:
    """Perplexity ratio cell / base per language, and `mean` = their geometric mean."""
    r = {k: math.exp(cell[k][0] / cell[k][1] - base[k][0] / base[k][1]) for k in base}
    r["mean"] = math.exp(sum(math.log(v) for v in r.values()) / len(r))
    return r


def main() -> None:
    from aya_circuits.config import load_config
    from aya_circuits.model import load_hf

    from interventions.cell import Cell, apply_hf

    cfg = load_config()
    run_root = Path(cfg["out_dir"])
    out_root = run_root / "fluency"
    out_root.mkdir(parents=True, exist_ok=True)
    cells = sorted(p.parent for p in (run_root / "interventions/b2").glob("*/*/cell.json"))
    todo = [d for d in cells if not (out_root / f"{Cell.peek(d).name}.json").exists()]
    if (out_root / "base.json").exists() and not todo:
        return
    texts = rows("test")
    model, tok = load_hf(cfg)
    if not (out_root / "base.json").exists():
        (out_root / "base.json").write_text(json.dumps(by_lang(model, tok, texts)))
    base = json.loads((out_root / "base.json").read_text())
    for d in todo:
        cell, tensors = Cell.load(d)
        undo = apply_hf(model, cell, tensors)
        try:
            res = by_lang(model, tok, texts)
        finally:
            undo()
        (out_root / f"{cell.name}.json").write_text(json.dumps(res))
        print(cell.name, {k: round(v, 3) for k, v in ratio(res, base).items()}, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        b = {"en": [300.0, 100], "hi": [500.0, 100]}
        c = {"en": [310.0, 100], "hi": [500.0, 100]}
        r = ratio(c, b)
        assert abs(r["en"] - math.exp(0.1)) < 1e-12 and r["hi"] == 1.0
        assert abs(r["mean"] - math.exp(0.05)) < 1e-12 and ratio(b, b)["mean"] == 1.0
        print("a23_fluency ok")
    else:
        main()
