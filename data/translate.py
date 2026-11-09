"""Model-based translation of B2 prompts into the project languages, with a back-translation check.

Where a human translation exists (MultiJail ar/sw) it is kept and used as the reference. Otherwise
Tiny Aya translates en -> target, then target -> en; the back-translation is compared with the
source by token-F1 (after NFKC + casefold). Rows below the threshold are kept but flagged
(`bt_f1 < 0.5`), so every later table can be filtered and the limitation reported (brief §9).
Harmful prompts are translated as *prompts*; no completion is generated or stored (brief §8).

Input : data/b2/pool_<cat>.jsonl   Output: same rows with <lang> text + bt_<lang> fields, in place.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap, generate, load_hf

HERE = Path(__file__).resolve().parent / "b2"
NAME = {"hi": "Hindi", "ar": "Arabic", "fr": "French", "sw": "Swahili", "en": "English"}


def tr_prompt(text: str, src: str, tgt: str) -> str:
    return (
        f"Translate the following {NAME[src]} text into {NAME[tgt]}. "
        "Reply with the translation only, "
        f"nothing else.\n\n{text}"
    )


def toks(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.findall(r"\w+", s)


def f1(a: str, b: str) -> float:
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    common = sum(min(ta.count(w), tb.count(w)) for w in set(ta))
    if common == 0:
        return 0.0
    p, r = common / len(tb), common / len(ta)
    return 2 * p * r / (p + r)


def clean(out: str) -> str:
    return out.strip().split("\n")[0].strip().strip('"')


def main() -> None:
    cfg = load_config()
    bs, mx = int(cfg.get("batch_size", 16)), 96
    model, tok = load_hf(cfg)
    for pool in sorted(HERE.glob("pool_*.jsonl")):
        rows = [json.loads(line) for line in pool.open()]
        for lang in [code for code in cfg["languages"] if code != "en"]:
            todo = [i for i, r in enumerate(rows) if lang not in r]
            if todo:
                outs = generate(
                    model,
                    tok,
                    [
                        chat_wrap(tok, tr_prompt(rows[i]["en"], "en", lang), cfg["chat_format"])
                        for i in todo
                    ],
                    mx,
                    bs,
                )
                for i, o in zip(todo, outs, strict=False):
                    rows[i][lang] = clean(o)
                    rows[i].setdefault("translated_by_model", []).append(lang)
            back = generate(
                model,
                tok,
                [chat_wrap(tok, tr_prompt(r[lang], lang, "en"), cfg["chat_format"]) for r in rows],
                mx,
                bs,
            )
            for r, b in zip(rows, back, strict=False):
                r[f"bt_{lang}"] = clean(b)
                r[f"bt_f1_{lang}"] = round(f1(r["en"], r[f"bt_{lang}"]), 3)
            flagged = sum(r[f"bt_f1_{lang}"] < 0.5 for r in rows)
            print(
                f"{pool.stem} {lang}: "
                f"mean bt_f1={sum(r[f'bt_f1_{lang}'] for r in rows) / len(rows):.3f} "
                f"flagged={flagged}/{len(rows)}",
                flush=True,
            )
        with pool.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
