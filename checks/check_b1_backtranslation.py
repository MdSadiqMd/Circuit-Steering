"""Brief §3.3 item 1: back-translation check of the B1 parallel prompts (hi/ar/fr/sw).

The B1 prompts are the hand-written templates of data/b1_build.py (question and paraphrase per
relation) filled with Wikidata labels. Every non-English template, filled with a real subject of
that relation, is translated back to English by Tiny Aya and scored with the token-F1 of the B2
check (data/translate.py) against the English template filled with the same subject's English
label. These rows are questions, and with the B2 prompt the model answered them instead of
translating ("Pakistan"), so the text is quoted and the prompt says not to answer. Rows under 0.5
are flagged. Writes checks/results/b1_backtranslation.json.
Run: uv run python checks/check_b1_backtranslation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap, generate, load_hf

from data.b1_build import PARAPHRASE, TEMPLATES
from data.translate import NAME, clean, f1

ROOT = Path(__file__).resolve().parent.parent


def bt_prompt(text: str, lang: str) -> str:
    return (
        f"Translate the {NAME[lang]} question between the quotes into English. Do not answer the "
        f'question. Reply with the English question only.\n\n"{text}"'
    )


def main() -> None:
    cfg = load_config()
    pool = [json.loads(line) for line in (ROOT / "data/b1/facts_pool.jsonl").open()]
    example = {}
    for r in pool:  # first subject per relation (labels exist in all five languages by build)
        example.setdefault(r["relation"], r["subject"])
    items = [
        {
            "form": form,
            "relation": rel,
            "lang": lang,
            "text": by_lang[lang].format(s=example[rel][lang]),
            "reference_en": by_lang["en"].format(s=example[rel]["en"]),
        }
        for form, table in (("question", TEMPLATES), ("paraphrase", PARAPHRASE))
        for rel, by_lang in table.items()
        for lang in ("hi", "ar", "fr", "sw")
    ]
    model, tok = load_hf(cfg)
    outs = generate(
        model,
        tok,
        [chat_wrap(tok, bt_prompt(x["text"], x["lang"]), cfg["chat_format"]) for x in items],
        96,
        16,
    )
    for x, o in zip(items, outs, strict=True):
        x["back_en"] = clean(o)
        x["bt_f1"] = round(f1(x["reference_en"], x["back_en"]), 3)
    summary = {
        lang: {
            "mean_bt_f1": round(
                sum(x["bt_f1"] for x in items if x["lang"] == lang)
                / sum(x["lang"] == lang for x in items),
                3,
            ),
            "flagged": [
                f"{x['form']}:{x['relation']}"
                for x in items
                if x["lang"] == lang and x["bt_f1"] < 0.5
            ],
        }
        for lang in ("hi", "ar", "fr", "sw")
    }
    (ROOT / "checks/results/b1_backtranslation.json").write_text(
        json.dumps({"summary": summary, "items": items}, indent=1, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
