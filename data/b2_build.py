"""B2 candidate pools: harmful-request categories with matched benign near-miss prompts (English).

Sources (prompts only; no generated harmful content is ever stored, brief §8):
  HarmBench standard behaviours (centerforaisafety/HarmBench, CSV)        -> semantic categories
  MultiJail (DAMO-NLP-SG, human-translated en/ar/sw + 7 more)              -> tag categories, gives
                                                                              human ar and sw for
                                                                              free
  XSTest safe prompts (paul-rottger/xstest, CSV)                           -> near-miss benign
Three candidate categories are built; the one used for B2 is chosen after measuring Tiny Aya's
baseline refusal per language (data/filter_model.py) and recorded in docs/preregistration.md.
Translations into hi/fr (and ar/sw where MultiJail has none) are produced by data/translate.py.

Output: data/b2/pool_<category>.jsonl with fields id, source, split(harmful|benign), en[, ar, sw].
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / ".cache/raw"
OUT = Path(__file__).resolve().parent / "b2"

# category -> (HarmBench SemanticCategory set, MultiJail tag set, XSTest focus/note keywords for
# near-miss)
CATEGORIES = {
    "cyber": (
        {"cybercrime_intrusion"},
        set(),
        {"hack", "steal", "exploit", "attack", "crack", "kill", "terminate", "execute"},
    ),
    "weapons_violence": (
        {"illegal", "harmful"},
        {"Weapons", "Violence & incitement", "Terrorism & organized crime"},
        {"violence", "weapon", "gun", "bomb", "explode", "shoot", "fire", "blow", "knife"},
    ),
    "theft_fraud": (
        set(),
        {"Theft", "Fraud & deception"},
        {"steal", "theft", "money", "scam", "fraud", "bank", "pick", "lock"},
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hb = [
        r
        for r in csv.DictReader((RAW / "harmbench/harmbench_behaviors_text_all.csv").open())
        if r["FunctionalCategory"] == "standard"
    ]
    mj = list(csv.DictReader((RAW / "MultiJail/MultiJail.csv").open(encoding="utf-8-sig")))
    xs = list(csv.DictReader((RAW / "xstest/xstest_prompts.csv").open()))
    safe = [r for r in xs if r["label"] == "safe"]
    stats = {}
    for cat, (hb_cats, mj_tags, kws) in CATEGORIES.items():
        rows = [
            {
                "id": f"hb:{r['BehaviorID']}",
                "source": "harmbench",
                "split": "harmful",
                "en": r["Behavior"],
            }
            for r in hb
            if r["SemanticCategory"] in hb_cats
        ]
        rows += [
            {
                "id": f"mj:{r['id']}",
                "source": "multijail",
                "split": "harmful",
                "en": r["en"],
                "ar": r["ar"],
                "sw": r["sw"],
            }
            for r in mj
            if set(ast.literal_eval(r["tags"])) & mj_tags  # a list literal in the CSV, never eval
        ]
        # near-miss benign: XSTest safe prompts whose focus word or note matches the category
        # lexicon
        for r in safe:
            blob = f"{r['focus']} {r['note']} {r['prompt']}".lower()
            if any(k in blob for k in kws):
                rows.append(
                    {
                        "id": f"xs:{r['id']}",
                        "source": "xstest",
                        "split": "benign",
                        "en": r["prompt"],
                        "xstest_type": r["type"],
                    }
                )
        with (OUT / f"pool_{cat}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[cat] = {
            "harmful": sum(r["split"] == "harmful" for r in rows),
            "benign": sum(r["split"] == "benign" for r in rows),
            "with_human_ar_sw": sum("sw" in r for r in rows),
        }
    # generic benign control set: all XSTest safe prompts (over-refusal), plus the unsafe contrast
    # set
    with (OUT / "xstest_all.jsonl").open("w") as f:
        for r in xs:
            f.write(
                json.dumps(
                    {
                        "id": f"xs:{r['id']}",
                        "type": r["type"],
                        "label": r["label"],
                        "en": r["prompt"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
