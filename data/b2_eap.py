"""Refusal EAP task per language from the baseline generations.

Clean = harmful prompt the model refused (all three candidate categories, so the refusal
*mechanism* is
discovered on ~200 prompts rather than the 34 cyber ones; the B2 intervention target stays the cyber
category); corrupt = benign near-miss prompt the model answered, with the same token count
(CircuitKIT pairs need equal n_pos). Labels: correct_idx = the model's own most
frequent refusal onset token in that language, incorrect_idx = its most frequent compliance onset
(both read from data/b2/baseline_summary.json, so the metric is the model's actual refuse-vs-comply
logit difference at the assistant marker, as in CircuitKIT case study 23). Falls back to pairing
harmful with the closest-length benign prompt and right-padding when no exact length match exists.

Output: data/b2/eap_<lang>.csv plus data/b2/eap_stats.json. Category comes from configs
(behaviour.b2_category) once chosen in docs/preregistration.md."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent / "b2"


def main() -> None:
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    rows, seen = [], set()
    for p in sorted(HERE.glob("baseline_*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    onsets = json.loads((HERE / "baseline_summary.json").read_text())["onsets"]
    rng, stats = random.Random(0), {}
    for lang in cfg["languages"]:

        def pick(kind: str, lang: str = lang) -> str:  # skip whitespace-only onsets
            return next(t for t, _ in onsets[lang][kind] if t.strip())

        ref_tok = tok.encode(pick("refusal"), add_special_tokens=False)[0]
        cmp_tok = tok.encode(pick("comply"), add_special_tokens=False)[0]
        harm = [
            r for r in rows if r["split"] == "harmful" and lang in r and r["gen"][lang]["refused"]
        ]
        ben = [
            r
            for r in rows
            if r["split"] == "benign" and lang in r and not r["gen"][lang]["refused"]
        ]
        if not harm or not ben:
            stats[lang] = {"pairs": 0, "note": "no refused-harmful or complied-benign prompts"}
            continue

        def wrap(r: dict, lang: str = lang) -> str:
            return chat_wrap(tok, r[lang], cfg["chat_format"])

        def ntok(r: dict, wrap=wrap) -> int:
            return len(tok(wrap(r), add_special_tokens=False)["input_ids"])

        # drop length outliers (> 64 tokens, 1-4 per language): a single 250-token prompt in a
        # batch of 4 blew
        # the 30 GB budget in EAP-IG (activation buffer scales with the longest sequence in the
        # batch)
        harm, ben = [r for r in harm if ntok(r) <= 64], [r for r in ben if ntok(r) <= 64]
        blen = defaultdict(list)
        for b in ben:
            blen[len(tok(wrap(b), add_special_tokens=False)["input_ids"])].append(b)
        pairs, exact = [], 0
        for h in harm:
            n = len(tok(wrap(h), add_special_tokens=False)["input_ids"])
            if blen.get(n):
                b, exact = rng.choice(blen[n]), exact + 1
            else:
                b = rng.choice(blen[min(blen, key=lambda k: abs(k - n))])
            pairs.append(
                {
                    "clean": wrap(h),
                    "corrupted": wrap(b),
                    "correct_idx": ref_tok,
                    "incorrect_idx": cmp_tok,
                }
            )
        with (HERE / f"eap_{lang}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"])
            w.writeheader()
            w.writerows(pairs)
        stats[lang] = {
            "pairs": len(pairs),
            "exact_length_pairs": exact,
            "refusal_onset": pick("refusal"),
            "comply_onset": pick("comply"),
            "n_harmful_refused": len(harm),
            "n_benign_complied": len(ben),
        }
    (HERE / "eap_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
