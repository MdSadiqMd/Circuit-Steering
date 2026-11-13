"""Amendment A22 data: splits and retain targets for closing the cross-lingual refusal gap.

  uv run python data/a22_build.py            # needs the model (translation + base replies)
  uv run python data/a22_build.py --check    # split logic only

Harmful prompts (data/b2/pool_*.jsonl, five languages each), split by the sha256 of the prompt id:
  weapons_violence  80 train / 24 dev / 80 test
  cyber, theft_fraud  every prompt is test (categories the adapter never sees; the three prompts
                      that are also in weapons_violence stay with that pool's split)
Benign near-miss prompts:
  the 191 XSTest safe prompts that are in no B2 pool: 151 train / 40 dev, translated here into
  hi/ar/fr/sw by the model with the back-translation check of data/translate.py
  the 59 distinct pool benign prompts (already in five languages): test only
Retain target of a benign train prompt = the base model's own greedy reply (first 32 tokens), so
the adapter is trained to leave benign behaviour where it is. Refusal target of a harmful train
prompt = the model's own most frequent short refusal in that language (REFUSAL).

Writes data/a22/split.json (ids only) and data/a22/benign.jsonl (prompts, translations, replies).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/a22"
TRAIN_CAT, N_TRAIN, N_DEV = "weapons_violence", 80, 24
N_BENIGN_DEV, REPLY_TOKENS, BT_MIN = 40, 32, 0.5
# Tiny Aya's own most frequent short refusals on the B2 pools (data/b2_baseline.py, raw previews)
REFUSAL = {
    "en": "I'm sorry, but I can't assist with that.",
    "hi": "मुझे खेद है, लेकिन मैं इस अनुरोध में सहायता नहीं कर सकता।",
    "ar": "أنا آسف، لكنني لا أستطيع المساعدة في ذلك.",
    "fr": "Je suis désolé, mais je ne peux pas vous aider avec cela.",
    "sw": "Samahani, siwezi kusaidia kwa ombi hilo.",
}


def by_hash(ids: list[str]) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())


def pools() -> dict[str, list[dict]]:
    return {
        p.stem[len("pool_") :]: [json.loads(line) for line in p.open()]
        for p in sorted((ROOT / "data/b2").glob("pool_*.jsonl"))
    }


def make_split(pool: dict[str, list[dict]], xstest: list[dict]) -> dict:
    harm = {c: [r["id"] for r in rows if r["split"] == "harmful"] for c, rows in pool.items()}
    w = by_hash(harm[TRAIN_CAT])
    pool_benign = [r for rows in pool.values() for r in rows if r["split"] == "benign"]
    seen = {r["en"] for r in pool_benign}
    fresh = by_hash([r["id"] for r in xstest if r["label"] == "safe" and r["en"] not in seen])
    return {
        "harmful_train": w[:N_TRAIN],
        "harmful_dev": w[N_TRAIN : N_TRAIN + N_DEV],
        "harmful_test": w[N_TRAIN + N_DEV :]
        # three MultiJail prompts sit in two pools; one in the train category is never a test item
        + [i for c in sorted(harm) if c != TRAIN_CAT for i in harm[c] if i not in set(w)],
        "benign_train": fresh[N_BENIGN_DEV:],
        "benign_dev": fresh[:N_BENIGN_DEV],
        # 12 benign prompts sit in more than one pool: 59 unique of 71 rows
        "benign_test": list(dict.fromkeys(r["id"] for r in pool_benign)),
        "xstest_unsafe": [r["id"] for r in xstest if r["label"] == "unsafe"],
    }


def load() -> tuple[dict, dict[str, dict], dict[str, dict]]:
    """(split, harmful/benign pool rows by id, A22 benign rows by id) for the cells and the eval."""
    split = json.loads((OUT / "split.json").read_text())
    rows: dict[str, dict] = {}
    for c, rs in sorted(pools().items(), key=lambda kv: kv[0] != TRAIN_CAT):
        for r in rs:
            rows.setdefault(r["id"], {**r, "category": c})
    benign = {r["id"]: r for r in map(json.loads, (OUT / "benign.jsonl").open())}
    return split, rows, benign


def main() -> None:
    import torch
    from aya_circuits.config import load_config
    from aya_circuits.model import chat_wrap, generate, load_hf
    from neb.judge import refused

    from data.translate import clean, f1, tr_prompt

    cfg = load_config()
    xstest = [json.loads(line) for line in (ROOT / "data/b2/xstest_all.jsonl").open()]
    split = make_split(pools(), xstest)
    OUT.mkdir(exist_ok=True)
    (OUT / "split.json").write_text(json.dumps(split, indent=1))
    want = set(split["benign_train"]) | set(split["benign_dev"])
    rows = [
        {"id": r["id"], "type": r["type"], "en": r["en"]}
        | {"split": "train" if r["id"] in set(split["benign_train"]) else "dev"}
        for r in xstest
        if r["id"] in want
    ]
    model, tok = load_hf(cfg)

    def wrap(texts: list[str]) -> list[str]:
        return [chat_wrap(tok, t, cfg["chat_format"]) for t in texts]

    for lang in [code for code in cfg["languages"] if code != "en"]:
        fwd = generate(model, tok, wrap([tr_prompt(r["en"], "en", lang) for r in rows]), 96, 16)
        for r, o in zip(rows, fwd, strict=True):
            r[lang] = clean(o)
        back = generate(model, tok, wrap([tr_prompt(r[lang], lang, "en") for r in rows]), 96, 16)
        for r, b in zip(rows, back, strict=True):
            r[f"bt_f1_{lang}"] = round(f1(r["en"], clean(b)), 3)
        print(lang, "flagged", sum(r[f"bt_f1_{lang}"] < BT_MIN for r in rows), flush=True)

    tok.padding_side = "left"
    for lang in cfg["languages"]:
        texts = wrap([r[lang] for r in rows])
        for i in range(0, len(rows), 16):
            enc = tok(
                texts[i : i + 16], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            with torch.no_grad():
                g = model.generate(
                    **enc,
                    max_new_tokens=REPLY_TOKENS,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            new = g[:, enc["input_ids"].shape[1] :]
            for r, ids in zip(rows[i : i + 16], new, strict=True):
                text = tok.decode(ids, skip_special_tokens=True)
                r.setdefault("reply", {})[lang] = {
                    "text": text,
                    "finished": bool((ids == tok.eos_token_id).any()),
                    "refused": refused(r[lang], text, lang),
                }
        print(lang, "base refuses", sum(r["reply"][lang]["refused"] for r in rows), flush=True)
    with (OUT / "benign.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        s = make_split(
            pools(), [json.loads(line) for line in (ROOT / "data/b2/xstest_all.jsonl").open()]
        )
        n = {k: len(v) for k, v in s.items()}
        assert n["harmful_train"] == 80 and n["harmful_dev"] == 24 and n["harmful_test"] == 159, n
        assert n["benign_train"] == 151 and n["benign_dev"] == 40 and n["benign_test"] == 59, n
        parts = [set(s[k]) for k in ("harmful_train", "harmful_dev", "harmful_test")]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        assert all(len(set(v)) == len(v) for v in s.values())
        assert not set(s["benign_train"]) & set(s["benign_dev"])
        assert s == make_split(
            pools(), [json.loads(line) for line in (ROOT / "data/b2/xstest_all.jsonl").open()]
        )
        print(n, "a22_build ok")
    elif sys.argv[1:] == ["--split"]:  # rewrite the ids only (no model)
        xs = [json.loads(line) for line in (ROOT / "data/b2/xstest_all.jsonl").open()]
        (OUT / "split.json").write_text(json.dumps(make_split(pools(), xs), indent=1))
    else:
        main()
