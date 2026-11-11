"""Bystander task data: language identification (per language) in the shared chat format.

IOI and capital_country are CircuitKIT built-ins (registered by name). Language-ID is ours: the
model is asked which language a sentence is written in; clean/corrupt pairs swap the sentence for
one in a
different language with the same token count. Sentences come from the MGSM/Global-MMLU question
fields already fetched (no new downloads), one task per answer language so the answer label is the
language name written in the prompt language.

Output: data/bystanders/eap_langid_<lang>.csv
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/bystanders"
RAW = ROOT / ".cache/raw"

PROMPT = {
    "en": "Which language is the following text written in? Text: {s}. Answer with only the "
    "language name.",
    "fr": "Dans quelle langue le texte suivant est-il écrit ? Texte : {s}. Réponds uniquement par "
    "le nom de la langue.",
    "hi": "निम्नलिखित पाठ किस भाषा में लिखा गया है? पाठ: {s}। केवल भाषा का नाम लिखिए।",
    "ar": "بأي لغة كُتب النص التالي؟ النص: {s}. أجب باسم اللغة فقط.",
    "sw": "Maandishi yafuatayo yameandikwa kwa lugha gani? Maandishi: {s}. Jibu kwa jina la lugha "
    "pekee.",
}
# language name as it should be answered, in the prompt language
NAMES = {
    "en": {"en": "English", "hi": "Hindi", "ar": "Arabic", "fr": "French", "sw": "Swahili"},
    "fr": {"en": "Anglais", "hi": "Hindi", "ar": "Arabe", "fr": "Français", "sw": "Swahili"},
    "hi": {"en": "अंग्रेज़ी", "hi": "हिन्दी", "ar": "अरबी", "fr": "फ़्रांसीसी", "sw": "स्वाहिली"},
    "ar": {
        "en": "الإنجليزية",
        "hi": "الهندية",
        "ar": "العربية",
        "fr": "الفرنسية",
        "sw": "السواحلية",
    },
    "sw": {
        "en": "Kiingereza",
        "hi": "Kihindi",
        "ar": "Kiarabu",
        "fr": "Kifaransa",
        "sw": "Kiswahili",
    },
}


def sentences(lang: str, n: int = 150) -> list[str]:
    df = pd.read_parquet(RAW / "Global-MMLU" / lang / "test-00000-of-00001.parquet")
    qs = [q.strip().split("\n")[0] for q in df["question"].tolist()]
    qs = [q for q in qs if 30 <= len(q) <= 140]
    return qs[:n]


def main() -> None:
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    sents = {lang: sentences(lang) for lang in cfg["languages"]}
    stats = {}
    for plang in cfg["languages"]:
        recs = []
        for slang, ss in sents.items():
            for s in ss:
                text = chat_wrap(tok, PROMPT[plang].format(s=s), cfg["chat_format"])
                base = tok(text, add_special_tokens=False)["input_ids"]
                full = tok(text + NAMES[plang][slang], add_special_tokens=False)["input_ids"]
                recs.append(
                    {"text": text, "slang": slang, "n_tok": len(base), "label": full[len(base)]}
                )
        groups = defaultdict(list)
        for r in recs:
            groups[r["n_tok"]].append(r)
        pairs = []
        for r in recs:
            cands = [
                c
                for c in groups[r["n_tok"]]
                if c["slang"] != r["slang"] and c["label"] != r["label"]
            ]
            if cands:
                c = rng.choice(cands)
                pairs.append(
                    {
                        "clean": r["text"],
                        "corrupted": c["text"],
                        "correct_idx": r["label"],
                        "incorrect_idx": c["label"],
                    }
                )
        with (OUT / f"eap_langid_{plang}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"])
            w.writeheader()
            w.writerows(pairs)
        stats[plang] = {"rows": len(recs), "pairs": len(pairs)}
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
