"""B1 prompt sets from the Wikidata fact pool, in five languages, in the shared chat format.

For every (relation, subject) row and language we emit:
  prompt        the question whose next token after the assistant marker should be the object
  answer        object label in that language (first token is the discovery label)
Corrupt partner for EAP: a different subject of the same relation and language whose *subject label*
has the same token count, so clean and corrupt prompts align position-for-position
(CircuitKIT `tokenize_batch_pair` requires equal n_pos per pair). Pairs whose first answer tokens
coincide are dropped (zero logit-diff signal, same rule as CircuitKIT `_materialise_eap_csv`).

Outputs (data/b1/):
  prompts_<lang>.jsonl         all rows: relation, qids, prompt, answer, subject, object
  eap_<lang>.csv               clean, corrupted, correct_idx, incorrect_idx  (chat-wrapped)
The model-dependent split into "known" and "false" (model-wrong) facts is data/filter_model.py.
"""

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

HERE = Path(__file__).resolve().parent / "b1"

# Question form with "the following X: {s}" keeps every language free of article/gender/case
# agreement
# with the subject, so one template serves every subject and the five prompts stay parallel. v2
# after
# the first model run showed the imperative form being echoed in ar/sw/hi (docs/notes.md).
TEMPLATES = {
    "capital": {
        "en": "What is the capital city of the following country: {s}? Answer with only the city "
        "name.",
        "fr": "Quelle est la capitale du pays suivant : {s} ? Réponds uniquement par le nom de la "
        "ville.",
        "hi": "निम्नलिखित देश की राजधानी कौन सी है: {s}? उत्तर में केवल शहर का नाम लिखें।",
        "ar": "ما هي عاصمة الدولة التالية: {s}؟ أجب باسم المدينة فقط.",
        "sw": "Mji mkuu wa nchi ifuatayo ni upi: {s}? Jibu kwa jina la mji pekee.",
    },
    "continent": {
        "en": "Which continent is the following country in: {s}? Answer with only the continent "
        "name.",
        "fr": "Sur quel continent se trouve le pays suivant : {s} ? Réponds uniquement par le nom "
        "du continent.",
        "hi": "निम्नलिखित देश किस महाद्वीप में है: {s}? उत्तर में केवल महाद्वीप का नाम लिखें।",
        "ar": "في أي قارة تقع الدولة التالية: {s}؟ أجب باسم القارة فقط.",
        "sw": "Nchi ifuatayo iko katika bara gani: {s}? Jibu kwa jina la bara pekee.",
    },
    "language": {
        "en": "What is the official language of the following country: {s}? Answer with only the "
        "language name.",
        "fr": "Quelle est la langue officielle du pays suivant : {s} ? Réponds uniquement par le "
        "nom de la langue.",
        "hi": "निम्नलिखित देश की आधिकारिक भाषा कौन सी है: {s}? उत्तर में केवल भाषा का नाम लिखें।",
        "ar": "ما هي اللغة الرسمية للدولة التالية: {s}؟ أجب باسم اللغة فقط.",
        "sw": "Lugha rasmi ya nchi ifuatayo ni ipi: {s}? Jibu kwa jina la lugha pekee.",
    },
    "currency": {
        "en": "What is the currency of the following country: {s}? Answer with only the currency "
        "name.",
        "fr": "Quelle est la monnaie du pays suivant : {s} ? Réponds uniquement par le nom de la "
        "monnaie.",
        "hi": "निम्नलिखित देश की मुद्रा कौन सी है: {s}? उत्तर में केवल मुद्रा का नाम लिखें।",
        "ar": "ما هي عملة الدولة التالية: {s}؟ أجب باسم العملة فقط.",
        "sw": "Sarafu ya nchi ifuatayo ni ipi: {s}? Jibu kwa jina la sarafu pekee.",
    },
    "country": {
        "en": "Which country is the following city in: {s}? Answer with only the country name.",
        "fr": "Dans quel pays se trouve la ville suivante : {s} ? Réponds uniquement par le nom "
        "du pays.",
        "hi": "निम्नलिखित शहर किस देश में है: {s}? उत्तर में केवल देश का नाम लिखें।",
        "ar": "في أي دولة تقع المدينة التالية: {s}؟ أجب باسم الدولة فقط.",
        "sw": "Mji ufuatao uko katika nchi gani: {s}? Jibu kwa jina la nchi pekee.",
    },
    "head_of_state": {
        "en": "Who is the current head of state of the following country: {s}? Answer with only "
        "the person's name.",
        "fr": "Qui est l'actuel chef d'État du pays suivant : {s} ? Réponds uniquement par le nom "
        "de la personne.",
        "hi": "निम्नलिखित देश का वर्तमान राष्ट्राध्यक्ष कौन है: {s}? उत्तर में केवल व्यक्ति का नाम लिखें।",
        "ar": "من هو رئيس الدولة الحالي للدولة التالية: {s}؟ أجب باسم الشخص فقط.",
        "sw": "Mkuu wa nchi wa sasa wa nchi ifuatayo ni nani: {s}? Jibu kwa jina la mtu pekee.",
    },
    "head_of_government": {
        "en": "Who is the current head of government of the following country: {s}? Answer with "
        "only the person's name.",
        "fr": "Qui est l'actuel chef du gouvernement du pays suivant : {s} ? Réponds uniquement "
        "par le nom de la personne.",
        "hi": "निम्नलिखित देश का वर्तमान शासनाध्यक्ष कौन है: {s}? उत्तर में केवल व्यक्ति का नाम लिखें।",
        "ar": "من هو رئيس الحكومة الحالي للدولة التالية: {s}؟ أجب باسم الشخص فقط.",
        "sw": "Mkuu wa serikali wa sasa wa nchi ifuatayo ni nani: {s}? Jibu kwa jina la mtu pekee.",
    },
}


# One paraphrase per relation and language (L1 paraphrase locality); same answer, different wording.
PARAPHRASE = {
    "capital": {
        "en": "Tell me the capital of {s}. Reply with the city name only.",
        "fr": "Dis-moi la capitale de ce pays : {s}. Réponds seulement avec le nom de la ville.",
        "hi": "इस देश की राजधानी बताएं: {s}। उत्तर केवल शहर का नाम हो।",
        "ar": "أخبرني بعاصمة هذه الدولة: {s}. اكتب اسم المدينة فقط.",
        "sw": "Niambie mji mkuu wa nchi hii: {s}. Andika jina la mji tu.",
    },
    "continent": {
        "en": "Tell me the continent that {s} belongs to. Reply with the continent name only.",
        "fr": "Dis-moi à quel continent appartient ce pays : {s}. Réponds seulement avec le nom "
        "du continent.",
        "hi": "इस देश का महाद्वीप बताएं: {s}। उत्तर केवल महाद्वीप का नाम हो।",
        "ar": "أخبرني بقارة هذه الدولة: {s}. اكتب اسم القارة فقط.",
        "sw": "Niambie bara la nchi hii: {s}. Andika jina la bara tu.",
    },
    "language": {
        "en": "Tell me the official language of {s}. Reply with the language name only.",
        "fr": "Dis-moi la langue officielle de ce pays : {s}. Réponds seulement avec le nom de la "
        "langue.",
        "hi": "इस देश की आधिकारिक भाषा बताएं: {s}। उत्तर केवल भाषा का नाम हो।",
        "ar": "أخبرني باللغة الرسمية لهذه الدولة: {s}. اكتب اسم اللغة فقط.",
        "sw": "Niambie lugha rasmi ya nchi hii: {s}. Andika jina la lugha tu.",
    },
    "currency": {
        "en": "Tell me the currency used in {s}. Reply with the currency name only.",
        "fr": "Dis-moi la monnaie de ce pays : {s}. Réponds seulement avec le nom de la monnaie.",
        "hi": "इस देश की मुद्रा बताएं: {s}। उत्तर केवल मुद्रा का नाम हो।",
        "ar": "أخبرني بعملة هذه الدولة: {s}. اكتب اسم العملة فقط.",
        "sw": "Niambie sarafu ya nchi hii: {s}. Andika jina la sarafu tu.",
    },
    "country": {
        "en": "Tell me the country where the city of {s} is located. Reply with the country name "
        "only.",
        "fr": "Dis-moi dans quel pays se situe cette ville : {s}. Réponds seulement avec le nom "
        "du pays.",
        "hi": "इस शहर का देश बताएं: {s}। उत्तर केवल देश का नाम हो।",
        "ar": "أخبرني بالدولة التي تقع فيها هذه المدينة: {s}. اكتب اسم الدولة فقط.",
        "sw": "Niambie nchi ambayo mji huu uko: {s}. Andika jina la nchi tu.",
    },
    "head_of_state": {
        "en": "Tell me who currently serves as head of state of {s}. Reply with the name only.",
        "fr": "Dis-moi qui est actuellement le chef d'État de ce pays : {s}. Réponds seulement "
        "avec le nom.",
        "hi": "इस देश के वर्तमान राष्ट्राध्यक्ष का नाम बताएं: {s}। उत्तर केवल नाम हो।",
        "ar": "أخبرني من يشغل حاليًا منصب رئيس الدولة في هذه الدولة: {s}. اكتب الاسم فقط.",
        "sw": "Niambie ni nani mkuu wa nchi wa sasa wa nchi hii: {s}. Andika jina tu.",
    },
    "head_of_government": {
        "en": "Tell me who currently serves as head of government of {s}. Reply with the name "
        "only.",
        "fr": "Dis-moi qui est actuellement le chef du gouvernement de ce pays : {s}. Réponds "
        "seulement avec le nom.",
        "hi": "इस देश के वर्तमान शासनाध्यक्ष का नाम बताएं: {s}। उत्तर केवल नाम हो।",
        "ar": "أخبرني من يشغل حاليًا منصب رئيس الحكومة في هذه الدولة: {s}. اكتب الاسم فقط.",
        "sw": "Niambie ni nani mkuu wa serikali wa sasa wa nchi hii: {s}. Andika jina tu.",
    },
}


def first_token(tok, prompt_text: str, answer: str) -> int:
    """First answer token when the answer continues the prompt (joint tokenisation, as CircuitKIT's
    `check_answer_discriminative` does, so BPE boundary effects cannot shift the label)."""
    base = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full = tok(prompt_text + answer, add_special_tokens=False)["input_ids"]
    assert full[: len(base)] == base, "prompt tokens changed when the answer was appended"
    return full[len(base)]


def main() -> None:
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    rows = [json.loads(line) for line in (HERE / "facts_pool.jsonl").open()]
    rng = random.Random(0)
    stats = {}
    for lang in cfg["languages"]:
        recs = []
        for r in rows:
            prompt = TEMPLATES[r["relation"]][lang].format(s=r["subject"][lang])
            text = chat_wrap(tok, prompt, cfg["chat_format"])
            recs.append(
                {
                    **{k: r[k] for k in ("relation", "subject_qid", "object_qid")},
                    "lang": lang,
                    "subject": r["subject"][lang],
                    "object": r["object"][lang],
                    "aliases": r.get("object_aliases", {}).get(lang, []),
                    "prompt": prompt,
                    "text": text,
                    "paraphrase_text": chat_wrap(
                        tok,
                        PARAPHRASE[r["relation"]][lang].format(s=r["subject"][lang]),
                        cfg["chat_format"],
                    ),
                    "n_tok": len(tok(text, add_special_tokens=False)["input_ids"]),
                    "label": first_token(tok, text, r["object"][lang]),
                }
            )
        with (HERE / f"prompts_{lang}.jsonl").open("w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # pair within (relation, token length); partner must have a different first answer token
        groups = defaultdict(list)
        for rec in recs:
            groups[(rec["relation"], rec["n_tok"])].append(rec)
        pairs, dropped = [], 0
        for rec in recs:
            cands = [
                c
                for c in groups[(rec["relation"], rec["n_tok"])]
                if c["label"] != rec["label"] and c["object_qid"] != rec["object_qid"]
            ]
            if not cands:
                dropped += 1
                continue
            c = rng.choice(cands)
            pairs.append(
                {
                    "clean": rec["text"],
                    "corrupted": c["text"],
                    "correct_idx": rec["label"],
                    "incorrect_idx": c["label"],
                }
            )
        with (HERE / f"eap_{lang}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"])
            w.writeheader()
            w.writerows(pairs)
        stats[lang] = {
            "rows": len(recs),
            "pairs": len(pairs),
            "dropped_no_partner": dropped,
            "mean_tokens": round(sum(r["n_tok"] for r in recs) / len(recs), 1),
        }
    (HERE / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
