"""One-hop ripple prompts for the B1 false facts (L1, brief §3.3 item 4; RippleEdits, Cohen et al.
2023): questions whose correct answer changes as a logical consequence of the corrected
fact.

Per false fact (subject s, relation r, corrected object o), English, gold labels and aliases from
the Wikidata pool (data/b1/facts_pool.jsonl):
  country   (city s -> country o)        compositional: capital of o; continent of o
  capital   (country s -> city o)        inverse: which country is the city o in -> s
  head_of_state / head_of_government     inverse: which country does o lead -> s
                                         compositional: capital of the country o leads -> capital(s)
  language, currency                     none (the inverse is not functional)
A row is emitted only when the pool has the gold value. Output: data/b1/ripple_en.jsonl, one row
per item with the fact key (relation, subject_qid) so the audit evaluates exactly the cell's fold.
Run: uv run python data/b1_ripple.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent / "b1"
Q = {
    "capital_of": "What is the capital city of the country where the city {o} is located? Answer "
    "with only the city name.",
    "continent_of": "Which continent is the city {o} in? Answer with only the continent name.",
    "inverse_capital": "Which country is the following city in: {o}? Answer with only the country "
    "name.",
    "inverse_head_of_state": "Of which country is {o} the current head of state? Answer with only "
    "the country name.",
    "inverse_head_of_government": "Of which country is {o} the current head of government? Answer "
    "with only the country name.",
    "capital_via_leader": "What is the capital city of the country whose current {role} is {o}? "
    "Answer with only the city name.",
}


def main() -> None:
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    pool = [json.loads(line) for line in (HERE / "facts_pool.jsonl").open()]
    by = {(r["relation"], r["subject_qid"]): r for r in pool}
    aliases: dict[str, list[str]] = {}  # qid -> English aliases, from rows where it is an object
    for r in pool:
        aliases.setdefault(r["object_qid"], list(r.get("object_aliases", {}).get("en", [])))
    false = [json.loads(line) for line in (HERE / "false_facts.jsonl").open()]
    out = []

    def item(f, kind, question, gold_label, gold_qid):
        out.append(
            {
                "fact_relation": f["relation"],
                "fact_subject_qid": f["subject_qid"],
                "relation": f"ripple_{kind}",
                "subject_qid": f["subject_qid"],
                "lang": "en",
                "text": chat_wrap(tok, question, cfg["chat_format"]),
                "object": gold_label,
                "aliases": aliases.get(gold_qid, []),
            }
        )

    for f in false:
        s_en, o_en, o_q, s_q = (
            f["subject"]["en"],
            f["object"]["en"],
            f["object_qid"],
            f["subject_qid"],
        )
        if f["relation"] == "country":
            if cap := by.get(("capital", o_q)):
                item(
                    f,
                    "capital_of",
                    Q["capital_of"].format(o=s_en),
                    cap["object"]["en"],
                    cap["object_qid"],
                )
            if cont := by.get(("continent", o_q)):
                item(
                    f,
                    "continent_of",
                    Q["continent_of"].format(o=s_en),
                    cont["object"]["en"],
                    cont["object_qid"],
                )
        elif f["relation"] == "capital":
            item(f, "inverse_capital", Q["inverse_capital"].format(o=o_en), s_en, s_q)
        elif f["relation"] in ("head_of_state", "head_of_government"):
            item(
                f,
                f"inverse_{f['relation']}",
                Q[f"inverse_{f['relation']}"].format(o=o_en),
                s_en,
                s_q,
            )
            if cap := by.get(("capital", s_q)):
                role = f["relation"].replace("_", " ")
                item(
                    f,
                    "capital_via_leader",
                    Q["capital_via_leader"].format(role=role, o=o_en),
                    cap["object"]["en"],
                    cap["object_qid"],
                )
    with (HERE / "ripple_en.jsonl").open("w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        len(out),
        "ripple items for",
        len({(r["fact_relation"], r["subject_qid"]) for r in out}),
        "of",
        len(false),
        "false facts;",
        dict(Counter(r["relation"] for r in out)),
    )


if __name__ == "__main__":
    main()
