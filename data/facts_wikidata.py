"""Build the multilingual fact pool for B1 from Wikidata (labels in en, hi, ar, fr, sw on both
ends).

Relations (functional after filtering to subjects with exactly one value):
  capital   country -> capital city            P36
  continent country -> continent               P30
  language  country -> sole official language  P37
  currency  country -> sole currency           P38
  country   city (pop > 1M) -> country         P17
  head_of_state      country -> current head of state       P35 (no end date)
  head_of_government country -> current head of government  P6  (no end date)
Aliases in every language are fetched alongside labels so model answers can be matched against
alternative names (e.g. "Nuku'alofa" vs "Nukuʻalofa", "Yaren" vs "Yaren District").
Stage 1: SPARQL for (subject, object) QIDs per relation (cheap). Stage 2: labels for every QID in
the five languages through wbgetentities (50 ids per call). Rows missing any label are dropped.
Output: data/b1/facts_pool.jsonl. The model-dependent filtering (which facts Tiny Aya gets wrong)
happens later in data/filter_model.py."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

LANGS = ["en", "hi", "ar", "fr", "sw"]
UA = "circuit-steering-research/0.1 (mailto:mohammadsadiq4950@gmail.com)"
SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
OUT = Path(__file__).resolve().parent / "b1" / "facts_pool.jsonl"

COUNTRY = "?s wdt:P31 wd:Q3624078 ."
CITY = "?s wdt:P31/wdt:P279* wd:Q515 ; wdt:P1082 ?pop . FILTER(?pop > 300000)"
RELATIONS = {
    "capital": (COUNTRY, "P36"),
    "continent": (COUNTRY, "P30"),
    "language": (COUNTRY, "P37"),
    "currency": (COUNTRY, "P38"),
    "country": (CITY, "P17"),
    "head_of_state": (COUNTRY, "P35"),
    "head_of_government": (COUNTRY, "P6"),
}
# Position-holder properties are multi-valued over time; keep only the statement without an end
# date.
CURRENT = {"P35", "P6"}


def _get(url: str, params: dict) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(
                url,
                params=params,
                headers={"Accept": "application/json", "User-Agent": UA},
                timeout=120,
            )
            if r.status_code == 200:
                return r.json()
            print(f"HTTP {r.status_code}, retrying", file=sys.stderr)
        except requests.RequestException as e:
            print(f"{e}, retrying", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def pairs(subject_clause: str, prop: str) -> list[tuple[str, str]]:
    """(subject_qid, object_qid) for subjects with exactly one (current) value of prop."""
    if prop in CURRENT:
        body = (
            f"{subject_clause} ?s p:{prop} ?st . ?st ps:{prop} ?o . ?st wikibase:rank ?rank . "
            "FILTER(?rank != wikibase:DeprecatedRank) FILTER NOT EXISTS { ?st pq:P582 ?end }"
        )
    else:
        body = f"{subject_clause} ?s wdt:{prop} ?o ."
    q = f"""SELECT ?s (SAMPLE(?o) AS ?obj) WHERE {{ {body} }}
            GROUP BY ?s HAVING (COUNT(DISTINCT ?o) = 1)"""
    data = _get(SPARQL, {"query": q, "format": "json"})

    def qid(u):
        return u.rsplit("/", 1)[-1]

    return [(qid(b["s"]["value"]), qid(b["obj"]["value"])) for b in data["results"]["bindings"]]


def labels(qids: list[str]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, list[str]]]]:
    """Per QID: {lang: label} and {lang: [aliases]}."""
    out: dict[str, dict[str, str]] = {}
    ali: dict[str, dict[str, list[str]]] = {}
    qids = sorted(set(qids))
    for i in range(0, len(qids), 50):
        data = _get(
            API,
            {
                "action": "wbgetentities",
                "ids": "|".join(qids[i : i + 50]),
                "props": "labels|aliases",
                "languages": "|".join(LANGS),
                "format": "json",
            },
        )
        for q, ent in data.get("entities", {}).items():
            labs = ent.get("labels", {})
            out[q] = {lang: labs[lang]["value"] for lang in LANGS if lang in labs}
            ali[q] = {
                lang: [a["value"] for a in ent.get("aliases", {}).get(lang, [])] for lang in LANGS
            }
    return out, ali


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rel_pairs = {rel: pairs(*spec) for rel, spec in RELATIONS.items()}
    for rel, ps in rel_pairs.items():
        print(f"{rel}: {len(ps)} (s,o) pairs", file=sys.stderr)
    lab, ali = labels([q for ps in rel_pairs.values() for p in ps for q in p])
    rows, dropped = [], 0
    for rel, ps in rel_pairs.items():
        for s, o in ps:
            ls, lo = dict(lab.get(s, {})), dict(lab.get(o, {}))
            fallback = []
            # Person names are not translated in Swahili or French; Wikidata simply lacks the label.
            if rel in ("head_of_state", "head_of_government") and "en" in lo:
                for lang in ("sw", "fr"):
                    if lang not in lo:
                        lo[lang] = lo["en"]
                        fallback.append(lang)
            if len(ls) < len(LANGS) or len(lo) < len(LANGS):
                dropped += 1
                continue
            rows.append(
                {
                    "relation": rel,
                    "subject_qid": s,
                    "object_qid": o,
                    "subject": ls,
                    "object": lo,
                    "object_aliases": ali.get(o, {}),
                    "label_fallback": fallback,
                }
            )
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(rows)} rows ({dropped} dropped for missing labels) -> {OUT}", file=sys.stderr
    )


if __name__ == "__main__":
    main()
