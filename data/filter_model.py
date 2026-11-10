"""Model-dependent split of the B1 pool: which facts Tiny Aya knows, per language.

For every prompt (shared chat format) we record the gold first-token probability and rank at the
answer position, the greedy top-1 token, and an 8-token greedy continuation. A fact is `known` in a
language when the continuation matches the gold label or one of its Wikidata aliases (see
`matches`); `false` when the continuation
is a confident non-gold answer (top-1 probability >= 0.3 and gold not in the continuation).
Outputs (data/b1/):
  model_eval_<lang>.jsonl   per-prompt record
  eap_known_<lang>.csv      discovery pairs where clean and corrupt subjects are both known
  false_facts.jsonl         facts wrong in English (any confidence; `false` flags confident ones,
  p>=0.3),
                            excluding the ambiguous `continent` relation (B1 correction targets)
  model_eval_summary.json   accuracy per language x relation
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from aya_circuits.config import load_config
from aya_circuits.manifest import Manifest
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_hf
from neb.behavioural import matches

HERE = Path(__file__).resolve().parent / "b1"


@torch.no_grad()
def score_prompts(
    model, tok, texts: list[str], labels: list[int], batch_size: int, max_new: int = 8
):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        logits = model(**enc).logits[:, -1].float()
        probs = logits.softmax(-1)
        gold = torch.tensor(labels[i : i + batch_size], device=probs.device)
        p_gold = probs.gather(1, gold[:, None])[:, 0]
        rank = (probs > p_gold[:, None]).sum(1)
        top = probs.argmax(-1)
        gen = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
        cont = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        out += [
            {
                "p_gold": p_gold[j].item(),
                "rank_gold": int(rank[j]),
                "top1_id": int(top[j]),
                "top1_p": probs[j, top[j]].item(),
                "top1_tok": tok.decode([int(top[j])]),
                "continuation": cont[j],
            }
            for j in range(len(cont))
        ]
    return out


def main() -> None:
    cfg = load_config()
    bs = int(cfg.get("batch_size", 16))
    mf, mw = Manifest("filter_model", cfg), MemWatch(cfg["memory_budget_gb"])
    model, tok = load_hf(cfg)
    summary, known_ids = {}, {}
    per_lang_eval = {}
    for lang in cfg["languages"]:
        recs = [json.loads(line) for line in (HERE / f"prompts_{lang}.jsonl").open()]
        mf.add_input(HERE / f"prompts_{lang}.jsonl")
        scored = score_prompts(
            model, tok, [r["text"] for r in recs], [r["label"] for r in recs], bs
        )
        mw.check()
        acc = defaultdict(lambda: [0, 0])
        with (HERE / f"model_eval_{lang}.jsonl").open("w") as f:
            for r, s in zip(recs, scored, strict=False):
                s["known"] = matches(s["continuation"], [r["object"], *r.get("aliases", [])])
                s["false"] = (not s["known"]) and s["top1_p"] >= 0.3
                acc[r["relation"]][0] += s["known"]
                acc[r["relation"]][1] += 1
                f.write(json.dumps({**r, **s}, ensure_ascii=False) + "\n")
        per_lang_eval[lang] = {
            (r["relation"], r["subject_qid"]): s for r, s in zip(recs, scored, strict=False)
        }
        known_ids[lang] = {k for k, s in per_lang_eval[lang].items() if s["known"]}
        summary[lang] = {rel: {"acc": round(a / n, 3), "n": n} for rel, (a, n) in acc.items()}
        summary[lang]["all"] = {
            "acc": round(sum(a for a, _ in acc.values()) / len(recs), 3),
            "n": len(recs),
        }
        # discovery pairs restricted to facts the model knows in this language (clean signal)
        rows = [json.loads(line) for line in (HERE / f"prompts_{lang}.jsonl").open()]
        by_key = {(r["relation"], r["subject_qid"]): r for r in rows}
        groups = defaultdict(list)
        for k, r in by_key.items():
            if k in known_ids[lang]:
                groups[(r["relation"], r["n_tok"])].append(r)
        rng, pairs = random.Random(0), []
        for k, r in by_key.items():
            if k not in known_ids[lang]:
                continue
            cands = [
                c
                for c in groups[(r["relation"], r["n_tok"])]
                if c["label"] != r["label"] and c["object_qid"] != r["object_qid"]
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
        with (HERE / f"eap_known_{lang}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"])
            w.writeheader()
            w.writerows(pairs)
        summary[lang]["eap_known_pairs"] = len(pairs)
        print(lang, json.dumps(summary[lang]), flush=True)

    # B1 targets: wrong in English (the edit language), gold labels exist everywhere by construction
    pool = {
        (r["relation"], r["subject_qid"]): r
        for r in (json.loads(line) for line in (HERE / "facts_pool.jsonl").open())
    }
    false_rows = []
    # `continent` is excluded as a correction target: Wikidata's P30 (North America) and the model's
    # "Central America" are both defensible, so a mismatch there is not a false fact.
    for k, s in per_lang_eval["en"].items():
        if k[0] == "continent":
            continue
        if not s["known"]:
            r = pool[k]
            false_rows.append(
                {
                    **r,
                    "model_answer_en": s["continuation"].strip(),
                    "top1_p_en": round(s["top1_p"], 3),
                    "known_in": [lang for lang in cfg["languages"] if k in known_ids[lang]],
                }
            )
    with (HERE / "false_facts.jsonl").open("w") as f:
        for r in false_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary["n_false_en"] = len(false_rows)
    summary["n_wrong_en_any_confidence"] = sum(not s["known"] for s in per_lang_eval["en"].values())
    (HERE / "model_eval_summary.json").write_text(json.dumps(summary, indent=2))
    mf.write(HERE / "model_eval_manifest.json", memory=mw.summary(), summary=summary)
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("n_")}))


if __name__ == "__main__":
    main()
