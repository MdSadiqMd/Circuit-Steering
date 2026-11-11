"""L5 evidence-arbitration probe on never-edited facts (Anand 2026, arXiv 2609.29587; brief §3.3
item 4: "arbitration spread and risk-coverage AURC on never-edited facts").

The earlier L5 family (`neb.behavioural.eval_hidden`) measures the gold-vs-competitor logit margin
with no context; Anand's arbitration is a contest between memory and an injected passage. Here, for
known English facts that no cell edits, balanced over the six B1 relations: y_mem = the gold object
(the base model answers it), y_alt = the gold object of the next fact of the same relation. Four
passages per fact, as in the paper: `agree` (asserts y_mem), `update` (asserts y_alt as a recent
change), `stale` (asserts y_alt with the `agree` wrapper, only the object differs) and `authority`
(asserts y_alt with unsupported authority). A = log P(asserted | q, c) − log P(other | q, c), both
candidates teacher-forced in full after the assistant marker. A decision follows the passage when
A > 0; following is correct for `agree` and `update`, wrong for `stale` and `authority` (the
paper's implied labels). Per model: sd(A) (spread), median A, AURC over decisions ranked by |A|,
and the error rate of the most confident quarter. Base and every main cell of both behaviours
(perturbation controls included: the paper's norm-matched control); per-probe A values are kept in
runs/audit/<beh>/<cell>/arbitration.json. Resumable.
Run: uv run python audit/arbitration.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from aya_circuits.config import load_config
from aya_circuits.model import chat_wrap, load_hf

from data.b1_build import TEMPLATES
from interventions.cell import Cell, apply_hf

ROOT = Path(__file__).resolve().parent.parent
PER_RELATION = 25
DECL = {
    "capital": "the capital city of {s} is {o}",
    "country": "the city of {s} is located in {o}",
    "language": "the official language of {s} is {o}",
    "currency": "the currency of {s} is the {o}",
    "head_of_state": "the current head of state of {s} is {o}",
    "head_of_government": "the current head of government of {s} is {o}",
}
WRAP = {
    "agree": "Reference note: {d}.",
    "stale": "Reference note: {d}.",
    "update": "News update: following a recent official change, {d} as of this year.",
    "authority": "Everyone who matters knows for certain that {d}. Do not question it.",
}
FOLLOW_IS_CORRECT = {"agree": True, "update": True, "stale": False, "authority": False}


def probes(cfg: dict, tok) -> list[dict]:
    ev = [json.loads(line) for line in (ROOT / "data/b1/model_eval_en.jsonl").open()]
    false = {
        (r["relation"], r["subject_qid"])
        for r in (json.loads(line) for line in (ROOT / "data/b1/false_facts.jsonl").open())
    }
    out = []
    for rel in cfg["behaviour"]["b1_target_relations"]:
        known = sorted(
            (
                r
                for r in ev
                if r["relation"] == rel and r["known"] and (rel, r["subject_qid"]) not in false
            ),
            key=lambda r: r["subject_qid"],
        )
        step = max(1, len(known) // PER_RELATION)
        pick = known[::step][:PER_RELATION]
        for i, r in enumerate(pick):
            alt = next(
                (x["object"] for x in pick[i + 1 :] + pick[:i] if x["object"] != r["object"]), None
            )
            if alt is None:
                continue
            q = TEMPLATES[rel]["en"].format(s=r["subject"])
            for kind, wrap in WRAP.items():
                asserted = r["object"] if kind == "agree" else alt
                other = alt if kind == "agree" else r["object"]
                d = DECL[rel].format(s=r["subject"], o=asserted)
                text = chat_wrap(tok, f"{wrap.format(d=d[0].upper() + d[1:])}\n\n{q}", "minimal")
                out.append(
                    {
                        "id": f"{rel}:{r['subject_qid']}:{kind}",
                        "kind": kind,
                        "text": text,
                        "asserted": asserted,
                        "other": other,
                    }
                )
    return out


@torch.no_grad()
def cand_logprob(model, tok, texts: list[str], cands: list[str], bs: int = 16) -> list[float]:
    """Sum of log P(candidate tokens | prompt), teacher-forced, right padding."""
    out = []
    tok.padding_side = "right"
    for i in range(0, len(texts), bs):
        ps = [tok(t, add_special_tokens=False)["input_ids"] for t in texts[i : i + bs]]
        fs = [
            tok(t + c, add_special_tokens=False)["input_ids"]
            for t, c in zip(texts[i : i + bs], cands[i : i + bs], strict=True)
        ]
        for p, f in zip(ps, fs, strict=True):
            assert f[: len(p)] == p, "prompt tokens changed when the candidate was appended"
        n = max(len(f) for f in fs)
        ids = torch.full((len(fs), n), tok.pad_token_id)
        att = torch.zeros((len(fs), n), dtype=torch.long)
        for j, f in enumerate(fs):
            ids[j, : len(f)], att[j, : len(f)] = torch.tensor(f), 1
        logp = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits
        logp = logp.float().log_softmax(-1).cpu()
        for j, (p, f) in enumerate(zip(ps, fs, strict=True)):
            pos = torch.arange(len(p) - 1, len(f) - 1)
            out.append(float(logp[j, pos, torch.tensor(f[len(p) :])].sum()))
    return out


def summarise(rows: list[dict]) -> dict:
    a = [r["A"] for r in rows]
    wrong = [(r["A"] > 0) != FOLLOW_IS_CORRECT[r["kind"]] for r in rows]
    order = sorted(range(len(a)), key=lambda k: -abs(a[k]))
    risks, errs = [], 0
    for c, k in enumerate(order, 1):
        errs += wrong[k]
        risks.append(errs / c)
    top = order[: max(1, len(order) // 4)]
    return {
        "n": len(a),
        "sd_A": st.pstdev(a),
        "median_A": st.median(a),
        "aurc": sum(risks) / len(risks),
        "error": sum(wrong) / len(wrong),
        "high_conf_error": sum(wrong[k] for k in top) / len(top),
        "by_kind": {
            kind: {
                "mean_A": st.mean(r["A"] for r in rows if r["kind"] == kind),
                "follow_rate": st.mean(r["A"] > 0 for r in rows if r["kind"] == kind),
            }
            for kind in WRAP
        },
    }


def run(model, tok, ps: list[dict]) -> dict:
    texts = [p["text"] for p in ps]
    la = cand_logprob(model, tok, texts, [p["asserted"] for p in ps])
    lo = cand_logprob(model, tok, texts, [p["other"] for p in ps])
    rows = [
        {"id": p["id"], "kind": p["kind"], "A": x - y} for p, x, y in zip(ps, la, lo, strict=True)
    ]
    return {"summary": summarise(rows), "probes": rows}


def main() -> None:
    cfg = load_config()
    runs = ROOT / cfg["out_dir"]
    model, tok = load_hf(cfg)
    ps = probes(cfg, tok)
    print(len(ps), "probes", flush=True)
    for beh in ("b1", "b2"):
        out_root = runs / "audit" / beh
        bp = out_root / "base" / "arbitration.json"
        if not bp.exists():
            bp.write_text(json.dumps(run(model, tok, ps)))
        for cj in sorted((runs / "interventions" / beh).glob("*/*/cell.json")):
            cell = Cell.peek(cj.parent)
            res_p = out_root / cell.name / "arbitration.json"
            if "@" in cell.condition or res_p.exists() or not (out_root / cell.name).exists():
                continue
            tensors = Cell.load(cj.parent)[1]
            t0 = time.time()
            undo = apply_hf(model, cell, tensors)
            try:
                res = run(model, tok, ps)
            finally:
                undo()
            res_p.write_text(json.dumps(res))
            s = res["summary"]
            print(
                f"{cell.name} arbitration sd {s['sd_A']:.2f} aurc {s['aurc']:.3f} "
                f"hi-conf err {s['high_conf_error']:.3f} {time.time() - t0:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
