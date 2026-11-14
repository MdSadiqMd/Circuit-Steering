"""HF Space: two circuit-restricted refusal adapters for Tiny Aya, and the audit that checked them.

Runs from published artifacts, no GPU: the `circuit-steering-audit` dataset repo (or a local copy
of it under CS_AUDIT_DIR). Three tabs:
  The adapters          A22 and A24 before/after on held-out test prompts, per language, with
                        intervals; base and adapter replies to benign prompts side by side
  Does the circuit matter?  the same adapters restricted to random, inverse and all components
  The 393 earlier runs  every intervention cell of the main study with its audit scorecard
Replies to harmful prompts are never shown or published; only judgments are.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import gradio as gr

NS = os.environ.get("HF_NAMESPACE", "MdSadiqMd")
RELEASED = "b2_lora_circuit30_s0"
LANGS = ["en", "hi", "ar", "fr", "sw"]
NAME = {"en": "English", "hi": "Hindi", "ar": "Arabic", "fr": "French", "sw": "Swahili"}
if os.environ.get("CS_AUDIT_DIR"):
    ROOT = Path(os.environ["CS_AUDIT_DIR"])
else:
    from huggingface_hub import snapshot_download

    ROOT = Path(
        snapshot_download(
            f"{NS}/circuit-steering-audit",
            repo_type="dataset",
            allow_patterns=["*.json", "*.csv", "standard_checks.md"],
        )
    )


def read(rel: str):
    p = next(ROOT.glob(f"**/{rel}"), None)
    return json.loads(p.read_text()) if p else None


RUNS = {
    "Second adapter, A24 (recommended: keeps fluency)": "a24",
    "First adapter, A22 (stronger effect)": "a22",
}
SUMMARY = {r: read(f"{r}_b2.json") or {} for r in RUNS.values()}
EVAL = {r: (read(f"{r}/eval/base.json"), read(f"{r}/eval/{RELEASED}.json")) for r in RUNS.values()}
JUDGE = {
    r: (read(f"{r}/judge/base.json"), read(f"{r}/judge/{RELEASED}.json")) for r in RUNS.values()
}


def headline(label: str) -> list[list[str]]:
    rows = [["outcome", "hi/ar/fr/sw", *(NAME[x] for x in LANGS)]]
    for name, by in SUMMARY[RUNS[label]].get("versus_base", {}).get(RELEASED, {}).items():
        if isinstance(by, dict):
            rows.append([name, by.get("xl", ""), *(by.get(x, "") for x in LANGS)])
    return rows


def nothing_broke(label: str) -> list[list[str]]:
    a = SUMMARY[RUNS[label]]
    rows = [["check", "base → adapter (Δ [95 % CI]), or ratio"]]
    rows += [[k, str(v)] for k, v in a.get("audit", {}).get(RELEASED, {}).items()]
    flu = a.get("fluency", {}).get(RELEASED, {})
    if flu:
        per = " · ".join(f"{k} {flu[k]:.2f}" for k in LANGS)
        rows.append(["perplexity ratio, unseen sentences", f"{flu['mean']:.3f} ({per})"])
    end = a.get("ended", {}).get(RELEASED, {})
    if end:
        rows.append(
            [
                "replies to harmful prompts that are only the refusal",
                " · ".join(f"{k} {end[k]:.0%}" for k in LANGS),
            ]
        )
    return rows


def labels(label: str, lang: str) -> list[list[str]]:
    """Judge labels of replies to the harmful test prompts, base against adapter."""
    out = [["label", "base model", "with adapter"]]
    b, c = JUDGE[RUNS[label]]
    if not (b and c):
        return out
    b, c = b["harmful_test"][lang], c["harmful_test"][lang]
    return out + [
        [lab, sum(v == lab for v in b.values()), sum(v == lab for v in c.values())]
        for lab in ("comply", "safe", "refuse")
    ]


def benign(label: str, lang: str, only_changed: bool) -> list[list[str]]:
    out = [["prompt", "base reply", "adapter reply", "base refused", "adapter refused"]]
    b, c = EVAL[RUNS[label]]
    if not (b and c):
        return out
    b, c = b["benign_test"][lang], c["benign_test"][lang]
    for sid, prompt in b.get("prompts", {}).items():
        if only_changed and b["scores"][sid] == c["scores"][sid]:
            continue
        out.append(
            [
                prompt,
                b["outputs"][sid],
                c["outputs"][sid],
                "yes" if b["scores"][sid] else "no",
                "yes" if c["scores"][sid] else "no",
            ]
        )
    return out


def conditions() -> list[list[str]]:
    keys = [
        ("comply_xl", "harmful comply, hi/ar/fr/sw"),
        ("comply_en", "harmful comply, English"),
        ("benign_refusal", "benign refusal, 5 languages"),
        ("wikitext_ppl_ratio", "WikiText perplexity ratio"),
        ("fluency_mean", "perplexity ratio, 5 languages"),
    ]
    rows = [["run · condition", *(k[1] for k in keys)]]
    base = SUMMARY["a22"].get("base", {})
    rows.append(["base model", *(f"{base[k]:.3f}" if k in base else "1.000" for k, _ in keys)])
    for run in ("a22", "a24"):
        for cond, m in SUMMARY[run].get("conditions", {}).items():
            rows.append(
                [f"{run.upper()} · {cond}", *(f"{m[k]:.3f}" if k in m else "–" for k, _ in keys)]
            )
    return rows


def contrasts() -> str:
    out = []
    for run in ("a22", "a24"):
        out += [
            f"- **{run.upper()}, {k}**: {v}" for k, v in SUMMARY[run].get("contrasts", {}).items()
        ]
    return "\n".join(out)


def cells(beh: str) -> list[str]:
    return sorted(p.name for p in (ROOT / beh).glob("*_s*") if (p / "hf.json").exists())


def scorecard(beh: str) -> list[list[str]]:
    p = next(ROOT.glob(f"**/{beh}_cells.csv"), None)
    if not p:
        return [["no scorecard yet"]]
    keys = [
        "cell",
        "efficacy",
        "l3_mean_abs_delta",
        "l1_neigh_en_delta",
        "l2_benign_refusal_delta_en",
        "l1_wikitext_ppl_ratio",
        "l5_top1_error_delta",
        "delta_norm",
    ]
    return [keys] + [[str(r.get(k, ""))[:10] for k in keys] for r in csv.DictReader(p.open())]


def compare(beh: str, cell: str, family: str) -> list[list[str]]:
    base = json.loads((ROOT / beh / "base" / "hf.json").read_text())
    new = json.loads((ROOT / beh / cell / "hf.json").read_text())
    out = [["sample", "base", "intervened", "base score", "new score"]]
    hidden = "(reply withheld: harmful prompt; judgment shown)"
    for k in [k for k in new if k.startswith(family)][:3]:
        b, n = base.get(k, {}), new[k]
        out += [
            [
                sid,
                b.get("outputs", {}).get(sid, hidden),
                n.get("outputs", {}).get(sid, hidden),
                str(b.get("scores", {}).get(sid)),
                str(n["scores"].get(sid)),
            ]
            for sid in list(n.get("scores", {}))[:8]
        ]
    return out


INTRO = """# Two refusal adapters for Tiny Aya, and the audit that checked them

Tiny Aya helps with a harmful request 12.4 % of the time in Hindi, Arabic, French and Swahili and
8.8 % in English. These two LoRA adapters bring that to 3.8–7.1 % and 1.9–4.4 %. They are trained
only on the top 30 % of components by refusal-circuit score (149 of 576 heads, 35 of 36 MLPs).
Every number is on held-out test prompts, and two of the three harm categories were never trained
on. Four of the five parts of our audit were run on both adapters. It found two defects in the
first adapter (A22): worse fluency, and refusals that do not stop. The second (A24) fixes them.
No model runs here: the page shows stored results.
"""
CIRCUIT = """The honest part. We trained the same adapter on random components (same count per
layer), on every component and, for A22, on the lowest-scoring components. If the circuit were
what makes the adapter work, it should beat the random sets. At 30 % it does not: the differences
are about a point or less and, in A24, indistinguishable from zero. At 10 % the circuit adapter
complies 2 points less but refuses 4 points more harmless prompts, so it moves along the same
trade-off. What works is the training recipe, and what we recommend reusing is the audit.
"""

with gr.Blocks(title="Refusal adapters for Tiny Aya, audited") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("The adapters"):
        which = gr.Radio(list(RUNS), value=next(iter(RUNS)), label="adapter")
        gr.Markdown(
            "### Before → after on held-out test prompts\n"
            "Lower is better for `comply` and for benign refusal."
        )
        head = gr.Dataframe(value=headline(next(iter(RUNS))), wrap=True)
        gr.Markdown("### Did anything else break?")
        broke = gr.Dataframe(value=nothing_broke(next(iter(RUNS))), wrap=True)
        gr.Markdown(
            "### What happened to replies to harmful prompts\n"
            "Judge labels only; the replies themselves are not published."
        )
        lang = gr.Dropdown(LANGS, value="hi", label="language")
        lab = gr.Dataframe(value=labels(next(iter(RUNS)), "hi"))
        gr.Markdown("### Benign prompts that look harmful: base reply against adapter reply")
        changed = gr.Checkbox(value=False, label="only prompts where the refusal judgment changed")
        ben = gr.Dataframe(value=benign(next(iter(RUNS)), "hi", False), wrap=True)
        which.change(headline, which, head)
        which.change(nothing_broke, which, broke)
        for trigger in (which, lang):
            trigger.change(labels, [which, lang], lab)
        for trigger in (which, lang, changed):
            trigger.change(benign, [which, lang, changed], ben)
    with gr.Tab("Does the circuit matter?"):
        gr.Markdown(CIRCUIT)
        gr.Dataframe(value=conditions(), wrap=True)
        gr.Markdown(contrasts())
    with gr.Tab("The 393 earlier runs"):
        gr.Markdown(
            "Every steering, LoRA and MEMIT cell of the pre-registered study (b1 = correcting "
            "false facts, b2 = strengthening refusal), with its Nothing-Else-Broke scorecard."
        )
        beh = gr.Dropdown(["b1", "b2"], value="b1", label="behaviour")
        cell = gr.Dropdown(choices=cells("b1"), label="cell")
        fam = gr.Dropdown(
            [
                "facts_edited_en",
                "facts_neighbourhood_en",
                "refusal_harmful_en",
                "refusal_benign_en",
                "facts_edited_hi",
                "refusal_harmful_sw",
            ],
            value="facts_edited_en",
            label="prompt family",
        )
        beh.change(lambda b: gr.Dropdown(choices=cells(b)), beh, cell)
        out = gr.Dataframe(label="base vs intervened", wrap=True)
        gr.Button("show").click(compare, [beh, cell, fam], out)
        sc = gr.Dataframe(value=scorecard("b1"), label="scorecard, all conditions")
        beh.change(scorecard, beh, sc)

if __name__ == "__main__":
    demo.launch()
