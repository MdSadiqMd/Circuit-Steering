"""Build the static Space (space/static/index.html) from the same stored results as app.py.

A Gradio Space on free hardware needs a paid plan; a static Space does not. The page embeds the
results as JSON in space/static_template.html and draws its charts in the browser, so it needs no
server and no model. Every number on the page comes from the files read here; the template holds
only wording.

  CS_AUDIT_DIR=<local copy of the circuit-steering-audit dataset> \
      uv run --with gradio python space/build_static.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import app  # noqa: E402

PAGE = (HERE / "static_template.html").read_text()
RUNS = ("a22", "a24")  # first and second adapter
CELL = "a22/b2/lora/circuit30_s0/cell.json"  # path in the circuit-steering-cells repo
AUDIT_ROWS = {
    "facts_neighbourhood": "Nearby facts",
    "facts_paraphrase": "Reworded facts",
    "mmlu": "General knowledge (MMLU)",
    "mgsm": "Maths (MGSM)",
    "ifeval": "Following instructions (IFEval)",
}
CONDITIONS = {
    "circuit30": "Circuit, top 30%",
    "random30": "Random 30%",
    "inverse30": "Lowest-scoring 30%",
    "unrestricted": "All components",
    "circuit10": "Circuit, top 10%",
    "random10": "Random 10%",
}
RATES = ("comply_xl", "comply_en", "benign_refusal")
SCORECARD = {  # column of <behaviour>_cells.csv: heading on the page
    "cell": "Run",
    "efficacy": "Effect on target",
    "l3_mean_abs_delta": "Change across languages",
    "l1_neigh_en_delta": "Nearby facts",
    "l2_benign_refusal_delta_en": "Harmless refusals",
    "l1_wikitext_ppl_ratio": "Perplexity ratio",
    "l5_top1_error_delta": "Errors on unedited facts",
    "delta_norm": "Size of change",
}
# "0.877 → 0.873 (-0.003 [-0.020, +0.010])": base, new, change, 95 % interval
# "Δ=+0.0073 [...] ...; per-seed Δ ..., paired t p=0.176": difference and its p-value over seeds
CONTRAST = re.compile(r"Δ=([-+][\d.]+) .*paired t p=([\d.]+)")
CHANGE = re.compile(r"([\d.]+) → ([\d.]+) \(([-+][\d.]+) \[([-+][\d.]+), ([-+][\d.]+)\]\)")

README = """---
title: Safer Tiny Aya in five languages, audited
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Two refusal adapters for Tiny Aya and their audit
---
Two refusal adapters for Tiny Aya Global, before and after on held-out test prompts, with the audit
that checked them and the 393 earlier intervention runs. The page shows stored results; no model
runs here. Built by `space/build_static.py` in https://github.com/MdSadiqMd/Circuit-Steering.
"""


def pc(x: float) -> float:
    return round(100 * float(x), 1)


def cells(run: str) -> list[dict]:
    return list(csv.DictReader(next(app.ROOT.glob(f"**/{run}_b2.csv")).open()))


def label_share(judge: dict, lang: str, label: str) -> float:
    by = judge["harmful_test"][lang]
    return pc(sum(v == label for v in by.values()) / len(by))


def phrase_refusal(ev: dict, lang: str) -> float:
    scores = ev["harmful_test"][lang]["scores"]
    return pc(sum(scores.values()) / len(scores))


def circuit() -> dict:
    """The released adapter's component set as heads[layer] = head indices and the MLP layers."""
    local = HERE.parent / "runs" / CELL.replace("a22/", "a22/interventions/", 1)
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local = Path(hf_hub_download(f"{app.NS}/circuit-steering-cells", CELL))
    nodes = json.loads(local.read_text())["nodes"]
    heads: list[list[int]] = [[] for _ in range(36)]
    for n in nodes:
        if n.startswith("A"):  # "A20.4" is layer 20 head 4, "MLP 20" the layer's MLP
            layer, head = n[1:].split(".")
            heads[int(layer)].append(int(head))
    return {
        "heads": [sorted(h) for h in heads],
        "mlps": sorted(int(n.split()[1]) for n in nodes if n.startswith("MLP")),
    }


def standard_checks() -> dict:
    """{kind: [passed the standard check, of those damaged at 0.10]} from standard_checks.md."""
    text = next(app.ROOT.glob("**/standard_checks.md")).read_text()
    out = {}
    for key, name in (("real", "real interventions"), ("noise", "norm-matched random noise")):
        cols = re.search(rf"\| {name} \|(.*)", text).group(1).split("|")
        out[key] = [int(cols[1].split()[0]), int(cols[3].split()[0])]
    return out


def scorecard(beh: str) -> list[list[str]]:
    """Every earlier run of one behaviour with its audit numbers, to three decimals."""
    rows = csv.DictReader(next(app.ROOT.glob(f"**/{beh}_cells.csv")).open())
    return [list(SCORECARD.values())] + [
        [r[k] if k == "cell" or not r[k] else f"{float(r[k]):.3f}" for k in SCORECARD] for r in rows
    ]


def circuit_vs_random(run: str) -> list[float]:
    """[difference in points, p-value] of harmful requests answered outside English, circuit minus
    random components at 30 %, from the run's pre-computed contrast."""
    contrasts = app.SUMMARY[run]["contrasts"]
    key = next(k for k in contrasts if k.startswith("circuit30 - random30: harmful comply"))
    delta, p = CONTRAST.search(contrasts[key]).groups()
    return [pc(delta), float(p)]


def runs_table() -> list[list]:
    """Mean over seeds of every adapter condition, in percent and perplexity ratios."""
    head = ["Run", "Seeds", "Harmful answered, outside English", "Harmful answered, English"]
    head += ["Harmless refused", "Perplexity ratio, English", "Perplexity ratio, 5 languages"]
    base = app.SUMMARY["a22"]["base"]
    out = [head, ["Base model", "", *(pc(base[k]) for k in RATES), "1.00", "1.00"]]
    for run, name in zip(RUNS, ("First adapter", "Second adapter"), strict=True):
        for cond, m in app.SUMMARY[run]["conditions"].items():
            what = CONDITIONS.get(cond)
            out.append(
                [
                    f"{name} · {what}" if what else "SafeTune steering",
                    sum(r["condition"] == cond for r in cells(run)),
                    *(pc(m[k]) for k in RATES),
                    f"{m['wikitext_ppl_ratio']:.2f}",
                    f"{m['fluency_mean']:.2f}" if "fluency_mean" in m else "",
                ]
            )
    return out


def main() -> None:
    rows = {run: {r["cell"]: r for r in cells(run)} for run in RUNS}
    released = {run: rows[run][app.RELEASED] for run in RUNS}
    steer = rows["a22"]["b2_steer_safetune_rd_s0"]
    base = app.SUMMARY["a22"]["base"]
    judge_base = app.JUDGE["a22"][0]
    assert judge_base["harmful_test"] == app.JUDGE["a24"][0]["harmful_test"], "base differs"

    def method(name: str, key: str, m: dict) -> dict:
        return {
            "name": name,
            "key": key,
            "comply_xl": pc(m["comply_xl"]),
            "comply_en": pc(m["comply_en"]),
            "benign": pc(m["benign_refusal"]),
            "ppl": round(float(m.get("wikitext_ppl_ratio", 1)), 2),
        }

    audit = []
    for key, name in AUDIT_ROWS.items():
        row = {"name": name}
        for run in RUNS:
            m = CHANGE.fullmatch(app.SUMMARY[run]["audit"][app.RELEASED][key])
            row[run] = [pc(x) for x in m.groups()]  # base, new, change, low, high
        audit.append(row)

    conditions = {}
    for run in RUNS:
        by: dict[str, list] = {}
        for r in rows[run].values():
            if r["condition"] in CONDITIONS:
                by.setdefault(r["condition"], []).append(
                    [int(r["seed"]), pc(r["comply_xl"]), pc(r["benign_refusal"])]
                )
        conditions[run] = [
            {"name": CONDITIONS[c], "circuit": c.startswith("circuit"), "seeds": sorted(by[c])}
            for c in CONDITIONS
            if c in by
        ]

    data = {
        "n": len(judge_base["harmful_test"]["en"]),
        "langs": app.LANGS,
        "names": app.NAME,
        "methods": [
            method("Base model", "base", base),
            method("SafeTune steering", "steer", steer),
            method("First adapter", "a22", released["a22"]),
            method("Second adapter", "a24", released["a24"]),
        ],
        "by_lang": {
            lang: {
                "base": label_share(judge_base, lang, "comply"),
                **{run: label_share(app.JUDGE[run][1], lang, "comply") for run in RUNS},
            }
            for lang in app.LANGS
        },
        "ended": {
            run: {k: pc(v) for k, v in app.SUMMARY[run]["ended"][app.RELEASED].items()}
            for run in RUNS
        },
        "ppl_rise": {
            run: {
                lang: round(100 * (app.SUMMARY[run]["fluency"][app.RELEASED][lang] - 1), 1)
                for lang in app.LANGS
            }
            for run in RUNS
        },
        "audit": audit,
        "conditions": conditions,
        "circuit": circuit(),
        "trained": [
            int(float(released["a22"]["n_trained"])),
            int(float(rows["a22"]["b2_lora_unrestricted_s0"]["n_trained"])),
        ],
        "versus_random": {run: circuit_vs_random(run) for run in RUNS},
        "standard": standard_checks(),
        "gap": {
            "phrase": [phrase_refusal(app.EVAL["a22"][0], lang) for lang in ("en", "hi")],
            "judge": [label_share(judge_base, lang, "refuse") for lang in ("en", "hi")],
            "comply": [label_share(judge_base, lang, "comply") for lang in ("en", "hi")],
        },
        "benign": {
            run: {lang: app.benign(label, lang, False)[1:] for lang in app.LANGS}
            for label, run in app.RUNS.items()
        },
        "runs_table": runs_table(),
        "scorecard": {b: scorecard(b) for b in ("b1", "b2")},
    }
    assert data["n"] and len(data["scorecard"]["b1"]) > 1, "no data found"
    out = HERE / "static"
    out.mkdir(exist_ok=True)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (out / "index.html").write_text(PAGE.replace("__DATA__", blob))
    (out / "README.md").write_text(README)
    print(out / "index.html", f"{len(blob) / 1e6:.2f} MB of data")


if __name__ == "__main__":
    main()
