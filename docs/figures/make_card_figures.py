"""The figures on the three model cards, from the generated results.

  uv run --with matplotlib python docs/figures/make_card_figures.py

Writes docs/figures/cards/*.png:
  a22_safety, a24_safety   harmful requests answered by language and harmless prompts refused:
                           base model, SafeTune steering and the card's adapter
  adapters_defects         the two defects the audit found in the first adapter, both adapters
  adapters_side_effects    change against the base model on five ability checks, 95 % intervals
  cells_contents           the runs in the circuit-steering-cells repo, weights published or not
  cells_locality           fact correction in the main study: effect and side effects by condition
Reads docs/results (a2*_b2.json and .csv, b1_cells.csv), runs/a2*/judge (labels) and
runs/**/interventions (cell.json files). One colour per method on every figure: grey base model,
teal SafeTune steering, red first adapter, blue second adapter.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "docs/figures/cards"
LANGS = ("en", "hi", "ar", "fr", "sw")
NAME = {"en": "English", "hi": "Hindi", "ar": "Arabic", "fr": "French", "sw": "Swahili"}
RELEASED = "b2_lora_circuit30_s0"
ADAPTER = {"a22": "First adapter", "a24": "Second adapter"}
BLUE, RED, TEAL, GREY = "#0F4D92", "#B64342", "#42949E", "#CFCECE"
PINK, DARK, INK, MUTED = "#E9A6A1", "#767676", "#272727", "#5F5F5F"
COLOR = {"a22": RED, "a24": BLUE}
# "0.877 → 0.873 (-0.003 [-0.020, +0.010])": base, new, change, 95 % interval
CHANGE = re.compile(r"([\d.]+) → ([\d.]+) \(([-+][\d.]+) \[([-+][\d.]+), ([-+][\d.]+)\]\)")
CHECKS = {
    "facts_neighbourhood": "Nearby facts",
    "facts_paraphrase": "Reworded facts",
    "mmlu": "General knowledge\n(Global-MMLU)",
    "mgsm": "Maths (MGSM)",
    "ifeval": "Following instructions\n(IFEval)",
}


def style() -> None:
    """Sized for a Hub model card, whose text column is about 600 px wide: figures are 8 to 9 in
    wide, so 13 pt text is about 12 px on the page."""
    plt.rcParams.update(
        {
            "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.6,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "legend.frameon": False,
            "savefig.dpi": 220,
            "savefig.facecolor": "white",
        }
    )


def heading(ax, title: str, note: str, legend: bool = True) -> None:
    """A bold title with a grey one-line note under it, left-aligned above the axes. Offsets are in
    points, so the spacing is the same on every figure; `legend` leaves room for one legend row."""
    up = 25 if legend else 0
    for text, dy, kw in (
        (title, 32, {"fontsize": 14.5, "fontweight": "bold", "color": INK}),
        (note, 12, {"fontsize": 12, "color": MUTED}),
    ):
        ax.annotate(
            text, (0, 1), (0, up + dy), xycoords="axes fraction", textcoords="offset points", **kw
        )


def legend_row(ax, ncols: int) -> None:
    ax.legend(
        ncols=ncols,
        loc="lower left",
        bbox_to_anchor=(0, 1),
        borderaxespad=0.2,
        fontsize=12,
        handlelength=1.1,
        handletextpad=0.5,
        columnspacing=1.5,
    )


def bars(ax, xs, ys, color, label=None, fmt="{:.1f}", width=0.26, size=10.5):
    ax.bar(xs, ys, width, color=color, edgecolor=INK, linewidth=1.1, label=label, zorder=2)
    for x, y in zip(xs, ys, strict=True):
        ax.annotate(
            fmt.format(y), (x, y), (0, 2.5), textcoords="offset points", ha="center", fontsize=size
        )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(OUT / f"{name}.png")


def comply(run: str, name: str) -> list[float]:
    j = json.loads((ROOT / f"runs/{run}/judge/{name}.json").read_text())["harmful_test"]
    return [100 * sum(v == "comply" for v in j[k].values()) / len(j[k]) for k in LANGS]


def summary(run: str) -> dict:
    return json.loads((ROOT / f"docs/results/{run}_b2.json").read_text())


def cells(run: str) -> dict[str, dict]:
    rows = csv.DictReader((ROOT / f"docs/results/{run}_b2.csv").open())
    return {r["cell"]: r for r in rows}


def safety(run: str) -> None:
    steer = "b2_steer_safetune_rd_s0"
    series = [
        ("Base model", comply("a22", "base"), GREY),
        ("SafeTune steering", comply("a22", steer), TEAL),
        (f"{ADAPTER[run]} (this model)", comply(run, RELEASED), COLOR[run]),
    ]
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(8.6, 6.2), gridspec_kw={"height_ratios": [3.1, 1.15], "hspace": 0.66}
    )
    for i, (label, ys, color) in enumerate(series):
        bars(ax, [k + (i - 1) * 0.27 for k in range(len(LANGS))], ys, color, label)
    ax.set_xticks(range(len(LANGS)), [NAME[k] for k in LANGS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Share answered (%)")
    ax.set_ylim(0, 18)
    ax.set_yticks(range(0, 19, 5))
    heading(ax, "Harmful requests answered", "159 held-out requests per language. Lower is better.")
    legend_row(ax, 3)

    benign = [
        100 * float(summary("a22")["base"]["benign_refusal"]),
        100 * float(cells("a22")[steer]["benign_refusal"]),
        100 * float(cells(run)[RELEASED]["benign_refusal"]),
    ]
    colors = [color for _, _, color in series]
    bx.barh(range(3), benign, 0.62, color=colors, edgecolor=INK, linewidth=1.1, zorder=2)
    for y, v in enumerate(benign):
        bx.annotate(
            f"{v:.1f}", (v, y), (5, 0), textcoords="offset points", va="center", fontsize=11
        )
    bx.set_yticks(range(3), [label.split(" (")[0] for label, _, _ in series])
    bx.tick_params(axis="y", length=0)
    bx.invert_yaxis()
    bx.set_xlim(0, 18)
    bx.set_xticks(range(0, 19, 5))
    bx.set_xlabel("Share refused (%)")
    heading(bx, "Harmless prompts refused", "Five languages. Lower is better.", legend=False)
    save(fig, f"{run}_safety")


def defects() -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 4.2))
    for i, run in enumerate(("a22", "a24")):
        xs = [k + (i - 0.5) * 0.38 for k in range(len(LANGS))]
        ended = summary(run)["ended"][RELEASED]
        fluency = summary(run)["fluency"][RELEASED]
        bars(ax, xs, [100 * ended[k] for k in LANGS], COLOR[run], ADAPTER[run], "{:.0f}", 0.36, 10)
        rise = [100 * (fluency[k] - 1) for k in LANGS]
        bars(bx, xs, rise, COLOR[run], fmt="+{:.0f}", width=0.36, size=10)
    for a in (ax, bx):
        a.set_xticks(range(len(LANGS)), [NAME[k] for k in LANGS], fontsize=11.5)
        a.tick_params(axis="x", length=0)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Share of replies to harmful requests (%)", fontsize=12)
    heading(ax, "Replies that stop at the refusal", "Higher is better.")
    legend_row(ax, 2)
    bx.set_ylim(0, 54)
    bx.set_yticks(range(0, 51, 10))
    bx.set_ylabel("Increase over the base model (%)", fontsize=12)
    heading(bx, "Perplexity on unseen text", "Fluency loss. Lower is better.")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "adapters_defects")


def side_effects() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.axvline(0, color=INK, linewidth=1.3, zorder=1)
    for i, run in enumerate(("a22", "a24")):
        audit = summary(run)["audit"][RELEASED]
        for row, key in enumerate(CHECKS):
            _, _, d, lo, hi = (100 * float(x) for x in CHANGE.fullmatch(audit[key]).groups())
            y = row + (i - 0.5) * 0.3
            ax.plot([lo, hi], [y, y], color=COLOR[run], linewidth=2.4, solid_capstyle="round")
            ax.plot(
                d,
                y,
                "o",
                color=COLOR[run],
                markersize=8.5,
                markeredgecolor="white",
                markeredgewidth=1.3,
                label=ADAPTER[run] if row == 0 else None,
            )
    ax.set_yticks(range(len(CHECKS)), list(CHECKS.values()))
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(-20, 20)
    ax.set_xticks(range(-20, 21, 5))
    ax.set_xlabel("Change against the base model (percentage points)")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    heading(ax, "Did anything else change?", "Dot: change. Line: 95 % interval. Left is worse.")
    legend_row(ax, 2)
    save(fig, "adapters_side_effects")


def contents() -> None:
    """Runs per folder of the cells repo, split into weights published and withheld. The withheld
    list comes from the same functions that publish the repo (data/push_hf.py)."""
    sys.path.insert(0, str(ROOT))
    from data.push_hf import run_withheld, withheld_cells

    kind = {"steer": "steering", "lora": "LoRA", "edit": "MEMIT editing"}
    main = Counter(p.parts[-4:-2] for p in (ROOT / "runs/interventions").glob("*/*/*/cell.json"))
    held = Counter(tuple(p.split("/")[:2]) for p in withheld_cells(ROOT / "runs"))
    rows = [
        (f"{'Fact correction' if b == 'b1' else 'Refusal'}: {kind[m]}", main[b, m], held[b, m])
        for b in ("b1", "b2")
        for m in ("steer", "lora", "edit")
    ]
    for run in ("a22", "a24"):
        n = len(list((ROOT / f"runs/{run}/interventions").glob("*/*/*/cell.json")))
        rows.append((f"{ADAPTER[run]}: training runs", n, len(run_withheld(ROOT / f"runs/{run}"))))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ys = range(len(rows))
    published = [n - h for _, n, h in rows]
    ax.barh(
        ys, published, 0.62, color=BLUE, edgecolor=INK, linewidth=1.1, label="Weights published"
    )
    ax.barh(
        ys,
        [h for _, _, h in rows],
        0.62,
        left=published,
        color="white",
        edgecolor=INK,
        linewidth=1.1,
        hatch="///",
        label="Weights withheld",
    )
    for y, (_, n, h) in zip(ys, rows, strict=True):
        note = f"{n} runs" if not h else f"{n} runs, {n - h} with weights"
        ax.annotate(note, (n, y), (6, 0), textcoords="offset points", va="center", fontsize=11.5)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.axhline(2.5, color="#BDBDBD", linewidth=1)
    ax.axhline(5.5, color="#BDBDBD", linewidth=1)
    ax.set_xlim(0, 228)
    ax.set_xticks(range(0, 151, 50))
    ax.set_xlabel("Number of runs")
    total = sum(n for _, n, _ in rows)
    heading(ax, f"{total} runs in this repo", "Weights are published only if refusal did not drop.")
    legend_row(ax, 2)
    save(fig, "cells_contents")


def locality() -> None:
    rows = list(csv.DictReader((ROOT / "docs/results/b1_cells.csv").open()))
    conds = [
        ("circuit", "Circuit", BLUE),
        ("random", "Random", GREY),
        ("inverse", "Lowest-scoring", PINK),
        ("unrestricted", "All components", DARK),
    ]
    kinds = [("steer", "Steering"), ("lora", "LoRA"), ("edit", "MEMIT")]
    panels = [
        ("efficacy", "False facts corrected", "Higher is better.", "Accuracy gain, edited facts"),
        ("l3_mean_abs_delta", "Side effects, other languages", "Lower is better.", "Mean change"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, (key, title, note, ylabel) in zip(axes, panels, strict=True):
        for i, (cond, label, color) in enumerate(conds):
            for k, (kind, _) in enumerate(kinds):
                vals = [
                    100 * float(r[key])
                    for r in rows
                    if r["intervention"] == kind and r["condition"] == cond
                ]
                x = k + (i - 1.5) * 0.2
                ax.bar(
                    x,
                    sum(vals) / len(vals),
                    0.19,
                    color=color,
                    edgecolor=INK,
                    linewidth=1.1,
                    label=label if k == 0 else None,
                    zorder=2,
                )
                spread = [x + (j - (len(vals) - 1) / 2) * 0.035 for j in range(len(vals))]
                ax.plot(
                    spread,
                    vals,
                    "o",
                    color="white",
                    markeredgecolor=INK,
                    markersize=4,
                    markeredgewidth=0.9,
                    zorder=3,
                )
        ax.set_xticks(range(len(kinds)), [name for _, name in kinds])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(f"{ylabel} (points)", fontsize=12)
        ax.set_ylim(0, None)
        heading(ax, title, note)
    legend_row(axes[0], 4)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "cells_locality")


def main() -> None:
    style()
    for run in ("a22", "a24"):
        safety(run)
    defects()
    side_effects()
    contents()
    locality()


if __name__ == "__main__":
    main()
