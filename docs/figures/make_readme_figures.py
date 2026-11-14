"""The figures in the READMEs of the three repos, from the generated results.

  uv run --with matplotlib python docs/figures/make_readme_figures.py

Writes
  docs/figures/readme/results.png   harmful requests answered by language and harmless prompts
                                    refused: base model, SafeTune steering and both adapters
  docs/figures/readme/control.png   the refusal circuit as a map of the model, and the adapter
                                    trained on the circuit against the same adapter elsewhere
  ../tiny-aya-circuits/docs/figures/circuits.png     the fact circuit as a map, and how much of
                                    each behaviour a circuit restores against random nodes
  ../nothing-else-broke/docs/figures/standard_check.png   runs that pass the usual side-effect
                                    check and are damaged anyway, and what moved in them
Reads docs/results, runs/a2*/judge, the released adapter's cell.json and
runs/circuits/component_scores.csv. Style and colours are those of make_card_figures.py.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_card_figures import (
    BLUE,
    GREY,
    INK,
    LANGS,
    MUTED,
    NAME,
    RED,
    RELEASED,
    ROOT,
    TEAL,
    bars,
    cells,
    comply,
    heading,
    legend_row,
    style,
    summary,
)

HERE = ROOT / "docs/figures/readme"
SIBLINGS = ROOT.parent
EMPTY = "#E9E9E9"
N_LAYERS, N_HEADS = 36, 16
CONTRAST = re.compile(r"Δ=([-+][\d.]+) .*paired t p=([\d.]+)")
STEER = "b2_steer_safetune_rd_s0"
TASKS = {
    "b1_en": "Facts, English",
    "b1_hi": "Facts, Hindi",
    "b1_ar": "Facts, Arabic",
    "b1_fr": "Facts, French",
    "b1_sw": "Facts, Swahili",
    "langid_en": "Naming the language",
    "ioi_fixed": "Indirect object (IOI)",
    "b2_en": "Refusal",
    "b2m_en": "Refusal, same length",
}
MOVED = {
    "l5_top1_error_delta": "Facts nobody edited,\nnow wrong",
    "l1_neigh_en_delta": "Nearby facts now wrong",
    "l3_mean_abs_delta": "Scores in other languages",
    "l2_benign_refusal_delta_en": "Harmless prompts refused more",
    "l2_harmful_refusal_delta_en": "Harmful prompts refused less",
}


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(path)


def results() -> None:
    series = [
        ("Base model", comply("a22", "base"), GREY, summary("a22")["base"]),
        ("SafeTune steering", comply("a22", STEER), TEAL, cells("a22")[STEER]),
        ("First adapter", comply("a22", RELEASED), RED, cells("a22")[RELEASED]),
        ("Second adapter", comply("a24", RELEASED), BLUE, cells("a24")[RELEASED]),
    ]
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(9.6, 6.6), gridspec_kw={"height_ratios": [3, 1.45], "hspace": 0.62}
    )
    for i, (label, ys, color, _) in enumerate(series):
        xs = [k + (i - 1.5) * 0.21 for k in range(len(LANGS))]
        bars(ax, xs, ys, color, label, width=0.2, size=9.5)
    ax.set_xticks(range(len(LANGS)), [NAME[k] for k in LANGS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Share answered (%)")
    ax.set_ylim(0, 18)
    ax.set_yticks(range(0, 19, 5))
    heading(ax, "Harmful requests answered", "159 held-out requests per language. Lower is better.")
    legend_row(ax, 4)

    benign = [100 * float(row["benign_refusal"]) for _, _, _, row in series]
    colors = [color for _, _, color, _ in series]
    bx.barh(range(4), benign, 0.62, color=colors, edgecolor=INK, linewidth=1.1, zorder=2)
    for y, v in enumerate(benign):
        bx.annotate(
            f"{v:.1f}", (v, y), (5, 0), textcoords="offset points", va="center", fontsize=11
        )
    bx.set_yticks(range(4), [label for label, _, _, _ in series])
    bx.tick_params(axis="y", length=0)
    bx.invert_yaxis()
    bx.set_xlim(0, 18)
    bx.set_xticks(range(0, 19, 5))
    bx.set_xlabel("Share refused (%)")
    heading(
        bx,
        "Harmless prompts refused",
        "Five languages, counted by refusal phrases. Lower is better.",
        legend=False,
    )
    save(fig, HERE / "results.png")


def circuit_map(ax, nodes: list[str], color: str) -> tuple[int, int]:
    """One square per attention head (layer across, head down) and a row of MLPs under them;
    the components in `nodes` ("A20.4", "MLP 20") are filled. Returns (heads, MLPs) filled."""
    members = set(nodes)
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            on = f"A{layer}.{head}" in members
            ax.add_patch(Rectangle((layer, head), 0.84, 0.84, color=color if on else EMPTY, lw=0))
        on = f"MLP {layer}" in members
        ax.add_patch(
            Rectangle((layer, N_HEADS + 0.9), 0.84, 1.5, color=color if on else EMPTY, lw=0)
        )
    ax.set_xlim(-0.2, N_LAYERS)
    ax.set_ylim(N_HEADS + 2.6, -0.2)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xticks([k + 0.42 for k in range(0, N_LAYERS, 5)], range(0, N_LAYERS, 5))
    ax.set_yticks([N_HEADS / 2, N_HEADS + 1.65], ["16 attention\nheads", "MLP"])
    ax.tick_params(length=0)
    ax.set_xlabel("Layer")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return sum(n.startswith("A") for n in members), sum(n.startswith("MLP") for n in members)


def control() -> None:
    cell = ROOT / "runs/a22/interventions/b2/lora/circuit30_s0/cell.json"
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.58}
    )
    heads, mlps = circuit_map(ax, json.loads(cell.read_text())["nodes"], BLUE)
    heading(
        ax,
        "The refusal circuit",
        f"Top 30 %: {heads} of {N_LAYERS * N_HEADS} attention heads, "
        f"{mlps} of {N_LAYERS} MLPs (blue).",
        legend=False,
    )

    groups = [
        ("a24", "Second adapter", BLUE, ["circuit30", "random30", "unrestricted"]),
        ("a22", "First adapter", RED, ["circuit30", "random30", "inverse30", "unrestricted"]),
    ]
    label = {
        "circuit30": "The circuit",
        "random30": "Random components",
        "inverse30": "Lowest-scoring",
        "unrestricted": "All components",
    }
    y, ticks, names, spans = 0.0, [], [], []
    for run, title, color, conds in groups:
        rows = list(cells(run).values())
        bx.annotate(
            title,
            (0, y - 0.25),
            (4, 0),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=color,
            va="center",
            annotation_clip=False,
        )
        y += 0.75
        top = y
        for cond in conds:
            vals = sorted(100 * float(r["comply_xl"]) for r in rows if r["condition"] == cond)
            circuit = cond == "circuit30"
            spread = 0.17 if len(vals) > 3 else 0  # nine random runs: three lanes, not one smear
            bx.plot(
                vals,
                [y + ((i % 3) - 1) * spread for i in range(len(vals))],
                "o",
                markersize=8,
                markeredgewidth=1.6,
                markeredgecolor=color,
                color=color if circuit else "white",
                zorder=3,
            )
            mean = sum(vals) / len(vals)
            bx.plot([mean, mean], [y - 0.3, y + 0.3], color=INK, linewidth=2.2, zorder=4)
            bx.annotate(
                f"{mean:.1f}",
                (10, y),
                ha="right",
                va="center",
                fontsize=11.5,
                fontweight="bold" if circuit else "normal",
            )
            ticks.append(y)
            names.append(label[cond])
            y += 0.8
        key = "circuit30 - random30: harmful comply, hi/ar/fr/sw (judge)"
        delta, p = CONTRAST.search(summary(run)["contrasts"][key]).groups()
        bx.annotate(
            f"Circuit: {100 * float(delta):.1f} points more than random (p = {float(p):.2f})",
            (0, y - 0.15),
            (4, 0),
            textcoords="offset points",
            fontsize=11.5,
            color=MUTED,
            va="center",
            annotation_clip=False,
        )
        spans.append((top - 0.4, y - 0.4))
        y += 1.2
    base = 100 * float(summary("a22")["base"]["comply_xl"])
    for lo, hi in spans:
        bx.plot([base, base], [lo, hi], color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    bx.annotate(
        f"Base\nmodel\n{base:.1f}",
        (base, sum(spans[0]) / 2),
        (6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=11.5,
        color=MUTED,
    )
    bx.set_yticks(ticks, names, fontsize=11.5)
    bx.tick_params(axis="y", length=0)
    bx.set_ylim(y - 0.8, -0.75)
    bx.set_xlim(0, 15.4)
    bx.set_xticks(range(0, 13, 4))
    bx.set_xlabel("Harmful requests answered, outside English (%)", fontsize=12)
    bx.spines["left"].set_visible(False)
    heading(
        bx,
        "Same adapter, different components",
        "Dot: one training run. Bar and number: the mean.",
        legend=False,
    )
    save(fig, HERE / "control.png")


def faithfulness() -> dict[str, list[tuple[float, float]]]:
    """Per task, (circuit, random nodes) patching ratio of every seed, from the first table of
    docs/results/circuits.md."""
    lines = (ROOT / "docs/results/circuits.md").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| task | seed | nodes"))
    head = [c.strip() for c in lines[start].strip("|").split("|")]
    out: dict[str, list[tuple[float, float]]] = {}
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        row = dict(zip(head, (c.strip() for c in line.strip("|").split("|")), strict=True))
        pair = (100 * float(row["patching"]), 100 * float(row["random_node_patching"]))
        out.setdefault(row["task"], []).append(pair)
    return out


def circuits() -> None:
    rows = csv.DictReader((ROOT / "runs/circuits/component_scores.csv").open())
    top = [r for r in rows if r["task"] == "b1_en" and r["seed"] == "0" and int(r["rank"]) <= 61]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(9.8, 4.3), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.68}
    )
    heads, mlps = circuit_map(ax, [r["component"] for r in top], BLUE)
    heading(
        ax,
        "The English fact-recall circuit (one run)",
        f"Blue: the top 10 % of components ({heads} heads, {mlps} MLPs).",
    )

    data = faithfulness()
    for y, task in enumerate(TASKS):
        circuit = [c for c, _ in data[task]]
        random = [r for _, r in data[task]]
        mean = sum(circuit) / len(circuit)
        bx.barh(y, mean, 0.62, color=BLUE, edgecolor=INK, linewidth=1.1, zorder=2)
        bx.plot(
            circuit,
            [y + dy for dy in (-0.14, 0, 0.14)],
            "o",
            color="white",
            markeredgecolor=INK,
            markersize=4.5,
            markeredgewidth=0.9,
            zorder=3,
        )
        bx.plot(
            sum(random) / len(random),
            y,
            "D",
            color="white",
            markeredgecolor=RED,
            markersize=6.5,
            markeredgewidth=1.8,
            zorder=4,
            label="Random components, same number" if y == 0 else None,
        )
        bx.annotate(
            f"{mean:.0f}".replace("-0", "0").replace("-", "−"),
            (116, y),
            ha="right",
            va="center",
            fontsize=11.5,
        )
    bx.set_yticks(range(len(TASKS)), list(TASKS.values()), fontsize=11.5)
    bx.tick_params(axis="y", length=0)
    bx.invert_yaxis()
    bx.set_xlim(-6, 118)
    bx.set_xticks(range(0, 101, 25))
    bx.set_xlabel("Share of the behaviour restored (%)", fontsize=12)
    bx.axvline(0, color=INK, linewidth=1.3, zorder=1)
    bx.spines["left"].set_visible(False)
    heading(bx, "Is the top 10 % enough on its own?", "Dot: one run. Bar and number: the mean.")
    legend_row(bx, 1)
    save(fig, SIBLINGS / "tiny-aya-circuits/docs/figures/circuits.png")


def standard_check() -> None:
    text = (ROOT / "docs/results/standard_checks.md").read_text()
    table = {}
    for name in ("real interventions", "norm-matched random noise"):
        row = re.search(rf"\| {name} \| (\d+) \| (\d+) \(\d+%\) \| \d+ \(\d+%\) \| (\d+) ", text)
        table[name] = tuple(int(x) for x in row.groups())
    moved = {k: int(n) for k, n in re.findall(r"- `(\w+)`: (\d+) cells", text)}

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(9.8, 3.6), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 1.0}
    )
    kinds = [
        ("real interventions", "Real changes\nto the model", RED),
        ("norm-matched random noise", "Random noise\nof the same size", GREY),
    ]
    for y, (key, _, color) in enumerate(kinds):
        _, passed, damaged = table[key]
        share = 100 * damaged / passed
        ax.barh(y, 100, 0.56, color="white", edgecolor=INK, linewidth=1.1, zorder=2)
        ax.barh(y, share, 0.56, color=color, edgecolor=INK, linewidth=1.1, zorder=3)
        ax.annotate(
            f"{damaged} of {passed} ({share:.0f} %)",
            (share, y),
            (8, 0),
            textcoords="offset points",
            va="center",
            fontsize=11.5,
            zorder=4,
        )
    ax.set_yticks(range(2), [name for _, name, _ in kinds])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_ylim(1.55, -0.55)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xlabel("Share of the changes that pass the usual check (%)", fontsize=12)
    ax.spines["left"].set_visible(False)
    heading(
        ax,
        "Passed the usual check, damaged anyway",
        "Damaged: a score we audit moved 10 points or more.",
        legend=False,
    )

    order = sorted(MOVED, key=lambda k: -moved[k])
    bx.barh(
        range(len(order)),
        [moved[k] for k in order],
        0.62,
        color=RED,
        edgecolor=INK,
        linewidth=1.1,
        zorder=2,
    )
    for y, k in enumerate(order):
        bx.annotate(
            str(moved[k]),
            (moved[k], y),
            (5, 0),
            textcoords="offset points",
            va="center",
            fontsize=11,
        )
    bx.set_yticks(range(len(order)), [MOVED[k] for k in order], fontsize=11.5)
    bx.tick_params(axis="y", length=0)
    bx.invert_yaxis()
    bx.set_xlim(0, 25)
    bx.set_xticks(range(0, 26, 5))
    bx.set_xlabel("Changes", fontsize=12)
    heading(bx, "What moved in those 31", "A change can count in several rows.", legend=False)
    save(fig, SIBLINGS / "nothing-else-broke/docs/figures/standard_check.png")


def main() -> None:
    style()
    results()
    control()
    circuits()
    standard_check()


if __name__ == "__main__":
    main()
