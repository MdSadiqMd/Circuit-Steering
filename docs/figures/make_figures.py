"""The write-up's figure, from the generated results.

  uv run --with matplotlib python docs/figures/make_figures.py

Left: harmful compliance on the held-out test prompts by language (LLM judge), base model against
the SafeTune steering baseline and the two released adapters. Right: every A22 and A24 cell as
harmful compliance (hi/ar/fr/sw) against the perplexity ratio on unseen sentences in five
languages. Reads runs/a2*/judge (labels) and docs/results/a2*_b2.csv; writes
docs/figures/adapters.png and .pdf.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
NAMES = {"a22": "first", "a24": "second"}  # the names the write-up uses for A22 and A24
CONDS = {
    "circuit30": "circuit 30 %",
    "random30": "random 30 %",
    "inverse30": "lowest-scoring 30 %",
    "unrestricted": "all components",
}
LANGS = ("en", "hi", "ar", "fr", "sw")
NAME = {"en": "English", "hi": "Hindi", "ar": "Arabic", "fr": "French", "sw": "Swahili"}
# one colour per method, the same on the model cards (make_card_figures.py) and the Space
BLUE, TEAL, RED, GREY = "#0F4D92", "#42949E", "#B64342", "#9B9B9B"
RELEASED = "b2_lora_circuit30_s0"


def comply(run: str, name: str) -> list[float]:
    j = json.loads((ROOT / f"runs/{run}/judge/{name}.json").read_text())["harmful_test"]
    return [100 * sum(v == "comply" for v in j[k].values()) / len(j[k]) for k in LANGS]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.25, 1]})

    series = [
        ("base model", comply("a22", "base"), GREY),
        ("SafeTune steering", comply("a22", "b2_steer_safetune_rd_s0"), TEAL),
        ("first adapter", comply("a22", RELEASED), RED),
        ("second adapter", comply("a24", RELEASED), BLUE),
    ]
    width = 0.2
    for i, (label, ys, color) in enumerate(series):
        xs = [k + (i - 1.5) * width for k in range(len(LANGS))]
        ax.bar(xs, ys, width, label=label, color=color, edgecolor="#272727", linewidth=1.1)
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, y + 0.25, f"{y:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(LANGS)), [NAME[k] for k in LANGS])
    ax.set_ylabel("harmful requests complied with (%)")
    ax.set_title("Held-out harmful prompts, 159 per language", fontsize=13, loc="left")
    ax.legend(ncols=2, loc="upper left", fontsize=11)
    ax.set_ylim(0, 22)
    ax.set_yticks(range(0, 21, 5))

    marker = {"circuit30": "o", "random30": "s", "unrestricted": "^", "inverse30": "v"}
    for run, color in (("a22", RED), ("a24", BLUE)):
        rows = list(csv.DictReader((ROOT / f"docs/results/{run}_b2.csv").open()))
        for cond, m in marker.items():
            pts = [r for r in rows if r["condition"] == cond]
            if not pts:
                continue
            bx.scatter(
                [100 * float(r["comply_xl"]) for r in pts],
                [float(r["fluency_mean"]) for r in pts],
                marker=m,
                s=70,
                facecolors=color if cond == "circuit30" else "none",
                edgecolors=color,
                linewidths=1.6,
                label=f"{NAMES[run]}, {CONDS[cond]}",
            )
        st = [r for r in rows if r["condition"] == "safetune_rd"]
        if st:
            bx.scatter(
                [100 * float(st[0]["comply_xl"])],
                [float(st[0]["fluency_mean"])],
                marker="D",
                s=70,
                color=TEAL,
                label="SafeTune steering",
            )
            base = json.loads((ROOT / "docs/results/a22_b2.json").read_text())["base"]
            bx.scatter(
                [100 * base["comply_xl"]], [1.0], marker="*", s=220, color=GREY, label="base model"
            )
    bx.axhline(1.0, color="#CFCECE", linewidth=1.2, zorder=0)
    bx.set_xlabel("harmful compliance, Hindi / Arabic / French / Swahili (%)")
    bx.set_ylabel("perplexity ratio, unseen text, 5 languages")
    bx.set_title(
        "Every 30 % and all-component adapter: safety against fluency", fontsize=13, loc="left"
    )
    bx.legend(fontsize=9, loc="upper right")
    bx.set_xlim(2.5, 13.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"docs/figures/adapters.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
