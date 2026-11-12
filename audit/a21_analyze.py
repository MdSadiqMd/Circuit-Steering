"""Amendment A21 analysis (plan item 2): MEMIT site sweep, uncapped / norm-capped / norm-matched
noise, from <out_dir> = runs/a21 (interventions/run_a21.py, audited on puget with hf_layers).

  uv run python audit/a21_analyze.py out_dir=runs/a21

Outcomes per cell (audit/analyze.py `load_rows`, the main study's definitions): efficacy
(paired English edited-fact gain), cross-lingual gain (mean of the hi/ar/fr/sw edited-fact
gains), L1 neighbourhood (en), MMLU (en), log WikiText perplexity ratio, L3 mean absolute change,
L5 AURC change. Site score a_L = |signed row sum of MLP L| in the fold's discovery seed (the score
the pre-registered circuit condition picks its edit layer by).
Sealed test: pooled partial Spearman between a_L and each outcome over the 3 folds × sites, with
fold fixed effects and the rank of depth L as covariates, plus the rank of log |ΔW| for uncapped
cells; t-test with n − 2 − q degrees of freedom (q covariates). Primary: {efficacy, L3, L1
neighbourhood} × {uncapped, capped}, Holm at 0.05. Edit − noise per site (capped) and every
per-fold value are reported as descriptive.
Writes docs/results/a21_b1.md and docs/results/a21_b1.csv.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from aya_circuits.config import load_config
from scipy.stats import rankdata
from scipy.stats import t as student_t

from audit.analyze import OTHER_LANGS, load_rows

ROOT = Path(__file__).resolve().parent.parent
OUTCOMES = (
    "efficacy",
    "xl_gain",
    "l1_neigh_en_delta",
    "l1_mmlu_en_delta",
    "log_ppl_ratio",
    "l3_mean_abs_delta",
    "l5_aurc_delta",
)
PRIMARY = ("efficacy", "l3_mean_abs_delta", "l1_neigh_en_delta")
KINDS = ("uncapped", "capped", "noise")
MARKED = {0: "circuit (pre-registered)", 3: "causal trace (A8)", 5: "circuit_deep"}


def kind_layer(cond: str) -> tuple[str, int] | None:
    m = re.fullmatch(r"(perturb_)?site(\d\d)(_cap)?", cond or "")
    if not m:
        return None
    kind = "noise" if m[1] else "capped" if m[3] else "uncapped"
    return kind, int(m[2])


def partial_spearman(y, x, covs: list) -> tuple[float, float, int]:
    """Spearman partial correlation of y and x given covariates (each a list; categorical
    covariates enter as given, continuous ones are ranked by the caller); returns (r, p, df)."""
    Z = np.column_stack([np.ones(len(y)), *covs])
    ry, rx = rankdata(y), rankdata(x)
    res_y = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    res_x = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    r = float(np.corrcoef(res_y, res_x)[0, 1])
    df = len(y) - 2 - (Z.shape[1] - 1)
    tval = r * math.sqrt(df / max(1e-300, 1 - r * r))
    return r, float(2 * student_t.sf(abs(tval), df)), df


def holm(ps: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    out, k = {}, len(ps)
    for i, (name, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        if p > alpha / (k - i):
            out |= {n: False for n, _ in sorted(ps.items(), key=lambda kv: kv[1])[i:]}
            break
        out[name] = True
    return out


def fmt(x, nd: int = 4) -> str:
    return "–" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def main() -> None:
    cfg = load_config()
    rows, _ = load_rows(cfg, "b1")
    score = {}
    for s in cfg["seeds"]:
        n = json.loads((ROOT / "runs/circuits/b1_en" / f"seed{s}" / "nodes.json").read_text())
        d = dict(zip(n["names"], n["signed_rowsum"], strict=True))
        score |= {(s, layer): abs(d[f"MLP {layer}"]) for layer in range(36)}
    data = []
    for r in rows:
        kl = kind_layer(r["condition"])
        if kl is None:
            continue
        xl = [r.get(f"h5_gain_{g}") for g in OTHER_LANGS]
        r = r | {
            "kind": kl[0],
            "layer": kl[1],
            "site_score": score[r["disc_seed"], kl[1]],
            "xl_gain": st.mean(xl) if None not in xl else None,
            "log_ppl_ratio": math.log(r["l1_wikitext_ppl_ratio"]),
        }
        data.append(r)
    L = ["# A21: MEMIT at every editable layer (B1, puget)\n"]
    L.append(f"{len(data)} cells; CAP = 59.187628. Means over the three folds per site.\n")
    for kind in KINDS:
        L.append(f"\n## {kind}\n")
        L.append("| layer | a_L (mean) | |ΔW| | " + " | ".join(OUTCOMES) + " | note |")
        L.append("|---|---|---|" + "---|" * len(OUTCOMES) + "---|")
        for layer in sorted({r["layer"] for r in data}):
            rs = [r for r in data if r["kind"] == kind and r["layer"] == layer]
            if not rs:
                continue
            cells = [
                fmt(st.mean(v) if (v := [r[o] for r in rs if r[o] is not None]) else None)
                for o in OUTCOMES
            ]
            L.append(
                f"| {layer} | {st.mean(r['site_score'] for r in rs):.3f} | "
                f"{st.mean(r['delta_norm'] for r in rs):.1f} | "
                + " | ".join(cells)
                + f" | {MARKED.get(layer, '')} |"
            )
    L.append("\n## Sealed test: partial Spearman of a_L with each outcome\n")
    L.append("Covariates: fold fixed effects, rank of depth, and (uncapped) rank of log |ΔW|.\n")
    L.append("| condition | outcome | n | partial ρ | p | df | per fold ρ (depth only) |")
    L.append("|---|---|---|---|---|---|---|")
    ps, res = {}, {}
    for kind in KINDS:
        sub = [r for r in data if r["kind"] == kind]
        for o in OUTCOMES:
            ok = [r for r in sub if r[o] is not None]
            if len(ok) < 8:
                continue
            folds = sorted({r["disc_seed"] for r in ok})
            covs = [[float(r["disc_seed"] == f) for r in ok] for f in folds[1:]]
            covs.append(rankdata([r["layer"] for r in ok]))
            if kind == "uncapped":
                covs.append(rankdata([math.log(r["delta_norm"]) for r in ok]))
            rho, p, df = partial_spearman([r[o] for r in ok], [r["site_score"] for r in ok], covs)
            per = []
            for f in folds:
                fr = [r for r in ok if r["disc_seed"] == f]
                per.append(
                    partial_spearman(
                        [r[o] for r in fr],
                        [r["site_score"] for r in fr],
                        [rankdata([r["layer"] for r in fr])],
                    )[0]
                )
            res[kind, o] = (rho, p, df, len(ok), per)
            if o in PRIMARY and kind in ("uncapped", "capped"):
                ps[f"{kind}:{o}"] = p
    keep = holm(ps)
    for (kind, o), (rho, p, df, n, per) in res.items():
        tag = f"{kind}:{o}"
        star = (" (Holm: reject)" if keep[tag] else " (Holm: keep)") if tag in keep else ""
        L.append(
            f"| {kind} | {o} | {n} | {rho:+.3f} | {p:.4f}{star} | {df} | "
            + " / ".join(f"{x:+.2f}" for x in per)
            + " |"
        )
    L.append("\n## Edit − noise at equal |ΔW| (capped), mean over folds\n")
    L.append("| layer | " + " | ".join(f"Δ {o}" for o in OUTCOMES) + " |")
    L.append("|---|" + "---|" * len(OUTCOMES))
    for layer in sorted({r["layer"] for r in data}):
        diffs = []
        for o in OUTCOMES:
            d = [
                c[o] - n[o]
                for c in data
                if c["kind"] == "capped" and c["layer"] == layer and c[o] is not None
                for n in data
                if n["kind"] == "noise"
                and n["layer"] == layer
                and n["disc_seed"] == c["disc_seed"]
                and n[o] is not None
            ]
            diffs.append(fmt(st.mean(d) if d else None))
        L.append(f"| {layer} | " + " | ".join(diffs) + " |")
    out_dir = ROOT / "docs/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a21_b1.md").write_text("\n".join(L) + "\n")
    with (out_dir / "a21_b1.csv").open("w", newline="") as f:
        keys = ["cell", "kind", "layer", "disc_seed", "site_score", "delta_norm", *OUTCOMES]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        assert kind_layer("site07") == ("uncapped", 7)
        assert kind_layer("site07_cap") == ("capped", 7)
        assert kind_layer("perturb_site07_cap") == ("noise", 7) and kind_layer("circuit") is None
        rng = np.random.default_rng(0)
        z = rng.normal(size=200)
        x, y = z + rng.normal(size=200), z + rng.normal(size=200)
        r_raw = partial_spearman(y, x, [])[0]
        r_part, p_part, _ = partial_spearman(y, x, [rankdata(z)])
        assert r_raw > 0.3 and abs(r_part) < 0.2 and p_part > 0.01, (r_raw, r_part, p_part)
        assert holm({"a": 0.001, "b": 0.04, "c": 0.03}) == {"a": True, "c": False, "b": False}
        assert holm({"a": 0.001, "b": 0.04, "c": 0.02}) == {"a": True, "c": True, "b": True}
        assert holm({"a": 0.001, "b": 0.02}) == {"a": True, "b": True}
        print("a21_analyze ok")
    else:
        main()
