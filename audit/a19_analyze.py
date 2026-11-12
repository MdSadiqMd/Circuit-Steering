"""Amendment A19 analysis: jackknifed paired CAD contrasts, TOST equivalence verdicts and the
attribution-free bystander measure, from runs/audit/a19/base/* and runs/audit/<beh>/<cell>/a19.*.

Paired contrast (per circuit cell c, control set R of the same discovery seed, bystander task b,
edge set E): Δ = CAD_E(c) − mean_{r∈R} CAD_E(r), where CAD uses the per-example means over the same
32 examples for base and cells. Leave-one-out jackknife over those examples (base and cell means
recomputed without example j, Δ_{-j}, `neb.attribution.jackknife_se`) gives the SE of the contrast
under common random numbers; the resample floor of the main audit measured something else (how the
base attribution moves when the examples change).
Pooling: contrasts of different tasks and seeds use disjoint example sets, so the pooled mean over
(seed, bystander) has SE = sqrt(Σ SE²) / k; the SD of the k contrasts is reported next to it.
TOST (two one-sided tests, α = 0.05): equivalent when the 90 % CI (t, 31 df) lies inside ±δ;
different when the 95 % CI excludes 0; otherwise inconclusive. δ_L4 = 0.05 CAD, δ_L3 = 0.01 mean
absolute accuracy change (sealed in A19). n_required = (z_0.95 + z_0.80)² σ̂² / δ² with
σ̂² = n · SE² gives the per-task examples a conclusive test would need.
L3 contrasts reuse audit/analyze.py's pairs (language × family, hf.json) with a percentile
bootstrap.
Behaviour (part d): per task, the clean loss vector under the cell minus the base's, paired per
example; δ_beh = 5 % of the task's base clean − corrupt span.
Writes docs/results/a19_<beh>.md and docs/results/a19_<beh>.csv.
Run: uv run python audit/a19_analyze.py grid.behaviour=b1
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.nodes import circuit_nodes
from circuitkit.backends.eap.graph import Graph
from neb import mechanistic as M
from neb.attribution import jackknife_se, loo_means
from safetensors.torch import load_file
from scipy.stats import norm
from scipy.stats import t as student_t

from audit.l4 import ROOT, disc_seed_of
from audit.l4_a19 import CONDITIONS, a19_root, tasks_for
from interventions.cell import Cell

GRAPH_CFG = {
    "n_layers": 36,
    "n_heads": 16,
    "d_model": 2048,
    "d_mlp": 11008,
    "parallel_attn_mlp": True,
}
DELTA = {"l4": 0.05, "l3": 0.01}
COMPARISONS = (
    ("circuit", "random"),
    ("circuit", "inverse"),
    ("circuit", "unrestricted"),
    ("circuit", "taylor"),
    ("circuit_joint", "random_joint"),
    ("circuit_joint", "inverse_joint"),
)
DF = 31


def tost(delta: float, se: float, bound: float, df: int = DF) -> dict:
    """Two one-sided tests at α = 0.05 via the 90 % CI, plus the 95 % CI for a difference."""
    t90, t95 = student_t.ppf(0.95, df), student_t.ppf(0.975, df)
    lo90, hi90 = delta - t90 * se, delta + t90 * se
    lo95, hi95 = delta - t95 * se, delta + t95 * se
    equivalent, different = (lo90 > -bound and hi90 < bound), (lo95 > 0 or hi95 < 0)
    verdict = (
        "equivalent (nonzero)"
        if equivalent and different
        else "equivalent"
        if equivalent
        else "different"
        if different
        else "inconclusive"
    )
    return {"ci90": [lo90, hi90], "ci95": [lo95, hi95], "verdict": verdict, "bound": bound}


def spread_tost(xs: list[float], bound: float) -> dict:
    """Post-hoc sensitivity (not sealed): TOST with SE = sample SD of the k contrasts / sqrt(k)
    and k − 1 df, so the interval also covers cell-to-cell variation, which the sealed jackknife SE
    (example sampling with the cells fixed) does not."""
    k = len(xs)
    if k < 2:
        return {"ci90": [float("nan")] * 2, "verdict": "–"}
    return tost(st.mean(xs), st.stdev(xs) / math.sqrt(k), bound, df=k - 1)


def seed_tost(contrasts: list[tuple[float, int]], bound: float) -> dict:
    """Post-hoc, seed-clustered: the bystanders of one seed share its circuit and control cells, so
    each seed's contrasts are averaged first (k = number of seeds, k − 1 df)."""
    by: dict[int, list[float]] = {}
    for x, seed in contrasts:
        by.setdefault(seed, []).append(x)
    return spread_tost([st.mean(v) for v in by.values()], bound)


def n_required(se: float, n: int, bound: float) -> int:
    """Examples per task for a conclusive TOST at power 0.8: (z_0.95 + z_0.80)² σ² / δ² with
    σ² = n · SE²."""
    return math.ceil((norm.ppf(0.95) + norm.ppf(0.80)) ** 2 * n * se**2 / bound**2)


def cad_series(base_perex: torch.Tensor, cell_perex: torch.Tensor) -> tuple[float, torch.Tensor]:
    """(CAD on the full sample, leave-one-out CADs [N]) from per-example scores [N, E] of base and
    cell on the same examples."""
    b, c = base_perex.double(), cell_perex.double()
    full = float((b.mean(0) - c.mean(0)).abs().sum() / b.mean(0).abs().sum())
    bl, cl = loo_means(b), loo_means(c)
    return full, (bl - cl).abs().sum(1) / bl.abs().sum(1)


def fmt(x, nd=4) -> str:
    return "–" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def main() -> None:
    cfg = load_config()
    beh = cfg["grid"]["behaviour"]
    runs, top = Path(cfg["out_dir"]), cfg["discovery"]["top_fraction"]
    graph = Graph.from_model(GRAPH_CFG)
    rule = json.loads((a19_root(cfg) / "rule.json").read_text())["rule"]
    tasks = tasks_for(beh)
    bystanders = tasks[1:]
    cells: dict[str, dict] = {}
    for cj in sorted((runs / "interventions" / beh).glob("*/*/cell.json")):
        c = Cell.peek(cj.parent)
        p = runs / "audit" / beh / c.name / "a19.safetensors"
        if c.condition in CONDITIONS and "@" not in c.condition and p.exists():
            cells[c.name] = {"cell": c, "seed": disc_seed_of(c, cfg), "t": load_file(str(p))}
    bases = {
        (task, s): load_file(str(a19_root(cfg) / "base" / f"{task}_s{s}.safetensors"))
        for task in tasks
        for s in cfg["seeds"]
    }
    masks: dict[tuple[str, int], torch.Tensor] = {}
    for task, s in bases:
        nodes = json.loads((runs / "circuits" / task / f"seed{s}" / "nodes.json").read_text())
        signed = dict(zip(nodes["names"], nodes["signed_rowsum"], strict=True))
        signed.pop("input", None)
        masks[task, s] = (
            M.induced_edge_mask(graph, circuit_nodes(signed, top)).flatten().nonzero().flatten()
        )

    def perex(name: str, task: str) -> torch.Tensor:
        """Per-example scores of a cell on the task's per-type edge set."""
        c = cells[name]
        idx_e, idx_u = masks[task, c["seed"]], c["t"][f"{task}_idx"]
        pos = torch.searchsorted(idx_u, idx_e)
        assert torch.equal(idx_u[pos], idx_e), "per-type edges missing from the stored union"
        return c["t"][f"{task}_perex"][:, pos]

    def base_perex(task: str, s: int) -> torch.Tensor:
        n = bases[task, s]["scores"].shape[0]
        return bases[task, s]["scores"].reshape(n, -1)[:, masks[task, s]]

    def by(interv: str, cond: str) -> list[str]:
        return [
            n
            for n, c in cells.items()
            if c["cell"].intervention == interv and c["cell"].condition == cond
        ]

    rows, L = [], [f"# A19 for {beh}: jackknifed paired CAD, TOST verdicts, bystander behaviour\n"]
    L.append(
        f"Rule `{rule}` (sealed by `audit/l4_a19.py stage=validate`, "
        "`checks/results/quadrature_tinyaya.json`), 32 examples per task, per-type edge set of "
        f"each task, {len(cells)} cells. δ_L4 = {DELTA['l4']}, δ_L3 = {DELTA['l3']}; TOST at "
        "α = 0.05 (90 % CI inside ±δ), 'different' when the 95 % CI excludes 0.\n"
    )
    # 1. paired L4 contrasts
    L.append("## L4: paired CAD contrasts on the bystander circuits\n")
    L.append(
        "| intervention | comparison | k (seed × bystander) | pooled Δ CAD | SE (jackknife) | "
        "SD across k | 90 % CI | verdict | n per task for a verdict | post-hoc 90 % CI "
        "(SE = SD/√k, k − 1 df) | post-hoc verdict | seed-clustered 90 % CI (k = seeds) | "
        "seed-clustered verdict |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for interv in ("steer", "lora", "edit"):
        for circ, ctrl in COMPARISONS:
            cs, rs = by(interv, circ), by(interv, ctrl)
            if not cs or not rs:
                continue
            contrasts = []
            for name in cs:
                s = cells[name]["seed"]
                ctrls = [r for r in rs if cells[r]["seed"] == s]
                if not ctrls:
                    continue
                for b in bystanders:
                    bp = base_perex(b, s)
                    full_c, loo_c = cad_series(bp, perex(name, b))
                    ctl = [cad_series(bp, perex(r, b)) for r in ctrls]
                    full = full_c - st.mean(f for f, _ in ctl)
                    loo = loo_c - torch.stack([series for _, series in ctl]).mean(0)
                    se = jackknife_se(loo)
                    contrasts.append((full, se, name, b))
                    rows.append(
                        {
                            "cell": name,
                            "comparison": f"{circ}-{ctrl}",
                            "task": b,
                            "cad_circuit": full_c,
                            "cad_control_mean": st.mean(f for f, _ in ctl),
                            "contrast": full,
                            "se_jackknife": se,
                            **{f"tost_{k}": v for k, v in tost(full, se, DELTA["l4"]).items()},
                        }
                    )
            if not contrasts:
                continue
            k = len(contrasts)
            pooled = st.mean(c[0] for c in contrasts)
            se_pooled = math.sqrt(sum(c[1] ** 2 for c in contrasts)) / k
            sd_across = st.pstdev(c[0] for c in contrasts) if k > 1 else 0.0
            v = tost(pooled, se_pooled, DELTA["l4"])
            w = spread_tost([c[0] for c in contrasts], DELTA["l4"])
            z = seed_tost([(c[0], cells[c[2]]["seed"]) for c in contrasts], DELTA["l4"])
            L.append(
                f"| {interv} | {circ} − {ctrl} | {k} | {pooled:+.4f} | {se_pooled:.4f} | "
                f"{sd_across:.4f} | [{v['ci90'][0]:+.4f}, {v['ci90'][1]:+.4f}] | {v['verdict']} | "
                f"{n_required(se_pooled, 32, DELTA['l4'])} | "
                f"[{w['ci90'][0]:+.4f}, {w['ci90'][1]:+.4f}] | {w['verdict']} | "
                f"[{z['ci90'][0]:+.4f}, {z['ci90'][1]:+.4f}] | {z['verdict']} |"
            )
    L.append(
        "\nPer-cell rows (contrast, jackknife SE, TOST) are in the CSV. A contrast's SE is the "
        "uncertainty of the paired difference under identical examples; it is not the resample "
        "floor."
    )
    # 2. L3 contrasts with TOST (existing hf.json pairs through audit/analyze.py)
    from audit.analyze import load_rows, pairs_for

    main_rows, _ = load_rows(cfg, beh)
    main_rows = [r for r in main_rows if r["condition"] and "@" not in r["condition"]]
    L.append("\n## L3: cross-lingual collateral contrasts with TOST\n")
    L.append(
        "| intervention | comparison | n pairs | Δ L3 | 90 % CI (bootstrap) | 95 % CI | verdict |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for interv in ("steer", "lora", "edit"):
        for circ, ctrl in COMPARISONS:
            c = [r for r in main_rows if r["intervention"] == interv and r["condition"] == circ]
            o = [r for r in main_rows if r["intervention"] == interv and r["condition"] == ctrl]
            pairs = pairs_for(c, o, "l3_mean_abs_delta")
            if len(pairs) < 2:
                continue
            diffs = [cc - bb for bb, cc, *_ in pairs]
            rng = random.Random(0)
            means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
            ci90, ci95 = (means[100], means[1899]), (means[50], means[1949])
            d = st.mean(diffs)
            eq, diff = ci90[0] > -DELTA["l3"] and ci90[1] < DELTA["l3"], ci95[0] > 0 or ci95[1] < 0
            verdict = (
                "equivalent (nonzero)"
                if eq and diff
                else "equivalent"
                if eq
                else "different"
                if diff
                else "inconclusive"
            )
            L.append(
                f"| {interv} | {circ} − {ctrl} | {len(pairs)} | {d:+.4f} | "
                f"[{ci90[0]:+.4f}, {ci90[1]:+.4f}] | [{ci95[0]:+.4f}, {ci95[1]:+.4f}] | {verdict} |"
            )
    L.append(
        "\nL3 pairs are language × family within a seed; the statistical unit remains the three "
        "discovery seeds (docs/audit-2026-11-08.md B.5)."
    )
    # 3. behaviour of the bystander tasks under the cells (attribution-free)
    L.append(
        "\n## Bystander behaviour under the cells (clean logit difference, paired per example)\n"
    )
    L.append(
        "| intervention | condition | n cells | task | mean Δ loss | SE | mean Δ accuracy | "
        "δ_beh | verdict |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for interv in ("steer", "lora", "edit"):
        for cond in sorted(CONDITIONS):
            names = by(interv, cond)
            if not names:
                continue
            for task in tasks:
                ds, accs, ses, bounds = [], [], [], []
                for name in names:
                    s = cells[name]["seed"]
                    lc, lb = cells[name]["t"][f"{task}_clean_loss"], bases[task, s]["clean_loss"]
                    d = (lc - lb).double()
                    ds.append(float(d.mean()))
                    ses.append(float(d.std(unbiased=True) / math.sqrt(d.numel())))
                    accs.append(float((lc < 0).float().mean() - (lb < 0).float().mean()))
                    bounds.append(0.05 * float((bases[task, s]["corrupt_loss"] - lb).mean().abs()))
                se = math.sqrt(sum(x**2 for x in ses)) / len(ses)
                v = tost(st.mean(ds), se, st.mean(bounds))
                L.append(
                    f"| {interv} | {cond} | {len(names)} | {task} | {st.mean(ds):+.4f} | "
                    f"{se:.4f} | {st.mean(accs):+.3f} | {st.mean(bounds):.3f} | {v['verdict']} |"
                )
    L.append(
        "\nLoss = −(logit difference) of the task's clean examples; accuracy = share with a "
        "positive logit difference. A positive Δ loss is a weaker bystander behaviour. The target "
        "task's row is the behavioural efficacy on the attribution examples."
    )
    out_dir = ROOT / "docs/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"a19_{beh}.md").write_text("\n".join(L) + "\n")
    if rows:
        with (out_dir / f"a19_{beh}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("\n".join(L))


if __name__ == "__main__":
    main()
