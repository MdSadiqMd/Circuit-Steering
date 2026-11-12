"""Amendment A20 analysis (plan item 1): the certified circuit, and steering / LoRA on it against
its controls, from <out_dir> = runs/a20 (built by interventions/run_a20.py, audited on puget).

  uv run python audit/a20_analyze.py out_dir=runs/a20

1. Certification per discovery seed (aya_circuits.certify): size, sufficiency, necessity, area.
2. Per condition: efficacy (paired edited-fact gain, English), L3, L1, L2, L5, |Δ|, trained entries
   (audit/analyze.py `load_rows`, the main study's definitions on hf.json).
3. H1 rule on the certified set: mean efficacy(cert) >= 0.9 × mean efficacy(unrestricted).
4. H2 contrasts cert − control, control in {cert_strat (primary, entry-matched), cert_random,
   cert_inverse, unrestricted}:
   L3  language × family pairs within a seed: analyze.test (pair and seed-cluster bootstrap) and a
       TOST at δ_L3 = 0.01 on the pair bootstrap (A19 (c));
   L4  per-example EAP-IG under A19's sealed rule (audit/l4_a19.py stage=cells, a19.safetensors):
       jackknifed paired CAD on each bystander's per-type edge set, pooled over seeds × bystanders,
       TOST at δ_L4 = 0.05 (A19 (b), (c)).
Writes docs/results/a20_b1.md and docs/results/a20_b1.csv.
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
from neb.attribution import jackknife_se
from safetensors.torch import load_file

from audit.a19_analyze import DELTA, GRAPH_CFG, cad_series, n_required, seed_tost, spread_tost, tost
from audit.analyze import load_rows, pairs_for, test
from audit.l4 import ROOT
from audit.l4_a19 import tasks_for

CONTROLS = ("cert_strat", "cert_random", "cert_inverse", "unrestricted")
METRICS = (
    "efficacy",
    "l3_mean_abs_delta",
    "l1_neigh_en_delta",
    "l1_mmlu_en_delta",
    "l1_wikitext_ppl_ratio",
    "l2_benign_refusal_delta_en",
    "l5_aurc_delta",
    "delta_norm",
    "n_trained",
)


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return st.mean(xs) if xs else None


def fmt(x, nd: int = 4) -> str:
    return "–" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def l3_tost(pairs: list[tuple]) -> str:
    diffs = [c - b for b, c, *_ in pairs]
    rng = random.Random(0)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
    lo90, hi90, lo95, hi95 = means[100], means[1899], means[50], means[1949]
    eq = lo90 > -DELTA["l3"] and hi90 < DELTA["l3"]
    diff = lo95 > 0 or hi95 < 0
    v = (
        ("equivalent (nonzero)" if diff else "equivalent")
        if eq
        else ("different" if diff else "inconclusive")
    )
    return f"90 % CI [{lo90:+.4f}, {hi90:+.4f}] → {v}"


def main() -> None:
    cfg = load_config()
    runs, beh, top = Path(cfg["out_dir"]), "b1", cfg["discovery"]["top_fraction"]
    L = ["# A20: certified circuit, steering and LoRA on it (B1, puget)\n"]
    L.append("## Certification (held-out fold, rule sealed in A20)\n")
    L.append(
        "| seed | certified k | heads | MLPs | sufficiency | necessity r′ (mean) | "
        "necessity r′ (zero) | area [0, 0.5] | held-out n |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in cfg["seeds"]:
        c = json.loads((runs / "circuits/b1_en" / f"seed{s}" / "certified.json").read_text())
        k = c["certified"]
        L.append(
            f"| {s} | {k and k['k']} | {k and k['n_heads']} | {k and k['n_mlps']} | "
            f"{fmt(k and k['sufficiency'])} | {fmt(k and k['necessity_mean'])} | "
            f"{fmt(k and k['necessity_zero'])} | {c['area_sufficiency_0_0.5']:.4f} | "
            f"{c['heldout']['n']} |"
        )
    rows, _ = load_rows(cfg, beh)
    L.append("\n## Cells: means per condition\n")
    L.append("| intervention | condition | n | " + " | ".join(METRICS) + " |")
    L.append("|---|---|---|" + "---|" * len(METRICS))
    for interv in ("steer", "lora"):
        for cond in ("cert", *CONTROLS):
            rs = [r for r in rows if r["intervention"] == interv and r["condition"] == cond]
            if rs:
                L.append(
                    f"| {interv} | {cond} | {len(rs)} | "
                    + " | ".join(fmt(mean(r.get(m) for r in rs)) for m in METRICS)
                    + " |"
                )
    L.append("\n## H1 on the certified set: efficacy(cert) ≥ 0.9 × efficacy(unrestricted)\n")
    for interv in ("steer", "lora"):
        c = [r for r in rows if r["intervention"] == interv and r["condition"] == "cert"]
        u = [r for r in rows if r["intervention"] == interv and r["condition"] == "unrestricted"]
        mc, mu = mean(r["efficacy"] for r in c), mean(r["efficacy"] for r in u)
        if mc is None or not mu:
            continue
        per = {r["disc_seed"]: r["efficacy"] for r in c}
        peru = {r["disc_seed"]: r["efficacy"] for r in u}
        L.append(
            f"- {interv}: {mc:.4f} / {mu:.4f} = ratio {mc / mu:.3f} → "
            f"{'met' if mc >= 0.9 * mu else 'not met'}; per seed "
            + ", ".join(f"s{s} {per[s]:+.3f} vs {peru[s]:+.3f}" for s in sorted(per) if s in peru)
        )
    L.append("\n## H2 on L3: cert − control (language × family pairs within a seed)\n")
    for interv in ("steer", "lora"):
        c = [r for r in rows if r["intervention"] == interv and r["condition"] == "cert"]
        for ctrl in CONTROLS:
            o = [r for r in rows if r["intervention"] == interv and r["condition"] == ctrl]
            pairs = pairs_for(c, o, "l3_mean_abs_delta")
            if len(pairs) >= 2:
                L.append(
                    f"- {interv}, cert − {ctrl}: {test(pairs)}; TOST δ = {DELTA['l3']}: "
                    f"{l3_tost(pairs)}"
                )
    out_rows = l4_section(cfg, runs, beh, top, L)
    out_dir = ROOT / "docs/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a20_b1.md").write_text("\n".join(L) + "\n")
    with (out_dir / "a20_b1.csv").open("w", newline="") as f:
        keys = ["cell", "intervention", "condition", "seed", "disc_seed", *METRICS]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if out_rows:
        with (out_dir / "a20_b1_l4.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
    print("\n".join(L))


def l4_section(cfg: dict, runs: Path, beh: str, top: float, L: list[str]) -> list[dict]:
    """Jackknifed paired CAD contrasts under A19's rule (same code path as audit/a19_analyze.py)."""
    graph = Graph.from_model(GRAPH_CFG)
    tasks = tasks_for(beh)
    a19 = runs / "audit" / "a19"
    rule = json.loads((a19 / "rule.json").read_text())["rule"]
    cells = {}
    for p in sorted((runs / "audit" / beh).glob("*/a19.safetensors")):
        cell = json.loads((p.parent / "a19.json").read_text())
        _, interv, rest = p.parent.name.split("_", 2)
        cond = rest.rsplit("_s", 1)[0]
        cells[p.parent.name] = {
            "interv": interv,
            "cond": cond,
            "seed": cell["disc_seed"],
            "t": load_file(str(p)),
        }
    if not cells:
        L.append("\n## L4: no a19.safetensors cells found\n")
        return []
    idx = {}
    for task in tasks:
        for s in cfg["seeds"]:
            nodes = json.loads((runs / "circuits" / task / f"seed{s}" / "nodes.json").read_text())
            signed = dict(zip(nodes["names"], nodes["signed_rowsum"], strict=True))
            signed.pop("input", None)
            idx[task, s] = (
                M.induced_edge_mask(graph, circuit_nodes(signed, top)).flatten().nonzero().flatten()
            )

    def perex(name: str, task: str) -> torch.Tensor:
        c = cells[name]
        e, u = idx[task, c["seed"]], c["t"][f"{task}_idx"]
        pos = torch.searchsorted(u, e)
        assert torch.equal(u[pos], e), "per-type edges missing from the stored union"
        return c["t"][f"{task}_perex"][:, pos]

    def base_perex(task: str, s: int) -> torch.Tensor:
        t = load_file(str(a19 / "base" / f"{task}_s{s}.safetensors"))["scores"]
        return t.reshape(t.shape[0], -1)[:, idx[task, s]]

    L.append(
        f"\n## H2 on L4: jackknifed paired CAD (rule `{rule}`, bystanders {', '.join(tasks[1:])})\n"
    )
    L.append(
        "| intervention | comparison | k | pooled Δ CAD | SE | SD across k | 90 % CI | "
        "verdict | n per task for a verdict | post-hoc 90 % CI (SE = SD/√k, k − 1 df) | "
        "post-hoc verdict | seed-clustered 90 % CI (k = seeds) | seed-clustered verdict |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    out, bases = [], {}
    for interv in ("steer", "lora"):
        for ctrl in CONTROLS:
            cs = [n for n, c in cells.items() if c["interv"] == interv and c["cond"] == "cert"]
            con = []
            for name in cs:
                s = cells[name]["seed"]
                rs = [
                    n
                    for n, c in cells.items()
                    if c["interv"] == interv and c["cond"] == ctrl and c["seed"] == s
                ]
                for b in tasks[1:] if rs else []:
                    bp = bases.setdefault((b, s), base_perex(b, s))
                    full_c, loo_c = cad_series(bp, perex(name, b))
                    ctl = [cad_series(bp, perex(r, b)) for r in rs]
                    d = full_c - st.mean(f for f, _ in ctl)
                    se = jackknife_se(loo_c - torch.stack([x for _, x in ctl]).mean(0))
                    con.append((d, se, s))
                    out.append(
                        {
                            "cell": name,
                            "comparison": f"cert-{ctrl}",
                            "task": b,
                            "cad_cert": full_c,
                            "cad_control_mean": st.mean(f for f, _ in ctl),
                            "contrast": d,
                            "se_jackknife": se,
                            "verdict": tost(d, se, DELTA["l4"])["verdict"],
                        }
                    )
            if not con:
                continue
            k = len(con)
            pooled, se_p = st.mean(c[0] for c in con), math.sqrt(sum(c[1] ** 2 for c in con)) / k
            v = tost(pooled, se_p, DELTA["l4"])
            w = spread_tost([c[0] for c in con], DELTA["l4"])
            z = seed_tost([(c[0], c[2]) for c in con], DELTA["l4"])
            L.append(
                f"| {interv} | cert − {ctrl} | {k} | {pooled:+.4f} | {se_p:.4f} | "
                f"{st.pstdev(c[0] for c in con) if k > 1 else 0.0:.4f} | "
                f"[{v['ci90'][0]:+.4f}, {v['ci90'][1]:+.4f}] | {v['verdict']} | "
                f"{n_required(se_p, 32, DELTA['l4'])} | "
                f"[{w['ci90'][0]:+.4f}, {w['ci90'][1]:+.4f}] | {w['verdict']} | "
                f"[{z['ci90'][0]:+.4f}, {z['ci90'][1]:+.4f}] | {z['verdict']} |"
            )
    return out


if __name__ == "__main__":
    main()
