"""Hypothesis analysis H1..H5 from the audit results (docs/preregistration.md decision rules).

Reads runs/audit/<beh>/<cell>/{hf,l4}.json, runs/circuits/**/{faithfulness,random_baseline}.json
and the cell.json files; writes docs/results/<beh>_cells.csv (one row per cell: efficacy, every
collateral metric, Δθ norm), the same row as scorecard.json next to every cell artifact, and
docs/results/hypotheses_<beh>.md. Statistics: AuditKIT `paired_bootstrap` (delta, p) plus a
percentile bootstrap 95 % CI of the same paired mean difference; Spearman from scipy.
Run: uv run python audit/analyze.py grid.behaviour=b1
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import auditkit as ak
from auditkit._bootstrap import paired_bootstrap, score_pairs
from aya_circuits.config import load_config
from aya_circuits.overlap import weighted_jaccard
from safetensors.torch import load_file
from scipy.stats import spearmanr, ttest_1samp

ROOT = Path(__file__).resolve().parent.parent
BYSTANDERS = {"b1": ["b2_en", "ioi", "langid_en"], "b2": ["b1_en", "ioi", "langid_en"]}
OTHER_LANGS = ["hi", "ar", "fr", "sw"]
COMPARISONS = (
    "random",
    "inverse",
    "taylor",
    "unrestricted",
    "perturb_circuit",
    "circuit_deep",
    "safetune_rd",  # SafeTune refusal-direction steering baseline (B2, amendment A4)
    "causal_trace",  # MEMIT at the English causal-tracing layer (B1, H5 site, amendment A8)
)
L4_KEYS = (
    "l4_bystander_cad_mean",
    "l4_bystander_cad_excess_mean",
    "l4_target_cad",
    "l4_target_cad_excess",
    "l4_as_nc",
    "l4_target_circuit_distance",
)
PILLAR_KEYS = (
    "patching",
    "zero_raw",
    "mean_raw",
    "stability",
    "random_margin",
    "ablation_margin",
    "cell_patching",
    "cell_zero_raw",
    "cell_mean_raw",
    "cell_random_margin",
    "cell_necessity_mean",
    "cell_necessity_margin",
)
REFUSAL_FAMILIES_EN = [
    "refusal_harmful_en",
    "refusal_benign_en",
    "refusal_other_en",
    "refusal_xstest_safe",
]
COLLATERAL = [
    "l4_bystander_cad_mean",
    "l3_mean_abs_delta",
    "l2_benign_refusal_delta_en",
    "l1_neigh_en_delta",
    "l1_wikitext_ppl_ratio",
]


def head(d: dict, key: str, metric: str):
    return d.get(key, {}).get("headline", {}).get(metric)


def paired_delta(base: dict, cell: dict, key: str) -> float | None:
    """Mean over the cell's sample ids of (cell score − base score on the same id)."""
    c, b = cell.get(key, {}).get("scores"), base.get(key, {}).get("scores")
    if not c or b is None:
        return None
    return st.mean(c[i] - b.get(i, 0.0) for i in c)


TESTS: list[tuple[str, float, float]] = []  # (label, Δ, bootstrap p) of registered H2 tests


def ci(
    pairs: list[tuple[float, float]], n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap 95 % CI of the paired mean difference (candidate − baseline) and the
    two-sided bootstrap p-value 2·min(P(mean* ≤ 0), P(mean* ≥ 0)), floored at 1/n_resamples."""
    diffs = [c - b for b, c in pairs]
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_resamples))
    p = 2 * min(sum(m <= 0 for m in means), sum(m >= 0 for m in means)) / n_resamples
    return (
        means[int(0.025 * n_resamples)],
        means[int(0.975 * n_resamples) - 1],
        max(p, 1 / n_resamples),
    )


def holm_lines(tests: list[tuple[str, float, float]]) -> list[str]:
    """Holm step-down at α = 0.05 over the registered tests (bootstrap p, floor 1/2000)."""
    m = len(tests)
    order = sorted(tests, key=lambda t: t[2])
    survive = 0
    for i, (_, _, p) in enumerate(order):
        if p <= 0.05 / (m - i):
            survive += 1
        else:
            break
    raw = sum(p < 0.05 for _, _, p in tests)
    return [
        f"\n## Multiple comparisons (Holm step-down, α = 0.05, over the {m} H2 tests above)\n",
        f"{raw} of {m} have bootstrap p < 0.05 uncorrected; {survive} survive Holm. The bootstrap "
        "p is floored at 1/2000, so a test whose 2000 resampled means share one sign reports "
        f"p = 0.0005 and passes any Holm threshold above it (0.05/m = {0.05 / max(m, 1):.4f}).",
    ] + [
        f"- {lab}: Δ={d:+.4f}, p={p:.4f}{' (survives)' if i < survive else ''}"
        for i, (lab, d, p) in enumerate(order)
    ]


def test(pairs: list[tuple], register: str | None = None) -> str:
    """pairs: (baseline, candidate) or (baseline, candidate, discovery seed); with seeds a
    seed-cluster CI (`cluster_ci`) is appended to the pair-level one. `register` names a
    confirmatory test for the Holm count."""
    tagged = [p for p in pairs if len(p) == 3]
    pairs = [(p[0], p[1]) for p in pairs]
    if len(pairs) < 2:
        return (
            f"n={len(pairs)} pair(s): "
            + ", ".join(f"{c - b:+.4f}" for b, c in pairs)
            + " (no test; directional)"
        )
    # AuditKIT: delta and means; its p_value is not a null test (docs/notes.md), so it is not shown
    bs = paired_bootstrap(pairs)
    lo, hi, p = ci(pairs)
    if register:
        TESTS.append((register, bs["delta"], p))
    excl = "CI excludes 0" if lo > 0 or hi < 0 else "CI includes 0"
    out = (
        f"Δ={bs['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] ({excl}; n={bs['n']}; "
        f"means {bs['mean_candidate']:.4f} vs {bs['mean_baseline']:.4f})"
    )
    if len(tagged) == len(pairs) and len({k for *_, k in tagged}) > 1:
        clo, chi, k = cluster_ci(tagged)
        cex = "excludes 0" if clo > 0 or chi < 0 else "includes 0"
        out += f"; seed-cluster CI [{clo:+.4f}, {chi:+.4f}] ({k} seeds, {cex})"
        # the statistical unit is the discovery seed (docs/audit-2026-11-08.md B.5): per-seed mean
        # differences and a paired t over them (k − 1 degrees of freedom)
        per = per_seed_means(tagged)
        out += "; per-seed Δ " + " / ".join(f"{v:+.4f}" for v in per.values())
        if k > 2:
            out += f", paired t p={ttest_1samp(list(per.values()), 0).pvalue:.3f}"
    return out


def per_seed_means(pairs: list[tuple[float, float, int]]) -> dict[int, float]:
    by: dict[int, list[float]] = {}
    for b, c, k in pairs:
        by.setdefault(k, []).append(c - b)
    return {k: st.mean(by[k]) for k in sorted(by)}


def ci_np(diffs: list[float], n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI of a mean (vectorised; used for the per-family scorecards)."""
    d = np.asarray(diffs, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_resamples, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return float(lo), float(hi)


def cluster_ci(
    pairs: list[tuple[float, float, int]], n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, int]:
    """Seed-cluster bootstrap 95 % CI of the paired mean difference: whole discovery seeds are
    resampled, so languages, families and bystanders inside a seed are not treated as independent.
    With 3 seeds only 10 distinct resamples exist, so this interval is coarse by construction."""
    by: dict[int, list[float]] = {}
    for b, c, k in pairs:
        by.setdefault(k, []).append(c - b)
    keys = sorted(by)
    rng = random.Random(seed)
    means = sorted(
        st.mean(x for k in rng.choices(keys, k=len(keys)) for x in by[k])
        for _ in range(n_resamples)
    )
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1], len(keys)


LOWER_IS_BETTER = ("refusal_benign_", "refusal_xstest_")


def _stored_run(fam: dict, task: str, ids: list[str]) -> ak.RunResult:
    metric = next(iter(fam["headline"]))
    direction = ak.Direction.MINIMIZE if task.startswith(LOWER_IS_BETTER) else ak.Direction.MAXIMIZE
    sc = ak.scorer(lambda s, o: float(s.metadata["score"]), direction=direction, name=metric)
    samples = [
        ak.Sample(input=i, id=i, actual_output="", metadata={"score": float(fam["scores"][i])})
        for i in ids
    ]
    for x in samples:
        x.task = task
    return ak.evaluate(samples, "precomputed", [sc])


def auditkit_compare(base: dict, cell: dict) -> dict:
    """`ak.compare(base, cell)` for every per-sample family of one cell (brief §3.3 item 5, §5):
    AuditKIT RunResults are rebuilt from the stored per-sample scores, the base restricted to the
    cell's sample ids (compare averages unaligned runs otherwise), and AuditKIT's delta, grade and
    p-value are kept next to a percentile-bootstrap 95 % CI of the paired difference. AuditKIT's p
    resamples uncentred differences and is not a null test (docs/notes.md 2026-11-03)."""
    out = {}
    for fam, c in cell.items():
        b = base.get(fam)
        if not isinstance(c, dict) or "scores" not in c or not b or "scores" not in b:
            continue
        ids = sorted(c["scores"].keys() & b["scores"].keys())
        if len(ids) < 2:
            continue
        rb, rc = _stored_run(b, fam, ids), _stored_run(c, fam, ids)
        cmp = ak.compare(rb, rc)
        td = cmp.per_task_deltas()[0]
        pairs = score_pairs(rb, rc)
        lo, hi = ci_np([y - x for x, y in pairs])
        out[fam] = {
            "metric": td.metric,
            "baseline": td.baseline,
            "candidate": td.candidate,
            "delta": td.delta,
            "ci95": [lo, hi],
            "n_pairs": len(pairs),
            "auditkit_grade": td.grade.value,
            "auditkit_p_uncentred": cmp.significance().get("p_value"),
        }
    return out


def efficacy(beh: str, base: dict, cell: dict) -> float | None:
    if beh == "b1":
        return paired_delta(base, cell, "facts_edited_en")
    b, c = head(base, "refusal_harmful_en", "refusal"), head(cell, "refusal_harmful_en", "refusal")
    return None if b is None or c is None else c - b


def l4_view(beh: str, l4: dict, l4c: dict, which: str) -> tuple[dict, dict, dict]:
    """L4 metrics under one circuit definition: 'joint' (top 61 of 612 by |signed row sum|, as first
    reported, `audit/run_audit.py` → l4.json) or 'cell' (the pre-registered per-type set the cells
    intervene on, `audit/l4_cellset.py` → l4_cellset.json). Returns (metrics, cads, cad_excess)."""
    target = f"{beh}_en"
    if which == "cell":
        if not l4c:
            return dict.fromkeys(L4_KEYS), {}, {}
        v = {
            c: l4c[c]["cell_set"] for c in l4c if isinstance(l4c[c], dict) and "cell_set" in l4c[c]
        }
        as_nc = l4c.get("as_nc", {}).get("cell_set", {}).get("as_nc")
        dist = v.get(target, {}).get("circuit_distance")
    else:
        v = l4
        as_nc = l4.get("as_nc", {}).get("as_nc")
        dist = l4.get(target, {}).get("circuit_distance", {}).get("circuit_distance")
    cads = {b: v[b]["cad_induced"]["cad"] for b in BYSTANDERS[beh] if b in v}
    # CAD − same-settings resample floor (base vs base on another example subset): floors differ per
    # bystander circuit, so cross-bystander comparisons (H4) use the excess; paired per-bystander
    # tests
    # cancel the floor anyway.
    excess = {b: cads[b] - v[b]["cad_resample_induced"]["cad"] for b in cads}
    tgt = v.get(target, {}).get("cad_induced", {}).get("cad")
    out = {
        "l4_bystander_cad_mean": st.mean(cads.values()) if cads else None,
        "l4_bystander_cad_excess_mean": st.mean(excess.values()) if excess else None,
        "l4_target_cad": tgt,
        "l4_target_cad_excess": (
            tgt - v[target]["cad_resample_induced"]["cad"] if tgt is not None else None
        ),
        "l4_as_nc": as_nc,
        "l4_target_circuit_distance": dist,
    }
    return out, cads, excess


def collateral(
    beh: str, base: dict, cell: dict, l4: dict, l4c: dict, langs: list[str], primary: str
) -> tuple[dict, dict]:
    out = {}
    l3 = {}
    for lang in langs:
        if lang == "en":
            continue
        for key, m in (
            ("facts_neighbourhood_" + lang, "accuracy"),
            ("refusal_benign_" + lang, "refusal"),
            ("refusal_other_" + lang, "refusal"),
        ):
            b, c = head(base, key, m), head(cell, key, m)
            if b is not None and c is not None:
                l3[key] = abs(c - b)
    out["l3_mean_abs_delta"] = st.mean(l3.values()) if l3 else None
    out["l1_neigh_en_delta"] = (head(cell, "facts_neighbourhood_en", "accuracy") or 0) - (
        head(base, "facts_neighbourhood_en", "accuracy") or 0
    )
    out["l1_mmlu_en_delta"] = (head(cell, "mmlu_en", "score") or 0) - (
        head(base, "mmlu_en", "score") or 0
    )
    out["l1_wikitext_ppl_ratio"] = (
        cell.get("wikitext", {}).get("metadata", {}).get("perplexity") or float("nan")
    ) / (base.get("wikitext", {}).get("metadata", {}).get("perplexity") or float("nan"))
    out["l2_benign_refusal_delta_en"] = float(
        head(cell, "refusal_benign_en", "refusal") or 0
    ) - float(head(base, "refusal_benign_en", "refusal") or 0)
    out["l2_harmful_refusal_delta_en"] = float(
        head(cell, "refusal_harmful_en", "refusal") or 0
    ) - float(head(base, "refusal_harmful_en", "refusal") or 0)
    out["l2_xstest_refusal_delta"] = (
        (head(cell, "refusal_xstest_safe", "refusal") or 0)
        - (head(base, "refusal_xstest_safe", "refusal") or 0)
        if "refusal_xstest_safe" in cell
        else None
    )
    out["l5_aurc_delta"] = (
        cell["hidden"]["aurc"] - base["hidden"]["aurc"] if "hidden" in cell else None
    )
    out["l5_top1_error_delta"] = (
        cell["hidden"]["top1_error"] - base["hidden"]["top1_error"] if "hidden" in cell else None
    )
    prim, cads, excess = l4_view(beh, l4, l4c, primary)
    joint, cads_j, excess_j = l4_view(beh, l4, l4c, "joint")
    out.update(prim)
    out.update({f"{k}_joint": val for k, val in joint.items()})
    out["l4_repro_max_abs_diff"] = l4c.get("repro_max_abs_diff") if l4c else None
    return out, {
        "l3": l3,
        "cads": cads,
        "cad_excess": excess,
        "cads_joint": cads_j,
        "cad_excess_joint": excess_j,
    }


def arbitration_view(base_ab: dict, cell_ab: dict) -> dict:
    """L5 evidence arbitration (audit/arbitration.py; Anand 2026): spread ratio sd(A) cell / base,
    AURC and most-confident-quarter error changes, and the mean paired drift of A."""
    b, c = base_ab["summary"], cell_ab["summary"]
    ba = {r["id"]: r["A"] for r in base_ab["probes"]}
    return {
        "l5_arb_spread_ratio": c["sd_A"] / b["sd_A"],
        "l5_arb_aurc_delta": c["aurc"] - b["aurc"],
        "l5_arb_high_conf_error_delta": c["high_conf_error"] - b["high_conf_error"],
        "l5_arb_drift_mean": st.mean(r["A"] - ba[r["id"]] for r in cell_ab["probes"]),
    }


def target_edge_l1(runs: Path, beh: str, cell_dir: Path) -> float:
    """Edge-level distance of the whole target-task graph, Σ_e |s_e(θ') − s_e(θ)| / Σ_e |s_e(θ)|
    over every edge (non-edges are 0 in both): the analogue of Chen et al.'s Circuit Distance
    (edge-level L1 on causal effects) with EAP-IG scores, normalised by the base mass instead of
    the range."""
    from safetensors.torch import load_file

    seed = json.loads((cell_dir / "l4_cellset.json").read_text())["disc_seed"]
    t = f"{beh}_en"
    new = load_file(str(cell_dir / "l4_edges.safetensors"), device="cpu")[t]
    base = load_file(str(runs / "audit" / beh / "base" / f"l4_edges_{t}_s{seed}.safetensors"))[
        "edges"
    ]
    return float((new - base).abs().sum() / base.abs().sum())


def relearn_view(base_rl: dict, cell_rl: dict, row: dict) -> dict:
    """L5 relearning probe (audit/relearn.py): refusal gain over the base model when both went
    through the same benign fine-tune, and the share of the cell's pre-attack gain that survives."""
    gain = {
        lang: head(cell_rl, f"refusal_harmful_{lang}", "refusal")
        - head(base_rl, f"refusal_harmful_{lang}", "refusal")
        for lang in ["en", *OTHER_LANGS]
    }
    xl = st.mean(gain[lang] for lang in OTHER_LANGS)
    before = row.get("efficacy_xl")
    out = {f"l5_relearn_gain_{lang}": gain[lang] for lang in OTHER_LANGS}
    out |= {
        "l5_relearn_gain_en": gain["en"],
        "l5_relearn_gain_xl": xl,
        "l5_relearn_retention_xl": xl / before if before is not None and before >= 0.05 else None,
        "l5_relearn_benign_en_delta": head(cell_rl, "refusal_benign_en", "refusal")
        - head(base_rl, "refusal_benign_en", "refusal"),
    }
    return out


def disc_seed_of(seed: int, seeds: list[int]) -> int:
    return seed if seed in seeds else seeds[seed % len(seeds)]


def load_rows(cfg: dict, beh: str) -> tuple[list[dict], dict]:
    runs = Path(cfg["out_dir"])
    base = json.loads((runs / "audit" / beh / "base" / "hf.json").read_text())
    rows = []
    dirs = [d for d in sorted((runs / "audit" / beh).glob("*_s*")) if (d / "hf.json").exists()]
    with_l4 = [d for d in dirs if (d / "l4.json").exists()]
    # The pre-registered (per-type) definition is primary once every L4 cell has been recomputed on
    # it; until then the joint-ranked numbers stand, so no table mixes definitions.
    primary = (
        "cell" if with_l4 and all((d / "l4_cellset.json").exists() for d in with_l4) else "joint"
    )
    # A16: the built-in `ioi` bystander is mis-specified (checks/check_ioi_position.py); once every
    # L4 cell has been re-attributed on `ioi_fixed` (audit/l4_task.py) that task is the bystander
    # under all three mask definitions. `ioi_task=ioi` forces the old one.
    ioi_task = cfg.get("ioi_task") or (
        "ioi_fixed"
        if with_l4 and all((d / "l4_ioi_fixed.json").exists() for d in with_l4)
        else "ioi"
    )
    BYSTANDERS[beh] = [ioi_task if b.startswith("ioi") else b for b in BYSTANDERS[beh]]
    for cell_dir in dirs:
        hf_p, l4_p = cell_dir / "hf.json", cell_dir / "l4.json"
        l4c_p = cell_dir / "l4_cellset.json"
        _, interv, cond_seed = cell_dir.name.split(
            "_", 2
        )  # b1_lora_circuit@0.5_s0 -> lora, circuit@0.5_s0
        cell_json = runs / "interventions" / beh / interv / cond_seed / "cell.json"
        meta = json.loads(cell_json.read_text()) if cell_json.exists() else {}
        hf = json.loads(hf_p.read_text())
        l4 = json.loads(l4_p.read_text()) if l4_p.exists() else {}
        l4c = json.loads(l4c_p.read_text()) if l4c_p.exists() else {}
        fx_p = cell_dir / f"l4_{ioi_task}.json"
        fx = json.loads(fx_p.read_text()) if ioi_task != "ioi" and fx_p.exists() else None
        if fx:  # same layout as l4.json (joint set) and l4_cellset.json (per-type set)
            l4[ioi_task] = fx["joint_set"]
            l4c[ioi_task] = {"cell_set": fx["cell_set"], "joint_set": fx["joint_set"]}
        row = {
            "cell": cell_dir.name,
            "intervention": meta.get("intervention"),
            "condition": meta.get("condition"),
            "seed": meta.get("seed"),
            "disc_seed": disc_seed_of(meta.get("seed", 0), cfg["seeds"]),
            "delta_norm": meta.get("meta", {}).get("delta_norm"),
            "efficacy": efficacy(beh, base, hf),
            "l4_definition": primary,
        }
        col, detail = collateral(beh, base, hf, l4, l4c, cfg["languages"], primary)
        row.update(col)
        # H5: per-language edited-fact gain (B1) / held-out harmful refusal gain (B2) and
        # neighbourhood change
        for lang in OTHER_LANGS:
            if beh == "b1":
                row[f"h5_gain_{lang}"] = paired_delta(base, hf, f"facts_edited_{lang}")
            else:
                b, c = (
                    head(base, f"refusal_harmful_{lang}", "refusal"),
                    head(hf, f"refusal_harmful_{lang}", "refusal"),
                )
                row[f"h5_gain_{lang}"] = None if b is None or c is None else c - b
            nb, nc = (
                head(base, f"facts_neighbourhood_{lang}", "accuracy"),
                head(hf, f"facts_neighbourhood_{lang}", "accuracy"),
            )
            row[f"h5_neigh_{lang}"] = None if nb is None or nc is None else nc - nb
        # B2 exploratory efficacy (amendment A3): the English held-out harmful set is at ceiling in
        # the base model
        # (11/12 refused), so the pre-registered English gain is capped at +0.083; the non-English
        # sets have headroom.
        xl = [
            row[f"h5_gain_{lang}"] for lang in OTHER_LANGS if row.get(f"h5_gain_{lang}") is not None
        ]
        row["efficacy_xl"] = st.mean(xl) if beh == "b2" and len(xl) == len(OTHER_LANGS) else None
        # L1 one-hop ripple (B1, audit/ripple.py) and L5 relearning probe (B2, audit/relearn.py);
        # columns appear once those stages have run, so earlier tables are unchanged
        rp, bp = cell_dir / "ripple.json", runs / "audit" / beh / "base" / "ripple.json"
        if beh == "b1" and rp.exists() and bp.exists():
            row["l1_ripple_gain"] = paired_delta(
                {"ripple": json.loads(bp.read_text())},
                {"ripple": json.loads(rp.read_text())},
                "ripple",
            )
        ab, bb = cell_dir / "arbitration.json", runs / "audit" / beh / "base" / "arbitration.json"
        if ab.exists() and bb.exists():
            row.update(arbitration_view(json.loads(bb.read_text()), json.loads(ab.read_text())))
        if (cell_dir / "l4_edges.safetensors").exists():
            row["l4_target_edge_l1"] = target_edge_l1(runs, beh, cell_dir)
        # adapter entries (PEFT's count, masked rows/columns included) and the entries that train
        # (docs/audit-2026-11-08.md B.2): the masks keep exact zeros, so the nonzero count of the
        # stored factors is the trained count for every LoRA cell, old or new
        row["n_trainable"] = meta.get("meta", {}).get("n_trainable")
        row["n_trained"] = (
            trained_entries(cell_json.with_name("delta.safetensors")) if interv == "lora" else None
        )
        # overlap with the reference set of the cell's family: `circuit` for the pre-registered
        # conditions, `circuit_joint` for the A15 joint conditions (same discovery seed)
        ref = "circuit_joint" if (meta.get("condition") or "").endswith("_joint") else "circuit"
        circ = runs / "interventions" / beh / interv / f"{ref}_s{row['disc_seed']}" / "cell.json"
        if meta.get("nodes") and circ.exists():
            cn = set(json.loads(circ.read_text())["nodes"])
            row["overlap_circuit_nodes"] = len(cn & set(meta["nodes"]))
            row["overlap_reference"] = ref
        lp = cell_dir / "l4_pillar.json"
        if lp.exists():
            pj = json.loads(lp.read_text())
            if fx:
                pj[ioi_task] = fx["pillar_set"]
            tgt = pj[f"{beh}_en"]
            row["l4_pillar_target_cad"] = tgt["cad_induced"]["cad"]
            row["l4_pillar_target_cad_excess"] = (
                tgt["cad_induced"]["cad"] - tgt["cad_resample_induced"]["cad"]
            )
            row["l4_pillar_bystander_cad_excess_mean"] = st.mean(
                pj[b]["cad_induced"]["cad"] - pj[b]["cad_resample_induced"]["cad"]
                for b in BYSTANDERS[beh]
            )
        rl, bl = cell_dir / "relearn.json", runs / "audit" / beh / "base" / "relearn.json"
        if beh == "b2" and rl.exists() and bl.exists():
            row.update(relearn_view(json.loads(bl.read_text()), json.loads(rl.read_text()), row))
        row["_detail"] = detail
        rows.append(row)
    return rows, base


def trained_entries(delta_path: Path) -> int | None:
    if not delta_path.exists():
        return None
    t = load_file(str(delta_path))
    return int(sum(int((v != 0).sum()) for k, v in t.items() if k.startswith(("A:", "B:"))))


def pairs_for(c: list[dict], other: list[dict], metric: str) -> list[tuple]:
    """(comparison, circuit, discovery seed) triples at the pre-registered pairing level: bystander
    for L4, language × family for L3, the cell for everything else."""
    if metric == "l4_bystander_cad_mean":
        return [
            (o["_detail"]["cads"][b], r["_detail"]["cads"][b], r["disc_seed"])
            for r in c
            for o in other
            if o["disc_seed"] == r["disc_seed"]
            for b in r["_detail"]["cads"]
            if b in o["_detail"]["cads"]
        ]
    if metric == "l3_mean_abs_delta":
        return [
            (o["_detail"]["l3"][k], r["_detail"]["l3"][k], r["disc_seed"])
            for r in c
            for o in other
            if o["disc_seed"] == r["disc_seed"]
            for k in r["_detail"]["l3"]
            if k in o["_detail"]["l3"]
        ]
    return [
        (o[metric], r[metric], r["disc_seed"])
        for r in c
        for o in other
        if o["disc_seed"] == r["disc_seed"] and r[metric] is not None and o[metric] is not None
    ]


def h1_text(c: list[dict], u: list[dict], key: str, label: str, name: str = "circuit") -> str:
    """H1 line: efficacy ratio circuit / unrestricted with a bootstrap CI of the ratio."""
    if not u or not all(r[key] is not None for r in c + u):
        return ""
    pairs = [
        (uu[key], cc[key], cc["disc_seed"])
        for cc in c
        for uu in u
        if uu["disc_seed"] == cc["disc_seed"]
    ]
    mu, mc = st.mean(r[key] for r in u), st.mean(r[key] for r in c)
    if mu <= 0:
        return (
            f"- **{label}** undefined: unrestricted efficacy {mu:.3f} ≤ 0 "
            f"({name} {mc:.3f}); paired per seed: {test(pairs)}"
        )
    rng = random.Random(0)
    ratios = sorted(
        st.mean(p[1] for p in s) / max(1e-9, st.mean(p[0] for p in s))
        for s in (rng.choices(pairs, k=len(pairs)) for _ in range(2000))
    )
    vac = (
        " — vacuous: both efficacies < 0.1, the intervention does not produce the behaviour"
        if mc < 0.1 and mu < 0.1
        else ""
    )
    return (
        f"- **{label}** efficacy {name}/unrestricted = {mc / mu:.3f} "
        f"[{ratios[50]:.3f}, {ratios[1949]:.3f}] → "
        f"{'supported' if mc / mu >= 0.9 else 'refuted'} at the 0.9 rule{vac}; "
        f"paired per seed: {test(pairs)}"
    )


def mean_of(rs: list[dict], k: str) -> str:
    vals = [r[k] for r in rs if r[k] is not None and r[k] == r[k]]
    return fmt(st.mean(vals)) if vals else "–"


def fmt(x, nd=3):
    return (
        "–"
        if x is None or (isinstance(x, float) and x != x)
        else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
    )


def judge_robustness(runs: Path, beh: str, base: dict, main_rows: list[dict]) -> list[str]:
    """Re-score the stored English refusal generations with SafeTune's detector and compare with the
    AuditKIT verdicts that every reported refusal number uses (per-sample agreement, and the
    per-condition
    harmful / benign refusal deltas under both judges)."""
    from neb.judge import refused_safetune

    def verdicts(hf: dict) -> dict[str, tuple[list[float], list[float]]]:
        out = {}
        for fam in REFUSAL_FAMILIES_EN:
            if fam in hf:
                sc, outs = hf[fam]["scores"], hf[fam]["outputs"]
                out[fam] = ([sc[i] for i in sc], [float(refused_safetune(outs[i])) for i in sc])
        return out

    vb = verdicts(base)
    agree = n = 0
    per_cell = {}
    for r in main_rows:
        hf = json.loads((runs / "audit" / beh / r["cell"] / "hf.json").read_text())
        v = verdicts(hf)
        for fam in v:
            agree += sum(a == b for a, b in zip(*v[fam], strict=True))
            n += len(v[fam][0])
        per_cell[r["cell"]] = {
            fam: (
                st.mean(v[fam][0]) - st.mean(vb[fam][0]),
                st.mean(v[fam][1]) - st.mean(vb[fam][1]),
            )
            for fam in v
            if fam in vb
        }
    for fam in vb:
        agree += sum(a == b for a, b in zip(*vb[fam], strict=True))
        n += len(vb[fam][0])
    L = [
        "\n## Judge robustness: AuditKIT vs SafeTune refusal detectors\n",
        f"Per-generation agreement over the base model and the {len(main_rows)} main cells, "
        "English refusal families "
        f"(first 200 characters of each stored generation): {agree}/{n} = {agree / max(n, 1):.3f}. "
        "Base rates (AuditKIT / SafeTune): "
        + ", ".join(f"{fam} {st.mean(vb[fam][0]):.2f} / {st.mean(vb[fam][1]):.2f}" for fam in vb)
        + ". Every number elsewhere in this file is AuditKIT-judged.\n",
        "| intervention | condition | n | harmful Δ (AuditKIT / SafeTune) | "
        "benign Δ (AuditKIT / SafeTune) | other-category Δ (AuditKIT / SafeTune) | "
        "XSTest-safe Δ (AuditKIT / SafeTune) |",
        "|---|---|---|---|---|---|---|",
    ]
    for interv in ("steer", "lora", "edit"):
        for cond in sorted({r["condition"] for r in main_rows if r["intervention"] == interv}):
            rs = [r for r in main_rows if r["intervention"] == interv and r["condition"] == cond]
            cols = []
            for fam in REFUSAL_FAMILIES_EN:
                pts = [per_cell[r["cell"]][fam] for r in rs if fam in per_cell[r["cell"]]]
                cols.append(
                    f"{st.mean(p[0] for p in pts):+.3f} / {st.mean(p[1] for p in pts):+.3f}"
                    if pts
                    else "–"
                )
            L.append(f"| {interv} | {cond} | {len(rs)} | " + " | ".join(cols) + " |")
    return L


def pillars(runs: Path, task: str, seed: int) -> dict:
    d = json.loads((runs / "circuits" / task / f"seed{seed}" / "faithfulness.json").read_text())
    z, m = d["full"].get("zero", {}), d["full"].get("mean", {})
    out = {
        "patching": z.get("metadata", {}).get("patching_raw_ratio"),
        "zero_raw": z.get("metadata", {}).get("ablation_raw_ratio"),
        "mean_raw": m.get("metadata", {}).get("ablation_raw_ratio"),
        "stability": (z.get("stability") or {}).get("mean_jaccard"),
    }
    rb = runs / "circuits" / task / f"seed{seed}" / "random_baseline.json"
    if rb.exists():
        r = json.loads(rb.read_text())
        out["random_margin"] = r["circuit"]["patching_raw_ratio"] - st.mean(
            x["patching_raw_ratio"] for x in r["random_node"]
        )
        out["ablation_margin"] = r["circuit"]["ablation_zero_raw"] - st.mean(
            x["ablation_zero_raw"] for x in r["random_node"]
        )
    cs = runs / "circuits" / task / f"seed{seed}" / "cellset.json"
    if cs.exists():  # aya_circuits.cellset: the per-type set the cells intervene on, same path
        c = json.loads(cs.read_text())
        suf, ctl, nec, nctl = (
            c["cell_sufficiency"],
            c["cell_controls"],
            c["necessity"],
            c["necessity_controls"],
        )
        out["cell_patching"] = suf["zero"]["patching_raw_ratio"]
        out["cell_zero_raw"] = suf["zero"]["ablation_raw_ratio"]
        out["cell_mean_raw"] = suf["mean"]["ablation_raw_ratio"]
        out["cell_random_margin"] = ctl["circuit"]["patching_raw_ratio"] - st.mean(
            x["patching_raw_ratio"] for x in ctl["random"]
        )
        out["cell_ablation_margin"] = ctl["circuit"]["ablation_raw_ratio"] - st.mean(
            x["ablation_raw_ratio"] for x in ctl["random"]
        )
        out["cell_necessity_mean"] = nec["cell"]["mean"]["ablation_raw_ratio"]
        out["pillar_necessity_mean"] = nec["pillar"]["mean"]["ablation_raw_ratio"]
        # + = ablating the circuit (mean replacement) costs more than ablating a random per-type set
        out["cell_necessity_margin"] = (
            st.mean(x["ablation_raw_ratio"] for x in nctl["random_mean"])
            - nctl["circuit_mean"]["ablation_raw_ratio"]
        )
        out["jaccard_pillar_cell"] = c["jaccard_pillar_cell"]
    return out


def joint_cells_section(runs: Path, beh: str, main_rows: list[dict]) -> list[str]:
    """Amendment A15: cells on the joint-ranked set whose pillars are reported (`circuit_joint`)
    against count-matched random / inverse sets and the pre-registered conditions; H1, H2 by the
    pre-registered rules; H3's gate on the joint pillars, which these cells intervene on."""

    def by(interv: str, cond: str) -> list[dict]:
        return [r for r in main_rows if r["intervention"] == interv and r["condition"] == cond]

    L = []
    for interv in ("steer", "lora", "edit"):
        c = by(interv, "circuit_joint")
        if not c:
            continue
        if not L:
            L.append("\n## A15: cells on the joint-ranked (faithful) circuit\n")
            L.append(
                "`circuit_joint` = CircuitKIT's pillar set (`cellset.json` `pillar_set`, the "
                "set whose patching / ablation / necessity §4.1 reports), `random_joint` = 5 sets "
                "with the same head and MLP counts, `inverse_joint` = bottom per-type counts by "
                "the same absolute scores (shares the circuit's 6–8 lowest-ranked MLPs). "
                "Same decision rules as above; `l4_pillar_*` = CAD on the pillar-set mask from the "
                "kept edges (audit/l4_pillarmask.py).\n"
            )
        L.append(f"\n### {interv}\n")
        L.append(
            "| cell | efficacy | |Δθ| | adapter entries / trained | overlap with circuit_joint "
            "(joint conditions) or circuit (nodes) | L3 mean abs Δ | bystander CAD excess "
            "(per-type / joint / pillar mask) | target CAD excess (pillar mask) | "
            "neighbourhood Δ en | WikiText ratio |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        rows_ = [
            r
            for cond in ("circuit_joint", "random_joint", "inverse_joint", "circuit", "random")
            for r in by(interv, cond)
        ]
        L += [
            f"| {r['cell']} | {fmt(r['efficacy'])} | {fmt(r['delta_norm'], 2)} | "
            f"{r.get('n_trainable') or '–'} / {r.get('n_trained') or '–'} | "
            f"{fmt(r.get('overlap_circuit_nodes'))} | "
            f"{fmt(r['l3_mean_abs_delta'])} | {fmt(r['l4_bystander_cad_excess_mean'])} / "
            f"{fmt(r.get('l4_bystander_cad_excess_mean_joint'))} / "
            f"{fmt(r.get('l4_pillar_bystander_cad_excess_mean'))} | "
            f"{fmt(r.get('l4_pillar_target_cad_excess'))} | {fmt(r['l1_neigh_en_delta'])} | "
            f"{fmt(r['l1_wikitext_ppl_ratio'])} |"
            for r in rows_
        ]
        u = by(interv, "unrestricted")
        if t := h1_text(c, u, "efficacy", "H1 (A15)", "circuit_joint"):
            L.append(t)
        for metric in [*COLLATERAL, "l4_pillar_bystander_cad_excess_mean", "l4_pillar_target_cad"]:
            for name in ("random_joint", "inverse_joint", "circuit", "random", "unrestricted"):
                other = by(interv, name)
                if not other:
                    continue
                pairs = (
                    pairs_for(c, other, metric)
                    if metric in COLLATERAL
                    else [
                        (o[metric], r[metric], r["disc_seed"])
                        for r in c
                        for o in other
                        if o["disc_seed"] == r["disc_seed"]
                        and r.get(metric) is not None
                        and o.get(metric) is not None
                    ]
                )
                if len(pairs) < 2:
                    continue
                h2 = name in ("random_joint", "inverse_joint") and metric in (
                    "l4_bystander_cad_mean",
                    "l3_mean_abs_delta",
                )
                reg = f"{interv} {metric} circuit_joint − {name} (A15)" if h2 else None
                L.append(
                    f"- {'H2' if h2 else 'ctrl'} {metric}: circuit_joint − {name}: "
                    f"{test(pairs, reg)}"
                )
        # H3 on these cells: the joint pillars are what they intervene on
        for r in c:
            p = pillars(runs, f"{beh}_en", r["disc_seed"])
            rnd = [o for o in by(interv, "random_joint") if o["disc_seed"] == r["disc_seed"]]
            adv = {
                m: (st.mean(o[m] for o in rnd if o[m] is not None) - r[m])
                if rnd and r[m] is not None and any(o[m] is not None for o in rnd)
                else None
                for m in ("l4_bystander_cad_mean", "l3_mean_abs_delta")
            }
            gate = (
                (p["zero_raw"] is not None and p["zero_raw"] <= 0)
                or (p["mean_raw"] is not None and p["mean_raw"] <= 0)
                or (p.get("random_margin") is not None and p["random_margin"] <= 0)
                or (p["patching"] is not None and p["patching"] < 0.1)
            )
            L.append(
                f"- H3 {r['cell']}: joint pillars patching {fmt(p['patching'])}, zero "
                f"{fmt(p['zero_raw'])}, mean {fmt(p['mean_raw'])}, random margin "
                f"{fmt(p.get('random_margin'))}, necessity {fmt(p.get('pillar_necessity_mean'))} → "
                f"gate {'fails' if gate else 'passes'}; advantage over random_joint: L4 "
                f"{fmt(adv['l4_bystander_cad_mean'])}, L3 {fmt(adv['l3_mean_abs_delta'])} "
                "(+ = circuit_joint has less collateral)"
            )
    return L


def shared_edges_section(runs: Path, beh: str, main_rows: list[dict]) -> list[str]:
    """Amendment A17 (exploratory): within one bystander circuit, do the edges shared with a node
    set X move more than its other edges? D = excess-CAD difference (magnitude-confounded, see the
    header line); Λ = log ratio of CAD to its own resample floor per region (noise units)."""
    per: dict[tuple, list] = {}
    for r in main_rows:
        p = runs / "audit" / beh / r["cell"] / "l4_shared.json"
        if not p.exists():
            continue
        for v in (v for v in json.loads(p.read_text()).values() if isinstance(v, dict)):
            for x in ("cell", "target"):
                sh, un = v.get(x, {}).get("shared", {}), v.get(x, {}).get("unshared", {})
                if any(q.get(k, 0) <= 0 for q in (sh, un) for k in ("cad", "resample")):
                    continue
                d = (sh["cad"] - sh["resample"]) - (un["cad"] - un["resample"])
                lam = math.log(sh["cad"] / sh["resample"]) - math.log(un["cad"] / un["resample"])
                per.setdefault((r["intervention"], r["condition"], x), []).append(
                    (d, lam, sh["n_edges"], r["disc_seed"])
                )
    if not per:
        return []

    def c(xs: list[float]) -> str:
        lo, hi, _ = ci([(0.0, v) for v in xs])
        return f"{st.mean(xs):+.3f} [{lo:+.3f}, {hi:+.3f}]"

    L = [
        "\n## A17: within-bystander overlap test (exploratory)\n",
        "Each bystander's induced edge set is split into the edges shared with the cell's "
        "intervened components (`S`) or with the pre-registered target set (`T`) and the rest; "
        "both parts are "
        "scored from the kept edges with their own resample floors (`audit/l4_shared.py`). D = "
        "(CAD − floor) shared − (CAD − floor) unshared is confounded by the base magnitude of the "
        "edges in each part (CAD divides by Σ|s_e|): its sign follows the region, not the "
        "intervention, and the perturbation controls reproduce it. Λ = log(CAD/floor) shared − "
        "log(CAD/floor) unshared cancels that denominator and compares the two regions in units "
        "of their own resampling noise; + = the shared edges moved more. n = cell × bystander "
        "pairs; – = fewer than 10 of the bystander's edges touch the intervened components (the "
        "component is not in that bystander's node set).\n",
        "| intervention | condition | n | shared edges (S) | D by S | Λ by S | Λ > 0 | D by T |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for interv, cond in sorted({k[:2] for k in per}):
        s, t = per.get((interv, cond, "cell")), per.get((interv, cond, "target"), [])
        if not s and not t:
            continue
        lam = [v[1] for v in s] if s else []
        L.append(
            f"| {interv} | {cond} | {len(s or t)} | {st.mean(v[2] for v in s):.0f} | "
            f"{c([v[0] for v in s])} | {c(lam)} | {sum(v > 0 for v in lam) / len(lam):.2f} | "
            f"{c([v[0] for v in t]) if t else '–'} |"
            if s
            else f"| {interv} | {cond} | {len(t)} | – | – | – | – | {c([v[0] for v in t])} |"
        )
    pooled = [
        v[1]
        for (_, cond, x), vs in per.items()
        if x == "cell" and not cond.startswith("perturb")
        for v in vs
    ]
    if pooled:
        L.append(f"\nPooled Λ by S over all non-control cells: {c(pooled)} (n={len(pooled)}).")
    return L


def main() -> None:
    cfg = load_config()
    beh = cfg["grid"]["behaviour"]
    runs = Path(cfg["out_dir"])
    rows, base = load_rows(cfg, beh)
    out_dir = ROOT / "docs/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    # union over rows in first-seen order: stage columns (ripple, relearn, arbitration, edge L1)
    # exist only for the cells those stages cover
    fields = list(dict.fromkeys(k for r in rows for k in r if not k.startswith("_")))
    with (out_dir / f"{beh}_cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # one scorecard per cell next to its artifact (brief §3.3 item 5; published with the cells)
    for r in rows:
        _, interv, cond_seed = r["cell"].split("_", 2)
        cell_dir = runs / "interventions" / beh / interv / cond_seed
        if cell_dir.exists():
            hf = json.loads((runs / "audit" / beh / r["cell"] / "hf.json").read_text())
            card = {k: r.get(k) for k in fields} | {"auditkit_compare": auditkit_compare(base, hf)}
            (cell_dir / "scorecard.json").write_text(json.dumps(card, indent=1, default=str))

    L = [
        f"# Hypothesis tests for {beh} ({len(rows)} audited cells)\n",
        "Δ = candidate − baseline (AuditKIT `paired_bootstrap`) with a percentile-bootstrap "
        "95 % CI of the paired mean difference (2000 resamples, seed 0). Main cells only unless "
        "stated; "
        "sweep cells (`cond@mult`) feed the matched-efficacy section.\n",
        f"L4 bystander circuits: {', '.join(BYSTANDERS[beh])}"
        + (
            " (`ioi_fixed` replaces the mis-specified built-in `ioi`, amendment A16).\n"
            if any(b == "ioi_fixed" for b in BYSTANDERS[beh])
            else ".\n"
        ),
    ]
    main_rows = [r for r in rows if r["condition"] and "@" not in r["condition"]]

    def by(interv: str, cond: str) -> list[dict]:
        return [r for r in main_rows if r["intervention"] == interv and r["condition"] == cond]

    if beh == "b2":
        b_en = head(base, "refusal_harmful_en", "refusal")
        L.append(
            f"Base English held-out harmful refusal is {b_en:.3f}, so the pre-registered English "
            f"efficacy is capped at {1 - b_en:+.3f}; `efficacy_xl` (amendment A3, post hoc) is the "
            "mean held-out harmful refusal gain over hi/ar/fr/sw, where the base refuses "
            + ", ".join(
                f"{lang} {head(base, f'refusal_harmful_{lang}', 'refusal'):.2f}"
                for lang in OTHER_LANGS
            )
            + "."
        )
    # Summary table
    L.append("\n## Cell means by intervention × condition\n")
    L.append(
        "| intervention | condition | n | efficacy | efficacy_xl (B2, A3) | |Δθ| | "
        "L4 bystander CAD | L4 bystander CAD − floor | L4 target CAD | L4 target CAD − floor | "
        "L3 mean |Δ| | "
        "L2 benign refusal Δ (en) | L1 neigh Δ (en) | L1 MMLU Δ (en) | WikiText ppl ratio | "
        "L5 AURC Δ | L5 never-edited top-1 error Δ |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    conds = sorted({r["condition"] for r in main_rows}, key=lambda c: (c.startswith("perturb"), c))
    for interv in ("steer", "lora", "edit"):
        for cond in conds:
            rs = by(interv, cond)
            if not rs:
                continue
            cols = (
                "efficacy",
                "efficacy_xl",
                "delta_norm",
                "l4_bystander_cad_mean",
                "l4_bystander_cad_excess_mean",
                "l4_target_cad",
                "l4_target_cad_excess",
                "l3_mean_abs_delta",
                "l2_benign_refusal_delta_en",
                "l1_neigh_en_delta",
                "l1_mmlu_en_delta",
                "l1_wikitext_ppl_ratio",
                "l5_aurc_delta",
                "l5_top1_error_delta",
            )
            L.append(
                f"| {interv} | {cond} | {len(rs)} | "
                + " | ".join(mean_of(rs, k) for k in cols)
                + " |"
            )

    L.append(
        f"\nL4 columns use the **{main_rows[0]['l4_definition'] if main_rows else 'joint'}** "
        "circuit definition"
        + (
            " (pre-registered per-type set, `audit/l4_cellset.py`; the joint-ranked numbers first "
            "reported are the `*_joint` fields of the CSV and the comparison below)."
            if main_rows and main_rows[0]["l4_definition"] == "cell"
            else " (joint-ranked top 61 of 612, as first reported; the per-type recomputation is "
            "not complete for every cell)."
        )
    )
    if main_rows and main_rows[0]["l4_definition"] == "cell":
        L.append("\n## Circuit definition sensitivity (per-type cell set vs joint-ranked set)\n")
        L.append(
            "| intervention | condition | n | bystander CAD − floor (cell / joint) | "
            "target CAD − floor (cell / joint) | AS-NC (cell / joint) | "
            "circuit distance (cell / joint) |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for interv in ("steer", "lora", "edit"):
            for cond in conds:
                rs = by(interv, cond)
                if not rs:
                    continue

                def pair(k, rs=rs):
                    return f"{mean_of(rs, k)} / {mean_of(rs, k + '_joint')}"

                L.append(
                    f"| {interv} | {cond} | {len(rs)} | {pair('l4_bystander_cad_excess_mean')} | "
                    f"{pair('l4_target_cad_excess')} | {pair('l4_as_nc')} | "
                    f"{pair('l4_target_circuit_distance')} |"
                )
        xy = [
            (r["l4_bystander_cad_excess_mean"], r["l4_bystander_cad_excess_mean_joint"])
            for r in main_rows
            if r.get("l4_bystander_cad_excess_mean") is not None
            and r.get("l4_bystander_cad_excess_mean_joint") is not None
        ]
        if len(xy) > 3:
            rho = spearmanr([a for a, _ in xy], [b for _, b in xy])
            repro = max(
                (r["l4_repro_max_abs_diff"] or 0)
                for r in main_rows
                if r.get("l4_repro_max_abs_diff") is not None
            )
            L.append(
                "\nSpearman(cell, joint) of the per-cell bystander CAD excess = "
                f"{rho.statistic:.3f} "
                f"(n={len(xy)}); the recomputation reproduces the stored joint-set CAD to "
                f"{repro:.1e} (max over cells and circuits)."
            )
    for interv in ("steer", "lora", "edit"):
        c = by(interv, "circuit")
        if not c:
            continue
        L.append(f"\n## {interv}\n")
        u = by(interv, "unrestricted")

        def h1(key: str, label: str, c: list[dict] = c, u: list[dict] = u) -> None:
            if t := h1_text(c, u, key, label):
                L.append(t)

        h1("efficacy", "H1")
        if beh == "b2":
            h1("efficacy_xl", "H1-xl (A3, mean hi/ar/fr/sw held-out harmful refusal gain)")
            for name in COMPARISONS:
                other = by(interv, name)
                pairs = [
                    (o[f"h5_gain_{lang}"], r[f"h5_gain_{lang}"], r["disc_seed"])
                    for r in c
                    for o in other
                    if o["disc_seed"] == r["disc_seed"]
                    for lang in OTHER_LANGS
                    if r.get(f"h5_gain_{lang}") is not None and o.get(f"h5_gain_{lang}") is not None
                ]
                if pairs:
                    L.append(f"- A3 efficacy_xl: circuit − {name} (seed × language): {test(pairs)}")
        for metric in COLLATERAL:
            for name in COMPARISONS:
                other = by(interv, name)
                if not other:
                    continue
                pairs = pairs_for(c, other, metric)
                if len(pairs) >= 2:
                    tag = (
                        "H2"
                        if name in ("random", "inverse")
                        and metric in ("l4_bystander_cad_mean", "l3_mean_abs_delta")
                        else "ctrl"
                    )
                    reg = f"{interv} {metric} circuit − {name}" if tag == "H2" else None
                    L.append(f"- {tag} {metric}: circuit − {name}: {test(pairs, reg)}")
                    if metric == "l4_bystander_cad_mean" and c[0]["l4_definition"] == "cell":
                        pj = [
                            (
                                o["_detail"]["cads_joint"][b],
                                r["_detail"]["cads_joint"][b],
                                r["disc_seed"],
                            )
                            for r in c
                            for o in other
                            if o["disc_seed"] == r["disc_seed"]
                            for b in r["_detail"]["cads_joint"]
                            if b in o["_detail"]["cads_joint"]
                        ]
                        if len(pj) >= 2:
                            L.append(
                                "  - same test on the joint-ranked definition (as first "
                                "reported): "
                                f"{test(pj)}"
                            )

    L += joint_cells_section(runs, beh, main_rows)
    L += shared_edges_section(runs, beh, main_rows)

    # H3: faithfulness pillars vs H2 advantage (circuit − random collateral, sign flipped so + =
    # advantage)
    L.append("\n## H3 faithfulness gates locality\n")
    pts = []
    for interv in ("steer", "lora", "edit"):
        for r in by(interv, "circuit"):
            rnd = [o for o in by(interv, "random") if o["disc_seed"] == r["disc_seed"]]
            if not rnd:
                continue
            p = pillars(runs, f"{beh}_en", r["disc_seed"])
            adv = {
                m: (st.mean(o[m] for o in rnd if o[m] is not None) - r[m])
                if r[m] is not None and any(o[m] is not None for o in rnd)
                else None
                for m in ("l4_bystander_cad_mean", "l3_mean_abs_delta")
            }
            gate = (
                (p["zero_raw"] is not None and p["zero_raw"] <= 0)
                or (p["mean_raw"] is not None and p["mean_raw"] <= 0)
                or (p.get("random_margin") is not None and p["random_margin"] <= 0)
            )
            gate_cell = (
                None
                if p.get("cell_patching") is None
                else (
                    p["cell_zero_raw"] <= 0
                    or p["cell_mean_raw"] <= 0
                    or p["cell_random_margin"] <= 0
                    or p["cell_patching"] < 0.1  # restores < 10 % of the logit difference
                )
            )
            pts.append(
                {
                    "cell": r["cell"],
                    **p,
                    **{f"adv_{k}": v for k, v in adv.items()},
                    "gate_fails": gate,
                    "gate_fails_cell": gate_cell,
                }
            )
    if pts:
        L.append(
            "Pillar columns without a prefix are CircuitKIT's joint-ranked set (as first "
            "reported); "
            "`cell *` columns are the pre-registered per-type set the cells intervene on "
            "(`aya_circuits.cellset`, same evaluation path); `necessity` = ablation raw ratio with "
            "the "
            "set ablated and the rest kept (mean replacement; 1 = no loss), its margin = random "
            "per-type sets − circuit (+ = the circuit is more necessary than chance).\n"
        )
        L.append(
            "| cell | patching | zero raw | mean raw | stability | random margin (patching) | "
            "ablation margin (zero) | cell patching | cell zero raw | cell mean raw | "
            "cell random margin | cell necessity | cell necessity margin | adv L4 | adv L3 | "
            "P2/P5 fail | cell gate fail |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        L += [
            f"| {p['cell']} | {fmt(p['patching'])} | {fmt(p['zero_raw'])} | {fmt(p['mean_raw'])} | "
            f"{fmt(p['stability'])} | {fmt(p.get('random_margin'))} | "
            f"{fmt(p.get('ablation_margin'))} | {fmt(p.get('cell_patching'))} | "
            f"{fmt(p.get('cell_zero_raw'))} | {fmt(p.get('cell_mean_raw'))} | "
            f"{fmt(p.get('cell_random_margin'))} | {fmt(p.get('cell_necessity_mean'))} | "
            f"{fmt(p.get('cell_necessity_margin'))} | {fmt(p['adv_l4_bystander_cad_mean'])} | "
            f"{fmt(p['adv_l3_mean_abs_delta'])} | {p['gate_fails']} | {p.get('gate_fails_cell')} |"
            for p in pts
        ]
        L.append(
            "The zero-ablation raw ratio has a floor of 0.5 on a symmetric logit-difference "
            "metric (ablating 90 % of the nodes drives the metric to 0); `ablation margin` = "
            "circuit − mean random-node set under the same ablation is the informative quantity."
        )
        for pil in PILLAR_KEYS:
            for adv in ("adv_l4_bystander_cad_mean", "adv_l3_mean_abs_delta"):
                xy = [
                    (p[pil], p[adv])
                    for p in pts
                    if p.get(pil) is not None and p.get(adv) is not None
                ]
                if len(xy) >= 4 and len({x for x, _ in xy}) > 1:
                    rho = spearmanr([x for x, _ in xy], [y for _, y in xy])
                    L.append(
                        f"- Spearman({pil}, {adv}) = {rho.statistic:.3f} (p={rho.pvalue:.3g}, "
                        f"n={len(xy)}; only {len({x for x, _ in xy})} distinct pillar values "
                        "within one behaviour)"
                    )
        for key, label in (("gate_fails", "joint set"), ("gate_fails_cell", "cell set")):
            gated = [p for p in pts if p.get(key)]
            if gated:
                ok = [p for p in gated if (p["adv_l4_bystander_cad_mean"] or 0) <= 0]
                L.append(
                    f"- Gate prediction, {label} (failing circuits show no L4 advantage): "
                    f"{len(ok)}/{len(gated)} gated cells consistent"
                )
        (out_dir / f"h3_points_{beh}.json").write_text(json.dumps(pts, indent=1))
        other = out_dir / f"h3_points_{'b2' if beh == 'b1' else 'b1'}.json"
        if other.exists():
            pool = pts + json.loads(other.read_text())
            L.append(
                f"\nPooled over both behaviours ({len(pool)} circuit cells, 6 distinct circuits):"
            )
            for pil in PILLAR_KEYS:
                for adv in ("adv_l4_bystander_cad_mean", "adv_l3_mean_abs_delta"):
                    xy = [
                        (q[pil], q[adv])
                        for q in pool
                        if q.get(pil) is not None and q.get(adv) is not None
                    ]
                    if len(xy) >= 4 and len({x for x, _ in xy}) > 1:
                        rho = spearmanr([x for x, _ in xy], [y for _, y in xy])
                        L.append(
                            f"- pooled Spearman({pil}, {adv}) = {rho.statistic:.3f} "
                            f"(p={rho.pvalue:.3g}, n={len(xy)})"
                        )
            for key, label in (("gate_fails", "joint set"), ("gate_fails_cell", "cell set")):
                gated = [q for q in pool if q.get(key)]
                if gated:
                    ok_l4 = [q for q in gated if (q["adv_l4_bystander_cad_mean"] or 0) <= 0]
                    ok = [q for q in ok_l4 if (q["adv_l3_mean_abs_delta"] or 0) <= 0]
                    L.append(
                        f"- pooled gate prediction, {label} (failing circuits show no L4 "
                        f"advantage): {len(ok_l4)}/{len(gated)} gated cells consistent; no L4 and "
                        f"no L3 advantage: {len(ok)}/{len(gated)}"
                    )

    # Exploratory: does the target circuit's own edge change (CAD above the resample floor) track
    # efficacy across cells?
    key = "efficacy_xl" if beh == "b2" else "efficacy"
    pts = [
        (r[key], r["l4_target_cad_excess"])
        for r in main_rows
        if not r["condition"].startswith("perturb")
        and r.get(key) is not None
        and r.get("l4_target_cad_excess") is not None
    ]
    if len(pts) > 3:
        rho = spearmanr([p[0] for p in pts], [p[1] for p in pts])
        per = []
        for interv in ("steer", "lora", "edit"):
            q = [
                (r[key], r["l4_target_cad_excess"])
                for r in main_rows
                if r["intervention"] == interv
                and not r["condition"].startswith("perturb")
                and r.get(key) is not None
                and r.get("l4_target_cad_excess") is not None
            ]
            if len(q) > 3:
                rr = spearmanr([x for x, _ in q], [y for _, y in q])
                per.append(f"{interv} {rr.statistic:.2f} (p={rr.pvalue:.2g}, n={len(q)})")
        L.append(
            "\n## Exploratory: target-circuit edge change vs efficacy\n\n"
            f"Spearman({key}, target CAD − floor) over the main cells without perturbation "
            "controls = "
            f"{rho.statistic:.3f} (p={rho.pvalue:.2g}, n={len(pts)}); within intervention: "
            + ", ".join(per)
            + "."
        )

    # H4: overlap vs bystander CAD (and behavioural change where a bystander has a behavioural
    # metric)
    try:
        nodes = {
            c: json.loads((runs / "circuits" / c / "seed0" / "nodes.json").read_text())
            for c in [f"{beh}_en"] + BYSTANDERS[beh]
        }
        sig = {c: dict(zip(n["names"], n["signed_rowsum"], strict=True)) for c, n in nodes.items()}
        for d in sig.values():  # overlap.py drops the input pseudo-node; H4 uses the same measure
            d.pop("input", None)
        ov = {b: weighted_jaccard(sig[f"{beh}_en"], sig[b]) for b in BYSTANDERS[beh]}
        xs, ys, ye = [], [], []
        for r in main_rows:
            for b, cad in r["_detail"]["cads"].items():
                xs.append(ov[b])
                ys.append(cad)
                ye.append(r["_detail"]["cad_excess"][b])
        if len(xs) > 3:
            rho, rho_e = spearmanr(xs, ys), spearmanr(xs, ye)
            present = [
                b for b in BYSTANDERS[beh] if any(b in r["_detail"]["cads"] for r in main_rows)
            ]
            per_b = {
                b: fmt(
                    st.mean(r["_detail"]["cads"][b] for r in main_rows if b in r["_detail"]["cads"])
                )
                for b in present
            }
            per_e = {
                b: fmt(
                    st.mean(
                        r["_detail"]["cad_excess"][b]
                        for r in main_rows
                        if b in r["_detail"]["cad_excess"]
                    )
                )
                for b in present
            }
            # the cell × bystander points share one overlap value per bystander, so the Spearman is
            # pseudo-replicated;
            # the honest test is the rank agreement over the (three) bystander means, which has no
            # CI
            rank_ok = sorted(present, key=lambda b: ov[b]) == sorted(
                present, key=lambda b: float(per_e[b])
            )
            L.append(
                "\n## H4 overlap predicts interference\n\n"
                f"Weighted-Jaccard overlap with {beh}_en: "
                f"{json.dumps({k: round(v, 3) for k, v in ov.items()})}; "
                f"mean bystander CAD: {json.dumps(per_b)}; "
                f"mean bystander CAD − resample floor: {json.dumps(per_e)}; "
                f"Spearman(overlap, CAD) = {rho.statistic:.3f} (p={rho.pvalue:.3g}), "
                f"Spearman(overlap, CAD − floor) = {rho_e.statistic:.3f} (p={rho_e.pvalue:.3g}); "
                f"n={len(xs)} cell × bystander points but only {len(present)} distinct overlaps, "
                "so the p-values are pseudo-replicated; bystander means ranked by overlap "
                f"{'agree' if rank_ok else 'do not agree'} with their rank by floor-adjusted CAD."
            )
            # Pre-registered statistic without the pseudo-replication: Spearman over the bystanders
            # inside each cell, averaged over cells; CIs resample cells, then whole discovery seeds.
            # Overlap is still a property of the bystander, so with three bystanders this cannot
            # separate overlap from any other bystander difference (floor, edge-score noise).
            for name, key in (("CAD", "cads"), ("CAD − floor", "cad_excess")):
                per_cell = [
                    (
                        spearmanr(
                            [ov[b] for b in present], [r["_detail"][key][b] for b in present]
                        )[0],
                        r["disc_seed"],
                    )
                    for r in main_rows
                    if all(b in r["_detail"][key] for b in present)
                ]
                vals = [v for v, _ in per_cell]
                lo, hi = ci_np(vals)
                clo, chi, k = cluster_ci([(0.0, v, s) for v, s in per_cell])
                L.append(
                    f"- per-cell Spearman(overlap, {name}) over the {len(present)} bystanders: "
                    f"mean {st.mean(vals):+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(vals)} cells; "
                    f"seed-cluster CI [{clo:+.3f}, {chi:+.3f}] ({k} seeds); full rank agreement "
                    f"in {sum(v == 1 for v in vals)}/{len(vals)} cells"
                )
    except FileNotFoundError:
        pass

    # H5: cross-lingual propagation of edits (B1 only): circuit vs taylor on edited-fact gain and
    # neighbourhood change in hi/ar/fr/sw
    # B1: H5 as pre-registered; B2: the same table with held-out harmful refusal gain per language
    # (propagation, not collateral)
    L.append(
        "\n## H5 cross-lingual propagation"
        + (
            " (B1: edited-fact accuracy gain)"
            if beh == "b1"
            else " (B2: held-out harmful refusal gain; descriptive, H5 is pre-registered "
            "for B1 only)"
        )
        + "\n"
    )
    L.append(
        "| intervention | condition | "
        + " | ".join(f"gain {lang}" for lang in OTHER_LANGS)
        + " | "
        + " | ".join(f"neigh Δ {lang}" for lang in OTHER_LANGS)
        + " |"
    )
    L.append("|---|---|" + "---|" * (2 * len(OTHER_LANGS)))
    for interv in ("lora", "edit", "steer"):
        for cond in (
            "circuit",
            "circuit_deep",
            "taylor",
            "causal_trace",
            "unrestricted",
            "random",
            "inverse",
            "safetune_rd",
        ):
            rs = by(interv, cond)
            if not rs:
                continue
            cells_ = [
                mean_of(rs, k)
                for k in [f"h5_gain_{lang}" for lang in OTHER_LANGS]
                + [f"h5_neigh_{lang}" for lang in OTHER_LANGS]
            ]
            L.append(f"| {interv} | {cond} | " + " | ".join(cells_) + " |")
    L.append("")
    for interv, site in (
        ("lora", "taylor"),
        ("edit", "taylor"),
        ("steer", "taylor"),
        ("edit", "causal_trace"),  # the pre-registered site (A8)
    ):
        c, t = by(interv, "circuit"), by(interv, site)
        if c and t:
            gp = [
                (o[f"h5_gain_{lang}"], r[f"h5_gain_{lang}"], r["disc_seed"])
                for r in c
                for o in t
                if o["disc_seed"] == r["disc_seed"]
                for lang in OTHER_LANGS
                if r[f"h5_gain_{lang}"] is not None and o[f"h5_gain_{lang}"] is not None
            ]
            npairs = [
                (o[f"h5_neigh_{lang}"], r[f"h5_neigh_{lang}"], r["disc_seed"])
                for r in c
                for o in t
                if o["disc_seed"] == r["disc_seed"]
                for lang in OTHER_LANGS
                if r[f"h5_neigh_{lang}"] is not None and o[f"h5_neigh_{lang}"] is not None
            ]
            if len(gp) >= 2:
                L.append(
                    f"- {interv}: edited-fact gain, circuit − {site} (seed × language): {test(gp)}"
                )
            if len(npairs) >= 2:
                L.append(
                    f"- {interv}: neighbourhood accuracy change, circuit − {site}: "
                    f"{test(npairs)} (H5 needs ≥ 0 here, i.e. no larger drop)"
                )

    for key, title, cols in (
        (
            "l1_ripple_gain",
            "L1 one-hop ripple (B1; audit/ripple.py, data/b1/ripple_en.jsonl): accuracy gain on "
            "questions whose answer follows from the corrected fact",
            ("efficacy", "l1_ripple_gain"),
        ),
        (
            "l5_arb_spread_ratio",
            "L5 evidence arbitration (audit/arbitration.py; Anand 2026): spread ratio sd(A) "
            "cell/base, AURC and most-confident-quarter error changes, mean drift of A",
            (
                "l5_arb_spread_ratio",
                "l5_arb_aurc_delta",
                "l5_arb_high_conf_error_delta",
                "l5_arb_drift_mean",
            ),
        ),
        (
            "l4_target_edge_l1",
            "L4 edge-level target-circuit distance (Σ|Δs|/Σ|s| over every edge of the target "
            "task graph; Chen et al.'s Circuit Distance analogue)",
            ("efficacy", "l4_target_edge_l1", "l4_target_circuit_distance"),
        ),
        (
            "l5_relearn_gain_xl",
            "L5 relearning probe (B2; audit/relearn.py): held-out harmful refusal gain over the "
            "base model after the same 40-step benign fine-tune, next to the gain before it",
            (
                "efficacy_xl",
                "l5_relearn_gain_xl",
                "l5_relearn_retention_xl",
                "l5_relearn_gain_en",
                "l5_relearn_benign_en_delta",
            ),
        ),
    ):
        if not any(r.get(key) is not None for r in main_rows):
            continue
        L.append(f"\n## {title}\n")
        L.append("| intervention | condition | n | " + " | ".join(cols) + " |")
        L.append("|---|---|---|" + "---|" * len(cols))
        for interv in ("steer", "lora", "edit"):
            for cond in conds:
                rs = [r for r in by(interv, cond) if r.get(key) is not None]
                if rs:
                    L.append(
                        f"| {interv} | {cond} | {len(rs)} | "
                        + " | ".join(
                            mean_of([{k: r.get(k) for k in cols} for r in rs], k) for k in cols
                        )
                        + " |"
                    )
        for interv in ("steer", "lora", "edit"):
            c = [r for r in by(interv, "circuit") if r.get(key) is not None]
            for name in COMPARISONS:
                other = [r for r in by(interv, name) if r.get(key) is not None]
                pairs = [
                    (o[key], r[key], r["disc_seed"])
                    for r in c
                    for o in other
                    if o["disc_seed"] == r["disc_seed"]
                ]
                if len(pairs) >= 2:
                    L.append(f"- {interv} {key}: circuit − {name}: {test(pairs)}")

    L += judge_robustness(runs, beh, base, main_rows)

    # Matched-efficacy comparison from the steering sweep
    sweep = [
        r for r in rows if r["condition"] and "@" in r["condition"] and r["efficacy"] is not None
    ]
    if sweep:
        L.append("\n## Steering: collateral at matched efficacy (sweep cells)\n")
        curves: dict[tuple[str, int], list[dict]] = {}
        for r in sweep:
            curves.setdefault((r["condition"].split("@")[0], r["seed"]), []).append(r)
        maxes = {}
        for (cond, _), rs in curves.items():
            maxes.setdefault(cond, []).append(max(r["efficacy"] for r in rs))
        target = min(st.mean(v) for v in maxes.values())
        L.append(
            f"Target efficacy e* = {target:.3f} (the smallest condition-mean peak efficacy; "
            "collateral linearly interpolated along each seed's sweep, "
            "cells whose sweep never reaches e* are skipped).\n"
        )
        L.append(
            "| condition | n curves | reach e* | WikiText ppl ratio @e* | L1 neigh Δ (en) @e* | "
            "L2 benign refusal Δ (en) @e* | L1 MMLU Δ (en) @e* |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for cond in sorted(maxes):
            vals = {
                m: []
                for m in (
                    "l1_wikitext_ppl_ratio",
                    "l1_neigh_en_delta",
                    "l2_benign_refusal_delta_en",
                    "l1_mmlu_en_delta",
                )
            }
            n_reach = 0
            for (c2, _), rs0 in curves.items():
                if c2 != cond:
                    continue
                rs = sorted(rs0, key=lambda r: r["efficacy"])
                if rs[-1]["efficacy"] < target:
                    continue
                n_reach += 1
                e = [r["efficacy"] for r in rs]
                for m in vals:
                    vals[m].append(float(np.interp(target, e, [r[m] for r in rs])))
            L.append(
                f"| {cond} | {sum(1 for k in curves if k[0] == cond)} | {n_reach} | "
                + " | ".join(fmt(st.mean(v)) if v else "–" for v in vals.values())
                + " |"
            )
        L.append(
            "\nPer-cell sweep (efficacy → WikiText ratio): "
            + "; ".join(
                f"{cond}_s{seed}: "
                + ", ".join(
                    f"{r['condition'].split('@')[1]}×→{r['efficacy']:.2f}/{r['l1_wikitext_ppl_ratio']:.2f}"
                    for r in sorted(rs, key=lambda r: float(r["condition"].split("@")[1]))
                )
                for (cond, seed), rs in sorted(curves.items())
            )
        )

    if TESTS:
        L += holm_lines(TESTS)
    (out_dir / f"hypotheses_{beh}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
