"""Nothing-Else-Broke audit for every cell of one behaviour (and the base model once).

  uv run python audit/run_audit.py behaviour=b1 [cells=steer,lora,edit] [side=hf|tl|both]
                                   [shard=i/n]

HF side (L1 behavioural, L2 safety, L3 cross-lingual, L5 hidden): apply the cell to the HF model,
run neb.behavioural families, undo. TL side (L4 mechanistic, audit/l4.py): apply to the TL model,
recompute signed edge scores of the target, every bystander circuit and the extra tasks on their
own data, undo. Results per cell: runs/audit/<behaviour>/<cell>/{hf.json, l4*.json,
l4_edges*.safetensors} with AuditKIT RunResults serialised; the base model's runs live in
runs/audit/<behaviour>/base/. Per-cell scorecards (scorecard.json next to each cell artifact) are
written by audit/analyze.py.
Resumable: existing result files are skipped unless force=true. The MPS allocator grows across
cells (the A18 B2 HF pass ran out of memory at its 16th cell), so after `audit.cells_per_process`
new cells the process exits with code 3 and the shell loop re-invokes it, as for run_grid.py.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401
import torch
from aya_circuits.config import load_config
from aya_circuits.manifest import Manifest
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_hf, load_tl
from neb import behavioural as B

from audit import l4 as L4
from interventions.cell import Cell, apply_hf
from interventions.run_grid import b1_folds

ROOT = L4.ROOT


def run_to_dict(r) -> dict:
    return {
        "headline": r.headline,
        "fingerprint": r.fingerprint,
        "n": len(r.predictions),
        "scores": {p.sample_id: p.score for p in r.predictions},
        "outputs": {p.sample_id: (p.raw_output or "")[:200] for p in r.predictions},
        "metadata": getattr(r, "metadata", {}),
    }


def hf_layers(model, tok, cfg: dict, beh: str, cell_meta: dict | None, full: bool = True) -> dict:
    """full=False (steering sweep cells `cond@mult`): English efficacy and core collateral only
    (edited facts,
    neighbourhood, MMLU, WikiText, refusal harmful/benign/XSTest); L3 languages, MGSM, IFEval and
    L5 are skipped."""
    a, langs = cfg["audit"], (cfg["languages"] if full else ["en"])
    out = {}
    facts = {
        lang: [json.loads(line) for line in (ROOT / f"data/b1/prompts_{lang}.jsonl").open()]
        for lang in langs
    }
    false_keys = [
        (f["relation"], f["subject_qid"])
        for f in (json.loads(line) for line in (ROOT / "data/b1/false_facts.jsonl").open())
    ]
    # base model: evaluate every false fact so each cell's fold has paired base scores; B1 cells
    # carry
    # `edited_keys` (edit/lora: the fold they trained on; steer: the same fold, set in main so every
    # intervention's efficacy is measured on the same facts per seed); B2 cells skip the family
    edited = (
        set(map(tuple, cell_meta["edited_keys"]))
        if cell_meta is not None
        else (set(false_keys) if beh == "b1" else set())
    )
    known_en = {
        (r["relation"], r["subject_qid"])
        for r in (json.loads(line) for line in (ROOT / "data/b1/model_eval_en.jsonl").open())
        if r["known"]
    }
    for lang in langs:
        by_key = {(r["relation"], r["subject_qid"]): r for r in facts[lang]}
        if edited:
            # sorted: set order follows the per-process hash seed, so batches (and bf16 padding)
            # differed between processes and near-tie answers could flip (found 2026-11-09)
            rows = [by_key[k] for k in sorted(edited) if k in by_key]
            out[f"facts_edited_{lang}"] = run_to_dict(
                B.eval_facts(model, tok, rows, lang, f"facts_edited_{lang}")
            )
            out[f"facts_edited_paraphrase_{lang}"] = run_to_dict(
                B.eval_facts(
                    model, tok, rows, lang, f"facts_edited_para_{lang}", key="paraphrase_text"
                )
            )
        neigh = [by_key[k] for k in sorted(known_en - set(false_keys)) if k in by_key][
            : a["l1"]["n_neighbourhood"]
        ]
        out[f"facts_neighbourhood_{lang}"] = run_to_dict(
            B.eval_facts(model, tok, neigh, lang, f"facts_neigh_{lang}")
        )
        out[f"facts_paraphrase_{lang}"] = run_to_dict(
            B.eval_facts(
                model,
                tok,
                neigh[: a["l1"]["n_paraphrase"]],
                lang,
                f"facts_para_{lang}",
                key="paraphrase_text",
            )
        )
        out[f"mmlu_{lang}"] = run_to_dict(B.eval_mmlu(model, tok, cfg, lang, a["l1"]["n_mmlu"]))
    if full:
        for lang in ("en", "fr", "sw"):
            out[f"mgsm_{lang}"] = run_to_dict(
                B.eval_mgsm(model, tok, cfg, lang, a["l1"]["n_gsm8k"])
            )
        out["ifeval"] = run_to_dict(B.eval_ifeval(model, tok, cfg, a["l1"]["n_ifeval"]))
    out["wikitext"] = run_to_dict(B.eval_wikitext(model, tok, a["l1"]["n_wikitext"]))
    # L2/L3 refusal: category held-out harmful, other categories, benign near-miss, XSTest safe
    cat = cfg["behaviour"]["b2_category"]
    split = (
        json.loads((Path(cfg["out_dir"]) / "interventions" / "b2_split.json").read_text())
        if (Path(cfg["out_dir"]) / "interventions" / "b2_split.json").exists()
        else None
    )
    pools = {
        p.stem[len("baseline_") :]: [json.loads(line) for line in p.open()]
        for p in (ROOT / "data/b2").glob("baseline_*.jsonl")
    }
    held = {r["id"] for r in pools[cat]} if split is None else set(split["held_out"])
    xs = [json.loads(line) for line in (ROOT / "data/b2/xstest_all.jsonl").open()]
    for lang in langs:
        harm = [r for r in pools[cat] if r["split"] == "harmful" and r["id"] in held][
            : a["l2"]["n_harmful"]
        ]
        ben = [r for r in pools[cat] if r["split"] == "benign"][: a["l2"]["n_benign"]]
        other = [
            r for c, rows in pools.items() if c != cat for r in rows if r["split"] == "harmful"
        ][: a["l2"]["n_harmful"]]
        out[f"refusal_harmful_{lang}"] = run_to_dict(
            B.eval_refusal(model, tok, cfg, harm, lang, f"refusal_harmful_{lang}")
        )
        out[f"refusal_benign_{lang}"] = run_to_dict(
            B.eval_refusal(model, tok, cfg, ben, lang, f"refusal_benign_{lang}")
        )
        if full:
            out[f"refusal_other_{lang}"] = run_to_dict(
                B.eval_refusal(model, tok, cfg, other, lang, f"refusal_other_{lang}")
            )
        if lang == "en":
            out["refusal_xstest_safe"] = run_to_dict(
                B.eval_refusal(
                    model,
                    tok,
                    cfg,
                    [r for r in xs if r["label"] == "safe"][: a["l2"]["n_benign"]],
                    "en",
                    "refusal_xstest_safe",
                )
            )
    # L5: never-edited facts (English), selective prediction
    if full:
        never = [
            r
            for r in facts["en"]
            if (r["relation"], r["subject_qid"]) in known_en
            and (r["relation"], r["subject_qid"]) not in set(false_keys)
        ][: a["l5"]["n_never_edited"]]
        out["hidden"] = B.eval_hidden(model, tok, never)
    return out


def main() -> None:
    cfg = load_config()
    beh = cfg["grid"]["behaviour"]
    side = cfg.get("side", "both")
    want = set(str(cfg.get("cells", "steer,lora,edit")).split(","))
    out_root = Path(cfg["out_dir"]) / "audit" / beh
    cells = sorted(
        (
            p.parent
            for p in (Path(cfg["out_dir"]) / "interventions" / beh).glob("*/*/cell.json")
            if p.parent.parent.name in want
        ),
        key=str,
    )
    # shard=i/n: this process audits cells[i::n] (several processes share one GPU); only shard 0
    # writes the base model's files, the others wait for them
    shard, n_shards = (int(x) for x in str(cfg.get("shard", "0/1")).split("/"))
    cells = cells[shard::n_shards]
    mw = MemWatch(cfg["memory_budget_gb"])
    print(f"{len(cells)} cells for {beh} (shard {shard}/{n_shards})", flush=True)
    cap, built = cfg["audit"].get("cells_per_process"), 0

    def release() -> None:
        nonlocal built
        built += 1
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if cap and built >= cap:
            print(
                f"stopping after {built} cells (audit.cells_per_process); rerun to continue",
                flush=True,
            )
            raise SystemExit(3)

    if side in ("hf", "both"):
        model, tok = load_hf(cfg)
        base_path = out_root / "base" / "hf.json"
        while shard and not base_path.exists():
            time.sleep(20)
        if not base_path.exists():
            base_path.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            base_path.write_text(json.dumps(hf_layers(model, tok, cfg, beh, None), default=str))
            print(f"base hf layers {time.time() - t0:.0f}s", flush=True)
        for d in cells:
            cell = Cell.peek(d)
            res_path = out_root / cell.name / "hf.json"
            if res_path.exists() and not cfg.get("force"):
                continue
            tensors = Cell.load(d)[1]
            res_path.parent.mkdir(parents=True, exist_ok=True)
            mf, t0 = Manifest("audit_hf", cfg, cell.seed), time.time()
            mf.add_input(d / "delta.safetensors")
            for rel in (
                "data/b1/false_facts.jsonl",
                "data/b1/model_eval_en.jsonl",
                *(f"data/b1/prompts_{lang}.jsonl" for lang in cfg["languages"]),
                *(
                    p.relative_to(ROOT).as_posix()
                    for p in (ROOT / "data/b2").glob("baseline_*.jsonl")
                ),
                "data/b2/xstest_all.jsonl",
            ):
                mf.add_input(ROOT / rel)
            if (
                beh == "b1" and "edited_keys" not in cell.meta
            ):  # steer cells: efficacy on the seed's fact fold
                disc_seed = (
                    cell.seed
                    if cell.seed in cfg["seeds"]
                    else cfg["seeds"][cell.seed % len(cfg["seeds"])]
                )
                cell.meta["edited_keys"] = [
                    [f["relation"], f["subject_qid"]] for f in b1_folds(cfg)[disc_seed]
                ]
            elif beh == "b2":
                cell.meta.setdefault("edited_keys", [])
            undo = apply_hf(model, cell, tensors)
            try:
                res = hf_layers(model, tok, cfg, beh, cell.meta, full="@" not in cell.condition)
            finally:
                undo()
            res_path.write_text(json.dumps(res, default=str))
            mf.write(
                out_root / cell.name / "manifest_hf.json",
                memory=mw.summary(),
                wall_s=round(time.time() - t0, 1),
            )
            print(f"{cell.name} hf {time.time() - t0:.0f}s", flush=True)
            mw.check()
            release()
        del model
        gc.collect()
        torch.mps.empty_cache()

    if side in ("tl", "both"):
        model = load_tl(cfg)
        bases: dict[int, L4.Base] = {}
        for d in cells:
            cell = Cell.peek(d)
            if (
                (out_root / cell.name / "l4.json").exists() and not cfg.get("force")
            ) or "@" in cell.condition:  # sweep cells: no L4
                continue
            seed = L4.disc_seed_of(cell, cfg)
            if seed not in bases:
                bases[seed] = L4.Base(model, cfg, beh, seed, out_root)
            L4.attribute_cell(model, cfg, cell, Cell.load(d)[1], bases[seed], out_root / cell.name)
            mw.check()
            release()


if __name__ == "__main__":
    main()
