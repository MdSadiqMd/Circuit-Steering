"""Amendment A22: refusal on the held-out test sets for the base model and every A22 cell.

  uv run python audit/a22_eval.py out_dir=runs/a22

Families (data/a22_build.py): harmful_test (159 prompts x 5 languages: 80 held-out prompts of the
trained category, 79 of two categories the adapter never saw), benign_test (59 near-miss benign
prompts x 5 languages), xstest_unsafe (200 English harmful prompts from a source the adapter never
saw). Generation and scoring go through neb.behavioural.eval_refusal (AuditKIT runs, the primary
regex judge of neb/judge.py), 48 new tokens.

Writes <out_dir>/eval/<cell>.json: the primary judge's score per item, and the reply text for
benign prompts only (base.json also carries the benign prompt text). Replies to harmful prompts
stay in .cache/private/a22/<cell>.json (brief §8), where audit/a22_judge.py reads them.
Resumable; exits 3 after `audit.cells_per_process` new cells.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / ".cache/private/a22"


def guard_disk(min_gb: float = 3.0) -> None:
    """Stop before the disk fills: macOS grows swap files on the same volume, and a full disk
    killed the first A22 run inside AuditKIT's result cache (docs/notes.md 2026-11-11)."""
    import shutil

    free = shutil.disk_usage(ROOT).free / 2**30
    if free < min_gb:
        raise SystemExit(f"only {free:.1f} GB free on disk (< {min_gb}); free space and rerun")


def families(which: str) -> dict[str, tuple[list[dict], list[str] | None]]:
    """name -> (prompt rows, languages or None for all). `which` = dev | test."""
    from data.a22_build import BT_MIN, load

    split, rows, benign = load()
    if which == "dev":
        dev = [benign[i] for i in split["benign_dev"]]
        # a flagged translation is not a fair benign probe: drop that language's text
        dev = [
            {k: v for k, v in r.items() if k == "en" or r.get(f"bt_f1_{k}", 1) >= BT_MIN}
            for r in dev
        ]
        return {
            "harmful_dev": ([rows[i] for i in split["harmful_dev"]], None),
            "benign_dev": (dev, None),
        }
    xs = {r["id"]: r for r in map(json.loads, (ROOT / "data/b2/xstest_all.jsonl").open())}
    return {
        "harmful_test": ([rows[i] for i in split["harmful_test"]], None),
        "benign_test": ([rows[i] for i in split["benign_test"]], None),
        "xstest_unsafe": ([xs[i] for i in split["xstest_unsafe"]], ["en"]),
    }


def run_families(model, tok, cfg: dict, fams: dict, max_new: int | None = None) -> dict:
    """{family: {lang: {"scores": {id: 0/1}, "outputs": {id: text}}}} with the primary judge."""
    from neb import behavioural as B

    if max_new:
        cfg = {**cfg, "audit": {**cfg["audit"], "max_new_tokens": max_new}}
    out: dict = {}
    for name, (prompts, langs) in fams.items():
        for lang in langs or cfg["languages"]:
            r = B.eval_refusal(model, tok, cfg, prompts, lang, f"a22_{name}_{lang}")
            out.setdefault(name, {})[lang] = {
                "scores": {p.sample_id: p.score for p in r.predictions},
                "outputs": {p.sample_id: p.raw_output or "" for p in r.predictions},
                "prompts": {f"{p['id']}:{lang}": p[lang] for p in prompts if lang in p},
            }
    return out


def rates(res: dict) -> dict[str, dict[str, float]]:
    return {
        fam: {lang: sum(v["scores"].values()) / len(v["scores"]) for lang, v in by.items()}
        for fam, by in res.items()
    }


def save(res: dict, name: str, out_root: Path, private: Path = PRIVATE) -> None:
    private.mkdir(parents=True, exist_ok=True)
    (private / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False))
    # public copy: judgments for every family; reply text for benign prompts only, and their
    # prompt text once (in base.json)
    extra = ("outputs", "prompts") if name == "base" else ("outputs",)
    public = {
        fam: {
            lang: {"scores": v["scores"]} | ({k: v[k] for k in extra} if "benign" in fam else {})
            for lang, v in by.items()
        }
        for fam, by in res.items()
    }
    (out_root / f"{name}.json").write_text(json.dumps(public, ensure_ascii=False))


def main() -> None:
    import torch
    from aya_circuits.config import load_config
    from aya_circuits.memwatch import MemWatch
    from aya_circuits.model import load_hf

    from interventions.cell import Cell, apply_hf

    cfg = load_config()
    run_root = Path(cfg["out_dir"])
    out_root = run_root / "eval"
    out_root.mkdir(parents=True, exist_ok=True)
    cells = sorted(p.parent for p in (run_root / "interventions/b2").glob("*/*/cell.json"))
    todo = [d for d in cells if not (out_root / f"{Cell.peek(d).name}.json").exists()]
    if (out_root / "base.json").exists() and not todo:
        return
    mw, fams = MemWatch(cfg["memory_budget_gb"]), families("test")
    private = Path(cfg.get("private") or PRIVATE)
    if cfg.get("limit"):  # smoke runs only: the first `limit` prompts of every family
        fams = {k: (v[: int(cfg["limit"])], langs) for k, (v, langs) in fams.items()}
    model, tok = load_hf(cfg)
    if not (out_root / "base.json").exists():
        save(run_families(model, tok, cfg, fams), "base", out_root, private)
        print("base", rates(json.loads((out_root / "base.json").read_text())), flush=True)
    cap = cfg["audit"].get("cells_per_process")
    for n, d in enumerate(todo):
        if cap and n >= cap:
            print(f"stopping after {n} cells (audit.cells_per_process); rerun to continue")
            raise SystemExit(3)
        guard_disk()
        cell, tensors = Cell.load(d)
        undo = apply_hf(model, cell, tensors)
        try:
            res = run_families(model, tok, cfg, fams)
        finally:
            undo()
        save(res, cell.name, out_root, private)
        print(cell.name, rates(res), flush=True)
        mw.check()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
