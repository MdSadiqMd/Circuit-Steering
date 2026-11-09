"""Keep generated replies to harmful prompts out of the repo and the Hub (brief §8: "store only
prompts, judgments and aggregate scores").

Two places hold such text:
  data/b2/baseline_<cat>.jsonl  `gen.<lang>.preview` (160 characters of the reply) on harmful rows
  runs/audit/**/hf.json          `outputs` of the refusal_harmful_* and refusal_other_* families
The judgments (`refused`, per-sample `scores`), first tokens and every prompt are kept; replies to
benign prompts are kept too (they are the over-refusal evidence and contain nothing harmful).

  uv run python data/sanitize.py baseline   copy the raw baseline files to .cache/private/ (local,
                                            gitignored) and strip harmful previews in place
  uv run python data/sanitize.py scan [paths...]   list every leak under the given paths
`stage_audit` builds the sanitized copy of runs/audit that data/push_hf.py publishes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARMFUL_FAMILIES = ("refusal_harmful_", "refusal_other_")
PRIVATE = ROOT / ".cache/private/b2_baseline_raw"


def strip_audit(hf: dict) -> dict:
    return {
        k: (
            {kk: vv for kk, vv in v.items() if kk != "outputs"}
            if k.startswith(HARMFUL_FAMILIES)
            else v
        )
        for k, v in hf.items()
    }


def strip_baseline_row(r: dict) -> dict:
    if r.get("split") != "harmful":
        return r
    gen = {
        lang: {k: v for k, v in g.items() if k != "preview"} for lang, g in r.get("gen", {}).items()
    }
    return {**r, "gen": gen}


def leaks(path: Path) -> list[str]:
    """Generated text for harmful prompts inside one file (hf.json or a baseline jsonl)."""
    out = []
    if path.name == "hf.json":
        d = json.loads(path.read_text())
        out += [
            f"{path}:{k}"
            for k, v in d.items()
            if k.startswith(HARMFUL_FAMILIES) and v.get("outputs")
        ]
    elif path.name.startswith("baseline_") and path.suffix == ".jsonl":
        for i, line in enumerate(path.open()):
            r = json.loads(line)
            if r.get("split") == "harmful" and any(
                "preview" in g for g in r.get("gen", {}).values()
            ):
                out.append(f"{path}:{i + 1}:{r['id']}")
    return out


def scan(paths: list[Path]) -> list[str]:
    found = []
    for p in paths:
        files = [p] if p.is_file() else [*p.rglob("hf.json"), *p.rglob("baseline_*.jsonl")]
        for f in files:
            found += leaks(f)
    return found


def sanitize_baseline() -> None:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    for p in sorted((ROOT / "data/b2").glob("baseline_*.jsonl")):
        raw = PRIVATE / p.name
        if not raw.exists():
            shutil.copy2(p, raw)  # the pre-registered file, byte for byte, kept local only
        rows = [strip_baseline_row(json.loads(line)) for line in raw.open()]
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(p, "sanitized; raw copy at", raw)


def stage_audit(src: Path, dst: Path) -> Path:
    """Mirror `src` (runs/audit) into `dst`: hf.json sanitized, every other file hard-linked."""
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        t = dst / f.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        if t.exists():
            t.unlink()
        if f.name == "hf.json":
            t.write_text(json.dumps(strip_audit(json.loads(f.read_text())), default=str))
        else:
            os.link(f, t)
    assert not scan([dst]), "sanitized staging still holds harmful-prompt generations"
    return dst


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if cmd == "baseline":
        sanitize_baseline()
    elif cmd == "scan":
        targets = [Path(a) for a in sys.argv[2:]] or [ROOT / "data", ROOT / "docs", ROOT / "space"]
        found = scan(targets)
        print("\n".join(found[:20]), f"\n{len(found)} leak(s)" if found else "no leaks")
        raise SystemExit(1 if found else 0)
    else:
        raise SystemExit(f"unknown command {cmd!r}")
