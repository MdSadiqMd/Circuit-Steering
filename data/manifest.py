"""sha256, size, row count and producing script of every dataset file under data/ (brief §11,
§12.10).

The data-building scripts ran before run manifests covered them (only filter_model.py writes one),
so this file records the released state after the fact: data/MANIFEST.json. Re-run after any data
change. Run: uv run python data/manifest.py
"""

from __future__ import annotations

import json
import sys
from fnmatch import fnmatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.manifest import git_state, sha256_file

HERE = Path(__file__).resolve().parent
PRODUCER = {  # file-name glob -> script that writes it; the first match wins
    "b1/facts_pool*": "data/facts_wikidata.py",
    "b1/prompts_*": "data/b1_build.py",
    "b1/eap_known_*": "data/filter_model.py",
    "b1/eap_*": "data/b1_build.py",
    "b1/build_stats*": "data/b1_build.py",
    "b1/model_eval*": "data/filter_model.py",
    "b1/false_facts*": "data/filter_model.py",
    "b1/ripple_en*": "data/b1_ripple.py",
    "b2/pool_*": "data/b2_build.py + data/translate.py",
    "b2/xstest_all*": "data/b2_build.py",
    "b2/build_stats*": "data/b2_build.py",
    "b2/baseline_*": "data/b2_baseline.py (sanitized by data/sanitize.py, amendment A7)",
    "b2/eap_*_matched.csv": "data/b2_matched.py",
    "b2/eap_*": "data/b2_eap.py",
    "a22/*": "data/a22_build.py",
    "a23/*": "data/a23_build.py",
    "a24/plain.jsonl": "data/a23_build.py variant=sampled",
    "bystanders/eap_ioi_fixed.csv": "data/ioi_fixed.py",
    "bystanders/*": "data/bystanders.py",
}


def producer(rel: str) -> str | None:
    return next((v for k, v in PRODUCER.items() if fnmatch(rel, k)), None)


def main() -> None:
    files = {}
    for p in sorted(HERE.rglob("*")):
        if p.is_dir() or p.suffix == ".py" or p.name == "MANIFEST.json" or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(HERE).as_posix()
        rows = (
            sum(1 for _ in p.open()) - (p.suffix == ".csv")
            if p.suffix in (".jsonl", ".csv")
            else None
        )
        files[rel] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "rows": rows,
            "producer": producer(rel),
        }
    out = {"git_sha": git_state()[0], "n_files": len(files), "files": files}
    (HERE / "MANIFEST.json").write_text(json.dumps(out, indent=1))
    print(f"{len(files)} files ->", HERE / "MANIFEST.json")


if __name__ == "__main__":
    main()
