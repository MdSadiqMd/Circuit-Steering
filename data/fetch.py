"""Fetch what a fresh clone needs before any model stage can run.

  uv run python data/fetch.py

  data/b2/, data/a22/   the built refusal sets. They hold harmful prompt text, so they are
                        published on the Hub (MdSadiqMd/circuit-steering-data) and not in git.
                        Every file is checked against its sha256 in data/MANIFEST.json.
  .cache/raw/           the public benchmark files the audit reads: Global-MMLU, MGSM, IFEval and
                        the WikiText-2 test split, from their own Hub repos.
Files that are already there are left alone, so the command is safe to repeat.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parent.parent
DATA_REPO = "MdSadiqMd/circuit-steering-data"
TEST = "test-00000-of-00001.parquet"
BENCHMARKS = {  # folder under .cache/raw -> (dataset repo, files)
    "Global-MMLU": (
        "CohereLabs/Global-MMLU",
        [f"{k}/{TEST}" for k in ("en", "hi", "ar", "fr", "sw")],
    ),
    "mgsm": ("juletxara/mgsm", [f"{k}/{TEST}" for k in ("en", "fr", "sw")]),
    "IFEval": ("google/IFEval", ["ifeval_input_data.jsonl"]),
    "wikitext": ("Salesforce/wikitext", [f"wikitext-2-raw-v1/{TEST}"]),
}


def main() -> None:
    manifest = json.loads((ROOT / "data/MANIFEST.json").read_text())["files"]
    wanted = [
        f for f in list_repo_files(DATA_REPO, repo_type="dataset") if f.startswith(("b2/", "a22/"))
    ]
    assert wanted, f"no refusal sets found in {DATA_REPO}"
    for name in wanted:
        path = ROOT / "data" / name
        if not path.exists():
            hf_hub_download(DATA_REPO, name, repo_type="dataset", local_dir=ROOT / "data")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest[name]["sha256"], f"{path} differs from data/MANIFEST.json"
    for folder, (repo, files) in BENCHMARKS.items():
        for name in files:
            if not (ROOT / ".cache/raw" / folder / name).exists():
                hf_hub_download(
                    repo, name, repo_type="dataset", local_dir=ROOT / ".cache/raw" / folder
                )
    print(f"fetch ok: {len(wanted)} refusal-set files verified, benchmarks in .cache/raw")


if __name__ == "__main__":
    main()
