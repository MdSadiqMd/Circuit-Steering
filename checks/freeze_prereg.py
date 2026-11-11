"""Freeze docs/preregistration.md: fill dataset sha256 hashes and the circuit-discovery settings,
mark the header FROZEN with a timestamp. Refuses to run twice (append amendments by hand
instead)."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.manifest import sha256_file

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "docs/preregistration.md"


def main() -> None:
    s = P.read_text()
    if s.startswith("# Pre-registration (FROZEN"):
        raise SystemExit("already frozen; append an amendment instead")
    langs = ["en", "hi", "ar", "fr", "sw"]
    lines = ["## Datasets (sha256 at freeze)"]
    for rel in (
        ["data/b1/facts_pool.jsonl", "data/b1/false_facts.jsonl"]
        + [f"data/b1/eap_known_{lang}.csv" for lang in langs]
        + [f"data/b1/prompts_{lang}.jsonl" for lang in langs]
        + ["data/b2/baseline_cyber.jsonl", "data/b2/xstest_all.jsonl"]
        + [f"data/b2/eap_{lang}.csv" for lang in langs]
        + [f"data/bystanders/eap_langid_{lang}.csv" for lang in langs]
    ):
        p = ROOT / rel
        if p.exists():
            lines.append(f"- `{rel}`: `{sha256_file(p)}`")
    start, end = (
        s.index("## Datasets (sha256 to be filled at freeze)"),
        s.index("## Analysis code path"),
    )
    s = s[:start] + "\n".join(lines) + "\n\n" + s[end:]
    stamp = dt.datetime.now().astimezone().isoformat(timespec="minutes")
    s = s.replace(
        "# Pre-registration (DRAFT v0 — not yet frozen)", f"# Pre-registration (FROZEN {stamp})", 1
    )
    s = s.replace(
        "Status: **draft**. This file is frozen (header changes to FROZEN with a date and "
        "the dataset hashes are\nfilled in) before the first intervention cell is built. After "
        'freezing, nothing above the\n"Amendments" line is edited; amendments are appended with '
        "dates and reasons (brief §5, §12.7).",
        f"Status: **frozen {stamp}**, before the first intervention cell (git SHA and uv.lock "
        'hash in the cell manifests).\nNothing above the "Amendments" line is edited after this '
        "point; amendments are appended with dates and reasons (brief §5, §12.7).",
    )
    P.write_text(s)
    print("frozen", stamp)


if __name__ == "__main__":
    main()
