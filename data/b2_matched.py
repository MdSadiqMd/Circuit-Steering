"""Refusal EAP pairs restricted to token-length-matched clean/corrupt rows (amendment A16).

`data/b2_eap.py` falls back to the closest-length benign prompt when no exact match exists, and
CircuitKIT end-pads such pairs, so 45 of the 207 English pairs compare clean and corrupt
activations at shifted positions (audit-2026-11-06 C8). This keeps only rows whose two texts have
the same token count, so the matched circuit isolates that defect. Output:
data/b2/eap_<lang>_matched.csv (task `b2m_<lang>`). Run: uv run python data/b2_matched.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent / "b2"


def main() -> None:
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"])

    def n_tok(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    for lang in cfg["languages"]:
        rows = list(csv.DictReader((HERE / f"eap_{lang}.csv").open()))
        kept = [r for r in rows if n_tok(r["clean"]) == n_tok(r["corrupted"])]
        out = HERE / f"eap_{lang}_matched.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"])
            w.writeheader()
            w.writerows(kept)
        print(f"{lang}: kept {len(kept)} of {len(rows)} -> {out.name}")


if __name__ == "__main__":
    main()
