"""A20 L4: audit/l4_a19.py (A19's per-example EAP-IG under its sealed rule) on the A20 cells, with
the A20 condition names added to the conditions it attributes; nothing else changes.
Run: uv run python audit/a20_l4.py out_dir=runs/a20 stage=cells grid.behaviour=b1 cells=steer,lora
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit import l4_a19


def main() -> None:
    l4_a19.CONDITIONS |= {"cert", "cert_random", "cert_strat", "cert_inverse"}
    l4_a19.main()


if __name__ == "__main__":
    main()
