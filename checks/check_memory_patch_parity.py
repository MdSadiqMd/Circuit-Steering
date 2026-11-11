"""Brief §11: enable CIRCUITKIT_TL_MEMORY_PATCH=1 only after confirming EAP-IG scores match the
unpatched run on a 16-example subset. Compares runs/smoke (unpatched) with runs/smoke_patched
(patched) for the same task/seed: Spearman over all 612 node scores, top-61 (10 %) node Jaccard,
max abs difference of signed edge scores. Writes checks/results/memory_patch_parity.json.
Run: uv run python checks/check_memory_patch_parity.py [task]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.overlap import jaccard
from safetensors.torch import load_file
from scipy.stats import spearmanr


def main() -> None:
    task, seed = (sys.argv[1] if len(sys.argv) > 1 else "b1_en"), 0
    a = Path("runs/smoke/circuits") / task / f"seed{seed}"
    b = Path("runs/smoke_patched/circuits") / task / f"seed{seed}"
    na, nb = json.loads((a / "nodes.json").read_text()), json.loads((b / "nodes.json").read_text())
    names = [n for n in na["names"] if n != "input"]
    sa = dict(zip(na["names"], na["signed_rowsum"], strict=False))
    sb = dict(zip(nb["names"], nb["signed_rowsum"], strict=False))

    def top(d):
        return sorted(names, key=lambda n: abs(d[n]), reverse=True)[:61]

    ea, eb = (
        load_file(str(a / "edges.safetensors"))["edges"],
        load_file(str(b / "edges.safetensors"))["edges"],
    )
    res = {
        "task": task,
        "node_spearman_signed": float(
            spearmanr([sa[n] for n in names], [sb[n] for n in names]).statistic
        ),
        "node_spearman_abs_artifact": float(
            spearmanr(
                [na["abs_scores"][n] for n in names], [nb["abs_scores"][n] for n in names]
            ).statistic
        ),
        "top10pct_jaccard": jaccard(top(sa), top(sb)),
        "edge_max_abs_diff": float((ea - eb).abs().max()),
        "edge_max_abs": float(ea.abs().max()),
        "edge_spearman": float(spearmanr(ea.flatten().numpy(), eb.flatten().numpy()).statistic),
    }
    Path("checks/results/memory_patch_parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
