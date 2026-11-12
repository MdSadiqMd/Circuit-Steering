"""Clean and corrupt metric means of every discovered task (seed-0 pillar examples), and what the
zero-ablation floor of the signed ratio is for each.

r = (m − c)/(cl − c) is the ratio CircuitKIT reports; a model whose logit difference is driven to 0
scores r0 = −c/(cl − c), which is 0.5 only when c = −cl. The zero-referenced ratio r' = m/cl (1 =
intact, 0 = destroyed, < 0 = inverted) is r' = r (1 − c/cl) + c/cl. Also records the bf16 quantum of
the mean metric (CircuitKIT accumulates the per-example metric in the model dtype), i.e. the
resolution of every reported ratio. Writes checks/results/metric_span.json.
Run: uv run python checks/check_metric_span.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401
import torch
from aya_circuits.config import load_config
from aya_circuits.faithfulness import discovery_cfg
from aya_circuits.model import load_tl
from aya_circuits.registry import resolve_task
from circuitkit.api import _make_eval_metric

ROOT = Path(__file__).resolve().parent.parent
TASKS = [
    "b1_en",
    "b1_hi",
    "b1_ar",
    "b1_fr",
    "b1_sw",
    "b2_en",
    "ioi",
    "langid_en",
    "ioi_fixed",
    "b2m_en",
]


def bf16_quantum(x: float) -> float:
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 0.0


@torch.no_grad()
def means(model, spec, cfg: dict) -> tuple[float, float]:
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair

    dl = spec.build_dataloader(model, discovery_cfg(cfg, spec, 0, "zero", None), cfg["device"])
    mf = _make_eval_metric(spec)
    side, templated = getattr(dl, "pair_padding_side", None), getattr(dl, "templated", False)
    cs, ks = [], []
    for clean, corrupt, labels in dl:  # same tokenisation as CircuitKIT's attribute / pillars
        tc, tk, mc, mk, lens, _ = tokenize_batch_pair(
            model, clean, corrupt, pair_padding_side=side, templated=templated
        )
        lc, lk = model(tc, attention_mask=mc), model(tk, attention_mask=mk)
        cs.append(mf(lc, lc, lens, labels).float().cpu())
        ks.append(mf(lk, lc, lens, labels).float().cpu())
    c, k = torch.cat(cs), torch.cat(ks)
    return float(c.mean()), float(k.mean())


def main() -> None:
    cfg = load_config()
    model = load_tl(cfg)
    out = {}
    for task in TASKS:
        cl, co = means(model, resolve_task(task, cfg), cfg)
        span = cl - co
        out[task] = {
            "clean": cl,
            "corrupt": co,
            "ratio_at_zero_metric": -co / span,
            "corrupt_over_clean": co / cl,
            "ratio_resolution_bf16": bf16_quantum(cl) / abs(span),
        }
        print(task, json.dumps(out[task]), flush=True)
    (ROOT / "checks/results/metric_span.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
