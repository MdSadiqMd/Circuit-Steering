"""EAP-IG vs true activation patching on the top-50 nodes of one circuit (brief §12.3; Syed et al.
2023, Zhang & Wang 2026 "When Attribution Patching Lies").

EAP-IG node scores are a first-order estimate of the loss change when a node's clean activation is
replaced by its corrupted-run activation (CircuitKIT `attribute`: score = Σ (corrupt − clean) ·
∂loss/∂act). Here that change is measured directly: for each of the top-50 nodes by |signed EAP-IG
score| (row sums of the signed edge matrix from discovery) the node's activation (head result or
MLP output, every position) is overwritten with the corrupted-run value during the clean forward
pass on a 32-example subset of the task data, and the discovery loss (−logit difference) is compared
with the clean loss.
Reports Spearman between the two rankings on signed values and on magnitudes plus the sign-agreement
rate; the brief's rule is to raise `ig_steps` or correct the attribution if ρ < 0.7.
Writes checks/results/eap_vs_patching.json (one entry per task/seed).
Run: CIRCUITKIT_TL_MEMORY_PATCH=1 uv run python checks/check_eap_vs_patching.py task=b1_en seed=0
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.model import load_tl
from aya_circuits.registry import resolve_task
from circuitkit.applications.steering import ActivationSteering
from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "checks/results/eap_vs_patching.json"


@torch.no_grad()
def main() -> None:
    cfg = load_config()
    task, seed = str(cfg.get("task", "b1_en")), int(cfg.get("seed", 0))
    n, top_k = int(cfg.get("n", 32)), int(cfg.get("top_k", 50))
    model = load_tl(cfg)
    spec = resolve_task(task, cfg)
    metric = spec.metric_fn()  # discovery loss, mean over the batch
    dcfg = {
        "algorithm": "eap-ig",
        "task": spec.name,
        "level": "node",
        "batch_size": 8,
        "data_params": {"num_examples": n, "seed": seed},
        "seed": seed,
        "cache_dir": cfg["cache_dir"],
    }
    dl = spec.build_dataloader(model, dcfg, cfg["device"])
    nodes = json.loads((ROOT / "runs/circuits" / task / f"seed{seed}" / "nodes.json").read_text())
    signed = dict(zip(nodes["names"], nodes["signed_rowsum"], strict=True))
    signed.pop("input", None)
    top = sorted(signed, key=lambda x: abs(signed[x]), reverse=True)[:top_k]
    helper = ActivationSteering(model, dict.fromkeys(top, 1.0))
    hook_of = {x: helper._get_hook_point_from_node(x) for x in top}
    names = set(hook_of.values())

    t0 = time.time()
    batches = []
    for clean, corrupted, label in dl:
        ct, kt, cm, km, lengths, _ = tokenize_batch_pair(
            model,
            clean,
            corrupted,
            pair_padding_side=getattr(dl, "pair_padding_side", None),
            templated=getattr(dl, "templated", False),
        )
        _, cache = model.run_with_cache(kt, attention_mask=km, names_filter=lambda nm: nm in names)
        base = float(metric(model(ct, attention_mask=cm), None, lengths, label))
        batches.append((ct, cm, lengths, label, {k: v.clone() for k, v in cache.items()}, base))
    sizes = [len(b[3]) for b in batches]

    delta = {}
    for x in top:
        name, head = hook_of[x], (None if x.startswith("MLP ") else helper._node_head_index(x))
        acc = 0.0
        for (ct, cm, lengths, label, cache, base), bs in zip(batches, sizes, strict=True):
            src = cache[name]

            def hook(act, hook, src=src, head=head):
                if head is None:
                    return src
                act[:, :, head, :] = src[:, :, head, :]
                return act

            logits = model.run_with_hooks(ct, attention_mask=cm, fwd_hooks=[(name, hook)])
            acc += (float(metric(logits, None, lengths, label)) - base) * bs
        delta[x] = acc / sum(sizes)

    xs, ys = [signed[x] for x in top], [delta[x] for x in top]
    res = {
        "task": task,
        "seed": seed,
        "n_examples": sum(sizes),
        "top_k": top_k,
        "spearman_signed": float(spearmanr(xs, ys).statistic),
        "spearman_abs": float(spearmanr([abs(v) for v in xs], [abs(v) for v in ys]).statistic),
        "sign_agreement": sum((a > 0) == (b > 0) for a, b in zip(xs, ys, strict=True)) / len(xs),
        "top10_by_eap": {x: {"eap": signed[x], "patch": delta[x]} for x in top[:10]},
        "wall_s": round(time.time() - t0, 1),
    }
    all_res = json.loads(OUT.read_text()) if OUT.exists() else {}
    all_res[f"{task}_s{seed}"] = res
    OUT.write_text(json.dumps(all_res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "top10_by_eap"}))
    if res["spearman_signed"] < 0.7:
        print("WARNING: rho < 0.7, attribution misranks components (brief §12.3)", file=sys.stderr)


if __name__ == "__main__":
    main()
