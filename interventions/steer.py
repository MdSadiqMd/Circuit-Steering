"""Circuit steering: difference-of-means vectors at the condition's nodes, applied at inference.

Vectors: for every node (head result or MLP output, CircuitKIT hook points via
`ActivationSteering._get_hook_point_from_node`) v = mean_target - mean_source of the activation at
the last prompt position (the assistant marker). CircuitKIT's `compute_steering_vector` averages
over all (zero-padded) positions; the last-position mean is the refusal-direction convention
(Arditi et
al. 2024; SafeTune RefusalDirection) and avoids padding contamination, so we collect it ourselves
with the same hook names. Application adds coeff·v at every position (apply_hf / apply_tl).

Size: S = sqrt(sum_nodes ||coeff·v||^2). Conditions are norm-matched to the circuit condition at
coefficient 1 (so a random set with smaller natural vectors gets a larger coefficient), and the
coefficient sweep in configs/base.yaml multiplies that matched base.

B2: source = benign near-miss prompts, target = harmful prompts the model refuses (positive coeff =
more refusal; only this sign is ever released). B1: source = facts the model gets wrong, target =
facts it gets right ("recall confidence" direction).
"""

from __future__ import annotations

import torch
from circuitkit.applications.steering import ActivationSteering

from interventions.cell import Cell


@torch.no_grad()
def last_position_means(
    model, texts: list[str], nodes: list[str], batch_size: int = 8
) -> dict[str, torch.Tensor]:
    """Mean activation at the last token for each node; attention nodes read their head slice."""
    helper = ActivationSteering(model, dict.fromkeys(nodes, 1.0))
    hooks = {n: helper._get_hook_point_from_node(n) for n in nodes}
    sums = {n: torch.zeros(model.cfg.d_model, dtype=torch.float32) for n in nodes}
    count = 0
    names = sorted(set(hooks.values()))
    for i in range(0, len(texts), batch_size):
        toks = model.to_tokens(texts[i : i + batch_size], prepend_bos=False, padding_side="left")
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
        for n in nodes:
            act = cache[hooks[n]][
                :, -1
            ]  # [batch, (head,) d_model], left padding puts the prompt end last
            if act.dim() == 3:
                act = act[:, helper._node_head_index(n)]
            sums[n] += act.float().sum(0).cpu()
        count += toks.shape[0]
    return {n: s / count for n, s in sums.items()}


def steering_vectors(
    model, source_texts, target_texts, nodes, batch_size=8
) -> dict[str, torch.Tensor]:
    ms, mt = (
        last_position_means(model, source_texts, nodes, batch_size),
        last_position_means(model, target_texts, nodes, batch_size),
    )
    return {n: mt[n] - ms[n] for n in nodes}


def size(vectors: dict[str, torch.Tensor], coeff: float) -> float:
    return float(torch.sqrt(sum((coeff * v).pow(2).sum() for v in vectors.values())))


def build_cell(
    model,
    behaviour: str,
    condition: str,
    seed: int,
    nodes: list[str],
    source,
    target,
    out_root,
    match_size: float | None,
    batch_size: int = 8,
    max_factor: float = 20.0,
) -> tuple[Cell, dict[str, torch.Tensor], float]:
    """Compute vectors, pick the norm-matched base coefficient (capped at max_factor), save the
    artifact.
    Returns size at coeff 1."""
    vecs = steering_vectors(model, source, target, nodes, batch_size)
    s1 = size(vecs, 1.0)
    coeff = 1.0 if match_size is None else min(max_factor, match_size / max(s1, 1e-8))
    cell = Cell(
        behaviour,
        "steer",
        condition,
        seed,
        nodes,
        meta={
            "size_at_1": s1,
            "base_coefficient": coeff,
            "n_source": len(source),
            "n_target": len(target),
            "per_node_norm": {n: float(v.norm()) for n, v in vecs.items()},
        },
    )
    tensors = {f"vec:{n}": v for n, v in vecs.items()} | {"coefficient": torch.tensor(coeff)}
    cell.save(out_root, tensors)
    return cell, tensors, s1
