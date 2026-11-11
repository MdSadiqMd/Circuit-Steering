"""Component selection for the five conditions (brief §3.3 item 3), from one circuit's node scores.

  circuit       top-k nodes by |score|  (k = top_fraction of all 612 heads+MLPs)
  inverse       bottom-k by |score|
  random        k nodes uniformly at random, seeded (>= 5 draws)
  taylor        top-k by gradient-magnitude importance  sum |theta * dL/dtheta| over the component's
                parameters on the behaviour data (second after IBCircuit in CircuitKIT E7);
                computed in interventions/taylor.py and passed in as a score dict
  unrestricted  every node (steering: a vector at all 612 node outputs; LoRA: adapters on every
                projection of every layer; editing: MEMIT over every editable MLP layer)

Node names follow CircuitKIT: 'A{layer}.{head}', 'MLP {layer}'. Head count is kept equal across
conditions, and so is the MLP count, so "size-matched" holds per component type, which is what
the LoRA parameter count and the steering norm depend on."""

from __future__ import annotations

import random

from aya_circuits.nodes import MLP, sizes, split, top_k  # one definition of the per-type rule
from aya_circuits.overlap import jaccard, weighted_jaccard  # (re-export)

_split = split


def select(
    condition: str,
    scores: dict[str, float],
    top_fraction: float,
    seed: int = 0,
    taylor_scores: dict[str, float] | None = None,
) -> list[str]:
    kh, km = sizes(scores, top_fraction)
    if condition in (
        "circuit",
        "circuit_deep",
    ):  # circuit_deep: same nodes; editing picks the layer differently
        return top_k(scores, kh, km)
    if condition == "inverse":
        return top_k(scores, kh, km, reverse=False)
    if condition == "random":
        heads, mlps = _split(list(scores))
        rng = random.Random(seed)
        return rng.sample(heads, kh) + rng.sample(mlps, km)
    if condition == "taylor":
        if taylor_scores is None:
            raise ValueError("taylor condition needs taylor_scores")
        return top_k(taylor_scores, kh, km)
    if condition == "unrestricted":
        return list(scores)
    raise ValueError(condition)


if __name__ == "__main__":
    # minimal runnable check
    sc = {f"A0.{h}": float(h) for h in range(10)} | {
        f"MLP {layer}": float(layer) for layer in range(5)
    }
    c = select("circuit", sc, 0.2)
    assert c == ["A0.9", "A0.8", "MLP 4"], c
    assert select("inverse", sc, 0.2) == ["A0.0", "A0.1", "MLP 0"]
    r1, r2 = select("random", sc, 0.2, seed=1), select("random", sc, 0.2, seed=1)
    assert r1 == r2 and len(r1) == 3 and sum(n.startswith(MLP) for n in r1) == 1
    assert len(select("unrestricted", sc, 0.2)) == 15
    assert abs(weighted_jaccard(sc, sc) - 1.0) < 1e-9 and jaccard(c, c) == 1.0
    print("select ok")
