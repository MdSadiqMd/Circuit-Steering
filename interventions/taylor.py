"""Taylor (gradient-magnitude) component importance on the TransformerLens model.

score(component) = sum over its parameters of |theta * dL/dtheta|, L = the task's discovery loss on
clean prompts (first-order Taylor estimate of the loss change from zeroing the parameters; second
after IBCircuit among CircuitKIT E7's selectors, 0.690 vs 0.697). Heads: W_Q[h], W_O[h], plus their
KV group's _W_K/_W_V share divided by the group size (GQA: 4 query heads per KV head on Tiny
Aya). MLPs: W_in, W_gate, W_out.
Returns a CircuitKIT-named dict {'A{l}.{h}': s, 'MLP {l}': s} so interventions/select.py can rank it
exactly like EAP-IG scores."""

from __future__ import annotations

import torch


def taylor_scores(model, dataloader, metric, n_batches: int | None = None) -> dict[str, float]:
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair

    cfg = model.cfg
    group = cfg.n_heads // (cfg.n_key_value_heads or cfg.n_heads)
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    for i, (clean, corrupted, label) in enumerate(dataloader):
        if n_batches is not None and i >= n_batches:
            break
        toks, _, mask, _, lengths, _ = tokenize_batch_pair(
            model,
            clean,
            corrupted,
            pair_padding_side=getattr(dataloader, "pair_padding_side", None),
            templated=getattr(dataloader, "templated", False),
        )
        logits = model(toks, attention_mask=mask)
        metric(logits, None, lengths, label).backward()
    scores: dict[str, float] = {}
    with torch.no_grad():
        for layer, blk in enumerate(model.blocks):
            a = blk.attn
            tq = (
                (a.W_Q * a.W_Q.grad).abs().sum(dim=(1, 2))
                if a.W_Q.grad is not None
                else torch.zeros(cfg.n_heads)
            )
            to = (
                (a.W_O * a.W_O.grad).abs().sum(dim=(1, 2))
                if a.W_O.grad is not None
                else torch.zeros(cfg.n_heads)
            )
            wk, wv = a._W_K, a._W_V
            tk = (
                (wk * wk.grad).abs().sum(dim=(1, 2))
                if wk.grad is not None
                else torch.zeros(wk.shape[0])
            )
            tv = (
                (wv * wv.grad).abs().sum(dim=(1, 2))
                if wv.grad is not None
                else torch.zeros(wv.shape[0])
            )
            for h in range(cfg.n_heads):
                kv = h // group
                scores[f"A{layer}.{h}"] = float(tq[h] + to[h] + (tk[kv] + tv[kv]) / group)
            m = blk.mlp
            s = 0.0
            for w in (m.W_in, m.W_out, getattr(m, "W_gate", None)):
                if w is not None and w.grad is not None:
                    s += float((w * w.grad).abs().sum())
            scores[f"MLP {layer}"] = s
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return scores
