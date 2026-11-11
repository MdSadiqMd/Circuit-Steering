"""Knowledge editing at circuit-chosen components, through CircuitKIT's ROME/MEMIT, with the
numerically sensitive statistics on CPU.

Layer choice per condition (editing acts on MLP W_out, so only MLP nodes count):
  circuit  -> MLP node with the largest |EAP-IG score| among the circuit's MLP nodes (MLP 0 on both
              behaviours in the smoke runs)
  circuit_deep -> exploratory: same, excluding layers 0-3 (pre-registered as secondary)
  inverse  -> the MLP node of the inverse (bottom-k) set with the smallest |score|
  random   -> seeded random MLP layer
  taylor   -> MLP layer with the largest Taylor score
  unrestricted -> MEMIT spread over every editable MLP layer (0..80 % depth, see EDITABLE_DEPTH)
B1 facts are edited as a batch with MEMIT (`MemitHandler.edit_multiple_facts`) at the chosen
layer(s); ROME (single fact, single layer) is not used. The resulting weight change is read back as
W_out_after - W_out_before per layer and stored as rank-k factors (SVD, exact since the MEMIT
update has rank <= n_facts).

MPS: `_covariance.get_covariance` accumulates in float64 on the activation device, which MPS lacks,
and CircuitKIT warns MPS editing can be silently wrong. We monkeypatch the module attribute with a
copy that moves activations to CPU before accumulating; `solve_with_C` then solves on CPU. The model
itself is upcast to float32 in place after a bf16 load for editing (a direct fp32 load peaks at 3×
weights = 40 GB) and the delta is applied to the bf16 audit model afterwards
(interventions/cell.py).
"""

from __future__ import annotations

import time
import warnings

import torch
from circuitkit.applications.common_utils import _covariance as _cov
from circuitkit.applications.editing.memit_wrapper import MemitHandler

from interventions.cell import Cell


def get_covariance_cpu(
    model,
    layer,
    hook_name,
    texts=None,
    n_samples=1000,
    max_seq_len=256,
    device=None,
    use_cache=True,
    corpus_id="default",
):
    """Same contract as circuitkit `_covariance.get_covariance`, accumulation on CPU float64."""
    model_name = getattr(getattr(model, "cfg", None), "model_name", "unknown")
    key = _cov._cache_key(model_name, layer, hook_name, n_samples, corpus_id)
    cache_path = _cov._cache_dir() / f"{key}.pt"
    if use_cache and cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    texts = list(texts if texts is not None else _cov._FALLBACK_TEXTS)[:n_samples]
    accum, n_tokens = None, 0

    def _hook(act, hook):
        nonlocal accum, n_tokens
        flat = (
            act.detach().to("cpu").to(torch.float64).reshape(-1, act.shape[-1])
        )  # move first: MPS cannot cast to float64
        contrib = flat.T @ flat
        accum = contrib if accum is None else accum + contrib
        n_tokens += flat.shape[0]
        return act

    try:
        for text in texts:
            toks = model.to_tokens(text)[:, :max_seq_len]
            with torch.no_grad():
                model.run_with_hooks(toks, fwd_hooks=[(f"blocks.{layer}.{hook_name}", _hook)])
    finally:
        model.reset_hooks()
    if accum is None:
        return None
    C = (accum / n_tokens).to(torch.float32)
    if use_cache:
        torch.save(C, cache_path)
    return C


_orig_layer_update = MemitHandler._compute_layer_update


def _layer_update_cpu(self, K, R, mom2_weight=15000.0, C=None):
    """MEMIT's solve does `K.double()` on the model device; MPS has no float64, the exception was
    swallowed and the edit silently became a no-op (first smoke: |Δ| = 0, n_success = 0). Solve on
    CPU.
    CircuitKIT also runs a full `eigvalsh` of the 11008×11008 system purely to log its condition
    number
    (100 s per layer on this CPU); it is wrapped in a try/except there, so we make it raise and
    skip it."""

    def skip(*a, **k):
        raise RuntimeError("eigvalsh skipped (diagnostic only)")

    orig_eig = torch.linalg.eigvalsh
    torch.linalg.eigvalsh = skip
    try:
        upd = _orig_layer_update(
            self,
            K.detach().cpu(),
            R.detach().cpu(),
            mom2_weight,
            None if C is None else C.detach().cpu(),
        )
    finally:
        torch.linalg.eigvalsh = orig_eig
    return None if upd is None else upd.to(K.device)


def patch_covariance() -> None:
    import circuitkit.applications.editing.memit_wrapper as _mw

    _cov.get_covariance = get_covariance_cpu
    _mw.get_covariance = get_covariance_cpu  # memit_wrapper binds the name at import time
    MemitHandler._compute_layer_update = _layer_update_cpu


EDITABLE_DEPTH = (
    0.8  # MEMIT needs downstream layers to propagate z: a target at the last layer gets zero
)
# gradient (smoke: |z-h_L| = 0 at L = 35). Editable layers are 0..int(0.8*n_layers) = 0..28,
# the upper bound of CircuitKIT's own default band (40-80 % depth).


def editable_layers(n_layers: int) -> list[int]:
    return list(range(int(EDITABLE_DEPTH * n_layers) + 1))


def mlp_layer_for(
    condition: str,
    nodes: list[str],
    scores: dict[str, float],
    seed: int,
    n_layers: int,
    taylor: dict[str, float] | None = None,
) -> list[int]:
    ok = set(editable_layers(n_layers))
    mlps = [int(n[4:]) for n in nodes if n.startswith("MLP ") and int(n[4:]) in ok]
    if condition == "unrestricted":
        return sorted(ok)
    if condition == "causal_trace":  # node set is the traced layer itself (run_grid.py, A8)
        return mlps
    if condition == "random":
        g = torch.Generator().manual_seed(seed)
        return [sorted(ok)[int(torch.randint(0, len(ok), (1,), generator=g))]]
    if condition == "taylor":
        cand = [layer for layer in ok if f"MLP {layer}" in taylor]
        return [max(cand, key=lambda layer: abs(taylor[f"MLP {layer}"]))]
    if (
        condition == "circuit_deep"
    ):  # exploratory: the circuit's top MLP excluding the detokenisation block (layers 0-3)
        deep = [layer for layer in mlps if layer >= 4] or [
            layer for layer in sorted(ok) if layer >= 4
        ]
        return [max(deep, key=lambda layer: abs(scores[f"MLP {layer}"]))]
    if not mlps:  # no editable MLP in the condition's node set: best editable MLP by |score|
        mlps = sorted(ok)

    def key(layer: int) -> float:
        return abs(scores[f"MLP {layer}"])

    return [max(mlps, key=key) if condition == "circuit" else min(mlps, key=key)]


@torch.no_grad()
def _snapshot(model, layers: list[int]) -> dict[int, torch.Tensor]:
    return {layer: model.blocks[layer].mlp.W_out.detach().float().cpu().clone() for layer in layers}


def factorize(delta: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """delta ≈ U V with rank k via SVD; returns (U, V, relative reconstruction error)."""
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    U, S, Vh = U[:, :k], S[:k], Vh[:k]
    Uk, Vk = U * S, Vh
    err = float((Uk @ Vk - delta).norm() / (delta.norm() + 1e-12))
    return Uk.contiguous(), Vk.contiguous(), err


class _lean_model:
    """Freeze parameters and drop the EAP hook flags while editing: MEMIT's z-optimisation
    backpropagates
    through the model and would otherwise allocate a full fp32 gradient for every weight (13 GB) and
    per-head attention results; the first smoke run swapped 30 GB on exactly this."""

    FLAGS = ("use_attn_result", "use_split_qkv_input", "use_hook_mlp_in")

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.saved = {f: getattr(self.model.cfg, f) for f in self.FLAGS}
        for f in self.FLAGS:
            setattr(self.model.cfg, f, False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    def __exit__(self, *exc):
        for f, v in self.saved.items():
            setattr(self.model.cfg, f, v)
        self.model.zero_grad(set_to_none=True)


def edit_cell(
    model,
    behaviour: str,
    condition: str,
    seed: int,
    nodes: list[str],
    facts: list[tuple[str, str, str]],
    layers: list[int],
    cov_texts: list[str],
    ecfg: dict,
    out_root,
) -> tuple[Cell, dict, dict]:
    """Apply a MEMIT batch edit at `layers`, capture the delta, restore the model, save factors."""
    patch_covariance()
    before = _snapshot(model, layers)
    handler = MemitHandler(model)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught, _lean_model(model):
        warnings.simplefilter("always")
        results = handler.edit_multiple_facts(
            facts,
            target_layers=layers,
            verify_interference=False,
            cov_n_samples=ecfg["cov_n_samples"],
            cov_texts=cov_texts,
            corpus_id="wikitext2",
            n_prefixes=ecfg["n_prefixes"],
            prefix_seed=seed,
        )
    wall = time.time() - t0
    after = _snapshot(model, layers)
    tensors, errs = {}, {}
    k = min(len(facts), 64)
    for layer in layers:
        d = after[layer] - before[layer]
        if float(d.norm()) == 0.0:
            continue
        U, V, err = factorize(d, k)
        tensors[f"U:{layer}"], tensors[f"V:{layer}"], errs[layer] = U, V, err
    with torch.no_grad():  # restore the base model
        for layer in layers:
            model.blocks[layer].mlp.W_out.copy_(
                before[layer].to(model.blocks[layer].mlp.W_out.dtype).to(model.cfg.device)
            )
    stats = {
        "layers": layers,
        "wall_s": round(wall, 1),
        "n_facts": len(facts),
        "n_success": sum(bool(r.success) for r in results),
        "svd_rel_err": errs,
        "conf_before_mean": sum(r.confidence_before for r in results) / max(1, len(results)),
        "conf_after_mean": sum(r.confidence_after for r in results) / max(1, len(results)),
        "errors": [r.error_message for r in results if getattr(r, "error_message", None)][:5],
        "warnings": sorted({str(w.message)[:160] for w in caught})[:10],
    }
    cell = Cell(
        behaviour,
        "edit",
        condition,
        seed,
        nodes,
        meta={"layers": layers, "n_facts": len(facts), **stats},
    )
    cell.save(out_root, tensors)
    return cell, tensors, stats
