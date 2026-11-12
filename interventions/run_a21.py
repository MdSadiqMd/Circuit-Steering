"""Amendment A21 (plan item 2): MEMIT at every editable MLP layer, uncapped, norm-capped and with
norm-matched noise, on the three B1 false-fact folds.

  uv run python interventions/run_a21.py out_dir=runs/a21 [a21.sites=[0,1,...]] [a21.folds=[0]]

Per fold d (seed d's fold of interventions/run_grid.b1_folds, the facts every B1 cell of seed d
trained on) and site L in `a21.sites` (default SITES: the even editable layers 0..28 plus 3, the
A8 causal-tracing site, and 5, the `circuit_deep` site):
  site<L>                 MEMIT at [L], pre-registered editor settings (interventions/edit.py)
  site<L>_cap             the same update rescaled to |ΔW|_F = CAP (one scalar per cell)
  perturb_site<L>_cap     norm-matched random rank-k noise at L with |ΔW|_F = CAP
                          (interventions/perturb.py, seed d + 100)
CAP = 59.187628, the median |ΔW|_F of the 20 single-layer B1 MEMIT cells of the main study (circuit,
circuit_deep, inverse, random, taylor, causal_trace), fixed in A21 before any run.

The key covariance C_L (1000 WikiText-2 texts, float64 accumulation, cached per layer under
KE_CACHE_DIR) is the same quantity as edit.get_covariance_cpu; here it is accumulated on the model
device in float64 and each forward stops after layer L, which changes the summation order only
(`--check-cov` compares both on real activations).
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import pandas as pd
import torch
from aya_circuits.config import load_config
from aya_circuits.manifest import Manifest
from aya_circuits.memwatch import MemWatch
from aya_circuits.model import load_tl
from circuitkit.applications.common_utils import _covariance as _cov

from interventions import edit
from interventions.cell import Cell, delta_norm
from interventions.perturb import perturbed
from interventions.run_grid import b1_folds, fact_prompt_en

ROOT = Path(__file__).resolve().parent.parent
CAP = 59.187628
SITES = sorted(set(range(0, 29, 2)) | {3, 5})


def get_covariance_device64(
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
    """edit.get_covariance_cpu with the float64 accumulation on the model device and the forward
    stopped after `layer` (same cache key and file, same texts, tokens and truncation)."""
    model_name = getattr(getattr(model, "cfg", None), "model_name", "unknown")
    key = _cov._cache_key(model_name, layer, hook_name, n_samples, corpus_id)
    cache_path = _cov._cache_dir() / f"{key}.pt"
    if use_cache and cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    texts = list(texts if texts is not None else _cov._FALLBACK_TEXTS)[:n_samples]
    accum, n_tokens = None, 0

    def _hook(act, hook):
        nonlocal accum, n_tokens
        flat = act.detach().to(torch.float64).reshape(-1, act.shape[-1])
        accum = flat.T @ flat if accum is None else accum.addmm_(flat.T, flat)
        n_tokens += flat.shape[0]
        return act

    try:
        for text in texts:
            toks = model.to_tokens(text)[:, :max_seq_len]
            with torch.no_grad():
                model.run_with_hooks(
                    toks,
                    fwd_hooks=[(f"blocks.{layer}.{hook_name}", _hook)],
                    stop_at_layer=layer + 1,
                )
    finally:
        model.reset_hooks()
    if accum is None:
        return None
    C = (accum / n_tokens).to(torch.float32).cpu()
    if use_cache:
        _cov._cache_dir().mkdir(parents=True, exist_ok=True)
        torch.save(C, cache_path)
    return C


def capped(cell: Cell, tensors: dict, cap: float) -> tuple[Cell, dict]:
    norm = delta_norm(tensors, "edit")
    scale = cap / norm
    out = {k: (v * scale if k.startswith("U:") else v) for k, v in tensors.items()}
    meta = {k: v for k, v in cell.meta.items() if k in ("layers", "edited_keys", "disc_seed")}
    meta |= {"capped_from": cell.name, "cap": cap, "scale": scale, "uncapped_norm": norm}
    return Cell(cell.behaviour, "edit", f"{cell.condition}_cap", cell.seed, cell.nodes, meta), out


def wikitext_texts(n: int) -> list[str]:
    df = pd.read_parquet(ROOT / ".cache/raw/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet")
    return [t.strip() for t in df.text.tolist() if len(t.strip()) > 200][:n]


def check_cov(cfg: dict) -> None:
    """Device-float64 covariance (early stop) vs the CPU function on 6 real texts, 2 layers."""
    model = load_tl(cfg).to(torch.float32)
    model.cfg.dtype = torch.float32
    texts = wikitext_texts(6)
    for layer in (0, 17):
        a = edit.get_covariance_cpu(model, layer, "mlp.hook_post", texts, 6, use_cache=False)
        b = get_covariance_device64(model, layer, "mlp.hook_post", texts, 6, use_cache=False)
        rel = float((a - b).norm() / a.norm())
        print(f"layer {layer}: rel diff {rel:.2e} max {float((a - b).abs().max()):.2e}")
        assert rel < 1e-6, rel


def main() -> None:
    cfg = load_config()
    beh = "b1"
    run_root = Path(cfg["out_dir"])
    assert run_root.resolve() != (ROOT / "runs").resolve(), (
        "A21 cells must not go into the main root"
    )
    ecfg = cfg["interventions"]["editing"]
    n_layers = 36
    sites = cfg.get("a21", {}).get("sites") or SITES
    assert set(sites) <= set(edit.editable_layers(n_layers)), sites
    mw = MemWatch(cfg["memory_budget_gb"])
    model = load_tl(cfg).to(torch.float32)  # bf16 load then upcast, as run_grid.py (edit)
    model.cfg.dtype = torch.float32
    assert model.cfg.n_layers == n_layers
    tok = model.tokenizer
    edit.get_covariance_cpu = get_covariance_device64  # picked up by edit.patch_covariance()
    cov_texts = wikitext_texts(ecfg["cov_n_samples"])
    folds = b1_folds(cfg)
    # a21.folds restricts which folds are built; cfg seeds must stay [0, 1, 2] because b1_folds
    # splits the facts by the number of seeds
    for d in cfg.get("a21", {}).get("folds") or cfg["seeds"]:
        facts = [fact_prompt_en(f, tok, cfg) for f in folds[d]]
        keys = [[f["relation"], f["subject_qid"]] for f in folds[d]]
        for layer in sites:
            cond = f"site{layer:02d}"
            base_cell = Cell(beh, "edit", cond, d, [f"MLP {layer}"])
            noise_dir = Cell(beh, "edit", f"perturb_{cond}_cap", d, []).dir(run_root)
            if (noise_dir / "delta.safetensors").exists() and not cfg.get("force"):
                print("cached", cond, d, flush=True)
                continue
            mf, t0 = Manifest("a21_edit", cfg, d), time.time()
            mf.add_input(ROOT / "data/b1/false_facts.jsonl")
            if (base_cell.dir(run_root) / "delta.safetensors").exists() and not cfg.get("force"):
                cell, tensors = Cell.load(base_cell.dir(run_root))
            else:
                cell, tensors, stats = edit.edit_cell(
                    model,
                    beh,
                    cond,
                    d,
                    [f"MLP {layer}"],
                    facts,
                    [layer],
                    cov_texts,
                    ecfg,
                    run_root,
                )
                if not any(k.startswith("U:") for k in tensors):
                    # a missing cell is reported by the analysis (A21 validity check), not fatal
                    print(f"FAILED {cond} s{d}: no weight change {json.dumps(stats)}", flush=True)
                    continue
                cell.meta.update(
                    delta_norm=delta_norm(tensors, "edit"),
                    edited_keys=keys,
                    disc_seed=d,
                    amendment="A21",
                )
                cell.save(run_root, tensors)
            cap_cell, cap_t = capped(cell, tensors, CAP)
            cap_cell.meta["delta_norm"] = delta_norm(cap_t, "edit")
            cap_cell.save(run_root, cap_t)
            pc, pt = perturbed(cap_cell, cap_t, seed=d + 100)
            pc.meta.update(
                delta_norm=delta_norm(pt, "edit"),
                edited_keys=keys,
                layers=[layer],
                disc_seed=d,
                amendment="A21",
            )
            pc.save(run_root, pt)
            mw.check()
            mf.add_output(cell.dir(run_root) / "delta.safetensors")
            mf.write(
                cell.dir(run_root) / "manifest.json",
                memory=mw.summary(),
                wall_s=round(time.time() - t0, 1),
            )
            print(
                f"{cell.name}: {time.time() - t0:.0f}s |Δ|={cell.meta['delta_norm']:.2f} "
                f"cap x{cap_cell.meta['scale']:.3f} noise |Δ|={pc.meta['delta_norm']:.2f} "
                f"success {cell.meta.get('n_success')}/{cell.meta.get('n_facts')}",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if "--check-cov" in sys.argv:
        sys.argv.remove("--check-cov")
        check_cov(load_config())
    elif sys.argv[1:] == ["--check"]:
        c = Cell("b1", "edit", "site03", 0, ["MLP 3"], {"layers": [3], "edited_keys": [["r", "q"]]})
        t = {"U:3": torch.randn(50, 4), "V:3": torch.randn(4, 20)}
        cc, ct = capped(c, t, CAP)
        assert abs(delta_norm(ct, "edit") - CAP) < 1e-3 and cc.condition == "site03_cap"
        pc, pt = perturbed(cc, ct, seed=100)
        assert abs(delta_norm(pt, "edit") - CAP) < 1e-3 and pc.condition == "perturb_site03_cap"
        assert len(SITES) == 17 and SITES[:4] == [0, 2, 3, 4] and SITES[-1] == 28
        print("run_a21 ok")
    else:
        main()
