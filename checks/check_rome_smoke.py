"""Brief §7.4: smoke-test one ROME edit on Tiny Aya through CircuitKIT's circuit-guided editor.

`CircuitKnowledgeEditor.edit_via_circuit(method="rome")` with the b1_en seed-0 circuit picks the
edit layer itself; the fact is the first false fact of seed 0's fold (English chat prompt). Same
numerics as the MEMIT cells (interventions/edit.py): fp32 TL model, covariance and solve on CPU
float64, parameters frozen. Records the chosen layer, target-token probability before/after,
|ΔW_out| and the editor's own result, restores the weights, and writes
checks/results/rome_smoke.json. A failure is recorded there rather than raised (the brief: "if
cohere2 editing fails, file the failure").
`edit_via_circuit` cannot pick a layer from a loaded `Circuit` (it expects `.nodes`/`.edges`, gets
None and fails), which is filed under `editor_result`; the smoke edit then calls CircuitKIT's
`RomeHandler` directly at MLP 5, the B1 `circuit_deep` site. CPU, because the editor clones every
parameter in fp32, which exceeds the MPS cap.
Run: uv run python checks/check_rome_smoke.py device=cpu
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # before transformer_lens
import pandas as pd
import torch
from aya_circuits.config import load_config
from aya_circuits.model import load_tl
from circuitkit.applications.editing import CircuitKnowledgeEditor
from circuitkit.applications.editing.rome_wrapper import RomeHandler

from interventions.edit import _lean_model, patch_covariance
from interventions.run_grid import b1_folds, fact_prompt_en

ROOT = Path(__file__).resolve().parent.parent
DIRECT_LAYER = 5


@torch.no_grad()
def p_target(model, prompt: str, target: str) -> float:
    toks = model.to_tokens(prompt, prepend_bos=False)
    # the reply starts with a space-prefixed token (greedy " Santiago"): score " " + target
    tid = model.to_tokens(prompt + " " + target, prepend_bos=False)[0, toks.shape[1]]
    return float(model(toks)[0, -1].float().softmax(-1)[tid])


@torch.no_grad()
def greedy(model, prompt: str, n: int = 12) -> str:
    toks = model.to_tokens(prompt, prepend_bos=False)
    start = toks.shape[1]
    for _ in range(n):
        toks = torch.cat([toks, model(toks)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return model.tokenizer.decode(toks[0, start:])


def main() -> None:
    cfg = load_config()
    out: dict = {"model": cfg["model"], "method": "rome"}
    model = load_tl(cfg).to(torch.float32)
    model.cfg.dtype = torch.float32
    patch_covariance()
    fact = b1_folds(cfg)[0][0]
    prompt, subject, target = fact_prompt_en(fact, model.tokenizer, cfg)
    out["fact"] = {"relation": fact["relation"], "subject": subject, "target": target}
    circuit = ck.load_scores(str(ROOT / "runs/circuits/b1_en/seed0/circuit.pt"))
    df = pd.read_parquet(ROOT / ".cache/raw/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet")
    cov = [t.strip() for t in df.text.tolist() if len(t.strip()) > 200][
        : cfg["interventions"]["editing"]["cov_n_samples"]
    ]
    before_w = {i: b.mlp.W_out.detach().cpu().clone() for i, b in enumerate(model.blocks)}
    out["p_target_before"] = p_target(model, prompt, target)
    out["greedy_before"] = greedy(model, prompt)
    t0 = time.time()
    try:
        editor = CircuitKnowledgeEditor(model)
        with _lean_model(model):
            res = editor.edit_via_circuit(
                prompt,
                subject,
                target,
                circuit=circuit,
                method="rome",
                cov_texts=cov,
                corpus_id="wikitext2",
                n_prefixes=cfg["interventions"]["editing"]["n_prefixes"],
            )
        out["editor_result"] = {
            k: getattr(res, k, None)
            for k in ("success", "confidence_before", "confidence_after", "error_message")
        }
        del editor
        with _lean_model(model):
            res = RomeHandler(model).edit_single_fact(
                prompt=prompt,
                subject=subject,
                target=target,
                target_layer=DIRECT_LAYER,
                cov_texts=cov,
                corpus_id="wikitext2",
                n_prefixes=cfg["interventions"]["editing"]["n_prefixes"],
            )
        out["direct_layer"] = DIRECT_LAYER
        out["direct_result"] = {
            k: getattr(res, k, None)
            for k in ("success", "confidence_before", "confidence_after", "error_message")
        }
    except Exception as e:  # filed, not raised: §7.4 asks for the failure to be recorded
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
    out["wall_s"] = round(time.time() - t0, 1)
    out["p_target_after"] = p_target(model, prompt, target)
    out["greedy_after"] = greedy(model, prompt)
    changed = {
        i: float((b.mlp.W_out.detach().cpu() - before_w[i]).norm())
        for i, b in enumerate(model.blocks)
    }
    out["edited_layers"] = {i: n for i, n in changed.items() if n > 0}
    with torch.no_grad():
        for i, b in enumerate(model.blocks):
            b.mlp.W_out.copy_(before_w[i].to(b.mlp.W_out.device))
    out["p_target_restored"] = p_target(model, prompt, target)
    out["passed"] = bool(out["edited_layers"]) and target.split()[0] in out["greedy_after"]
    (ROOT / "checks/results/rome_smoke.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "traceback"}, indent=1, default=str))


if __name__ == "__main__":
    main()
