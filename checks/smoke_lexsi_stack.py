"""Acceptance check (brief §6): the three Lexsi packages load, run on CohereLabs/tiny-aya-global on
this machine, and talk to each other. Writes checks/results/stack_check.json; exit 0 only if every
assert passes. `--dry` only checks imports, versions and entry points (no model). Signatures were
verified against the installed source (docs/notes.md 2026-11-02)."""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auditkit as ak
import circuitkit as ck  # must be imported before transformer_lens
import safetune  # noqa: F401
from aya_circuits.config import load_config
from aya_circuits.model import load_tl
from circuitkit.applications.editing import CircuitKnowledgeEditor
from circuitkit.applications.steering import ActivationSteering
from safetune.evaluate import evaluate as st_eval
from safetune.runner import steer as st_steer


def main() -> None:
    dry = "--dry" in sys.argv
    cfg = load_config([a for a in sys.argv[1:] if "=" in a])
    model_id = cfg["model"]
    versions = {
        p: md.version(p)
        for p in (
            "circuitkit",
            "auditkit",
            "safetune",
            "transformer-lens",
            "transformers",
            "torch",
            "peft",
        )
    }

    assert callable(st_eval) and hasattr(st_steer, "RefusalDirectionTrainer")
    assert callable(ck.discover) and callable(ck.faithfulness) and callable(ck.selective_finetune)
    assert callable(ak.evaluate) and callable(ak.compare)
    if dry:
        print("DRY OK", versions)
        return

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set; accept the Cohere licence on HF first"
    t0 = time.time()
    model = load_tl(cfg)
    arch = getattr(model.cfg, "original_architecture", None) or str(type(model))
    assert "cohere" in str(arch).lower(), f"not a Cohere architecture: {arch}"
    load_s = time.time() - t0

    # 1. CircuitKIT discovers and evaluates a circuit on the real model.
    out = Path(cfg["cache_dir"]) / "smoke" / "tiny_aya_ioi.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    circuit = ck.discover(
        model,
        "ioi",
        algorithm="eap-ig",
        n_examples=16,
        batch_size=4,
        ig_steps=3,
        output_path=str(out),
        seed=0,
    )
    disc_s = time.time() - t0
    assert len(circuit.scores) > 0
    t0 = time.time()
    report = ck.faithfulness(
        model,
        circuit,
        "ioi",
        pillars=["patching", "ablation"],
        n_examples=16,
        batch_size=8,
        device=cfg["device"],
    )
    faith_s = time.time() - t0

    # 2. The three intervention entry points accept this circuit.
    steer = ActivationSteering(model, circuit.scores)
    masks = ck.selective_finetune(circuit, model_name=model_id, top_fraction=0.2, scope="both")
    assert masks.attn or masks.mlp
    editor = CircuitKnowledgeEditor(model)  # construction only; edits are experiments

    # 3. AuditKIT scores the same checkpoint (hf backend, same device) and compares.
    samples = [
        ak.Sample(input="The capital of France is", target="Paris"),
        ak.Sample(input="La capitale de la France est", target="Paris"),
    ]
    base = ak.evaluate(
        samples,
        f"hf:{model_id}",
        ["exact_match", "f1_score"],
        device=cfg["device"],
        dtype=cfg["dtype"],
    )
    cmp = ak.compare(base, base)
    assert cmp is not None

    result = {
        "model": model_id,
        "arch": str(arch),
        "device": cfg["device"],
        "dtype": cfg["dtype"],
        "versions": versions,
        "load_s": round(load_s, 1),
        "discover_s": round(disc_s, 1),
        "faithfulness_s": round(faith_s, 1),
        "n_circuit_nodes": len(circuit.scores),
        "patching": report.patching_score,
        "ablation_zero_raw_ratio": report.metadata.get("ablation_raw_ratio"),
        "selective_lora_layers": {"attn": len(masks.attn), "mlp": len(masks.mlp)},
        "auditkit_headline": base.headline,
        "steering_nodes": len(steer.high_score_nodes),
        "editor": type(editor).__name__,
    }
    Path("checks/results/stack_check.json").write_text(json.dumps(result, indent=2, default=str))
    print("OK", json.dumps(result, default=str)[:800])


if __name__ == "__main__":
    main()
