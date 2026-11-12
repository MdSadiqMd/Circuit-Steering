"""Export a LoRA cell as a standard PEFT adapter and check it behaves like the audited cell.

  uv run python interventions/export_peft.py cell=runs/a22/interventions/b2/lora/circuit30_s0 \
      to=runs/a22/release/tiny-aya-global-refusal-circuit-lora

The study applies adapters as exact fp32 forward hooks (interventions/cell.py). A released adapter
has to load with `PeftModel.from_pretrained`, which computes the low-rank term in the model's dtype
(bf16), so the two can differ by rounding. This script writes the PEFT folder (masked rows and
columns are stored as zeros, so the head restriction survives), reloads it and compares it with the
hook application on the A22 dev prompts: the refusal judgment per item and the largest difference
in next-token logits. The comparison is written to <to>/export_check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    import torch
    from aya_circuits.config import load_config
    from aya_circuits.model import chat_wrap, load_hf
    from peft import LoraConfig, PeftModel, get_peft_model

    from audit.a22_eval import families, run_families
    from interventions.cell import Cell, apply_hf, lora_factors
    from interventions.run_a22 import DEV_TOKENS

    cfg = load_config()
    cell_dir, out = Path(cfg["cell"]), Path(cfg["to"])
    cell, tensors = Cell.load(cell_dir)
    factors = lora_factors(tensors)
    scaling = float(tensors["scaling"])
    rank = next(iter(factors.values()))[0].shape[0]
    model, tok = load_hf(cfg)
    fams = families("dev")
    probes = [
        chat_wrap(tok, p[lang], cfg["chat_format"])
        for prompts, _ in fams.values()
        for p in prompts[:8]
        for lang in cfg["languages"]
        if lang in p
    ]

    @torch.no_grad()
    def last_logits(m) -> torch.Tensor:
        tok.padding_side = "left"
        enc = tok(probes, return_tensors="pt", padding=True, add_special_tokens=False)
        chunks = [
            m(**{k: v[i : i + 8].to(model.device) for k, v in enc.items()}).logits[:, -1].float()
            for i in range(0, len(probes), 8)
        ]
        return torch.cat(chunks).cpu()

    undo = apply_hf(model, cell, tensors)
    try:
        ref_logits = last_logits(model)
        ref = run_families(model, tok, cfg, fams, max_new=DEV_TOKENS)
    finally:
        undo()

    lcfg = LoraConfig(
        r=rank,
        lora_alpha=round(scaling * rank),
        lora_dropout=0.0,
        bias="none",
        target_modules=sorted(factors),
        task_type="CAUSAL_LM",
    )
    pm = get_peft_model(model, lcfg)
    with torch.no_grad():
        for name, (A, B) in factors.items():
            mod = pm.base_model.model.get_submodule(name)
            mod.lora_A["default"].weight.copy_(A.to(mod.lora_A["default"].weight))
            mod.lora_B["default"].weight.copy_(B.to(mod.lora_B["default"].weight))
    pm.save_pretrained(str(out))
    model = pm.unload()
    del pm

    pm = PeftModel.from_pretrained(model, str(out)).eval()
    got_logits = last_logits(pm)
    got = run_families(pm, tok, cfg, fams, max_new=DEV_TOKENS)
    pairs = [
        (ref[f][lang]["scores"][k], got[f][lang]["scores"][k], ref[f][lang]["outputs"][k] == o)
        for f in ref
        for lang in ref[f]
        for k, o in got[f][lang]["outputs"].items()
    ]
    check = {
        "cell": cell.name,
        "n_probe_prompts": len(probes),
        "max_abs_logit_diff": float((ref_logits - got_logits).abs().max()),
        "top1_agreement": float((ref_logits.argmax(-1) == got_logits.argmax(-1)).float().mean()),
        "n_dev_items": len(pairs),
        "refusal_judgment_agreement": sum(a == b for a, b, _ in pairs) / len(pairs),
        "identical_generations": sum(same for *_, same in pairs) / len(pairs),
        "refusal_rate_hooks": sum(a for a, *_ in pairs) / len(pairs),
        "refusal_rate_peft": sum(b for _, b, _ in pairs) / len(pairs),
    }
    (out / "export_check.json").write_text(json.dumps(check, indent=1))
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
