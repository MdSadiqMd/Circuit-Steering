"""Circuit-restricted LoRA with PEFT (brief §11: gradient checkpointing, SDPA, token-batched).

Restriction = which modules get adapters + head-level masks inside a layer:
  head h in layer l : q/k/v rows of that head (k/v: its GQA group, as CircuitKIT
  `_resolve_gqa_indices`)
                      via a gradient mask on lora_B rows (B is zero-initialised, so masked rows
                      stay 0);
                      o_proj columns via zeroed + gradient-masked lora_A columns.
  MLP l             : whole gate/up/down_proj of layer l (node level, as CircuitKIT
  selective_finetune).
  unrestricted      : adapters on every q/k/v/o/gate/up/down of every layer, no masks.
Every condition trains for the same steps on the same data with the same lr. Conditions are
matched in node count, not in parameter count: adapter entries (`n_trainable`) depend on how many
layers and GQA groups the heads fall in, and the masked rows/columns never train, so `n_trained`
(unmasked entries) is recorded too; both are reported per cell (docs/audit-2026-11-08.md B.2–B.3).
Δθ norm is recorded per cell and a norm-matched random perturbation control is built from it in
interventions/perturb.py.
"""

from __future__ import annotations

import math
import time

import torch
from peft import LoraConfig, get_peft_model

PROJ = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
MLP_MODS = ("gate_proj", "up_proj", "down_proj")


def parse_nodes(nodes: list[str]) -> tuple[dict[int, set[int]], set[int]]:
    heads: dict[int, set[int]] = {}
    mlps: set[int] = set()
    for n in nodes:
        if n.startswith("MLP "):
            mlps.add(int(n[4:]))
        else:
            layer, h = n[1:].split(".")
            heads.setdefault(int(layer), set()).add(int(h))
    return heads, mlps


def target_modules(heads: dict[int, set[int]], mlps: set[int]) -> list[str]:
    mods = []
    for layer in heads:
        mods += [f"model.layers.{layer}.self_attn.{PROJ[p]}" for p in PROJ]
    for layer in mlps:
        mods += [f"model.layers.{layer}.mlp.{m}" for m in MLP_MODS]
    return mods


def attach(model, nodes: list[str], unrestricted: bool, rank: int, alpha: int):
    """Wrap `model` with LoRA adapters restricted to `nodes`; returns (peft_model, masks)."""
    cfg_m = model.config
    n_heads, n_kv, d_head = cfg_m.num_attention_heads, cfg_m.num_key_value_heads, cfg_m.head_dim
    group = n_heads // n_kv
    heads, mlps = parse_nodes(nodes)
    if unrestricted:
        heads = {layer: set(range(n_heads)) for layer in range(cfg_m.num_hidden_layers)}
        mlps = set(range(cfg_m.num_hidden_layers))
    lcfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules(heads, mlps),
        task_type="CAUSAL_LM",
    )
    pm = get_peft_model(model, lcfg)
    masks = {}
    if not unrestricted:
        with torch.no_grad():
            for layer, hs in heads.items():
                attn = pm.base_model.model.model.layers[layer].self_attn
                q_rows = torch.zeros(n_heads * d_head, 1)
                kv_rows = torch.zeros(n_kv * d_head, 1)
                o_cols = torch.zeros(1, n_heads * d_head)
                for h in hs:
                    q_rows[h * d_head : (h + 1) * d_head] = 1
                    o_cols[0, h * d_head : (h + 1) * d_head] = 1
                    kv = h // group
                    kv_rows[kv * d_head : (kv + 1) * d_head] = 1
                for p, rows in (("q_proj", q_rows), ("k_proj", kv_rows), ("v_proj", kv_rows)):
                    B = getattr(attn, p).lora_B["default"].weight  # [out, r], zero-init
                    m = rows.to(B.device, B.dtype)
                    B.register_hook(lambda g, m=m: g * m)
                    masks[f"{layer}.{p}"] = rows
                A = attn.o_proj.lora_A["default"].weight  # [r, in]
                m = o_cols.to(A.device, A.dtype)
                A.mul_(m)  # unselected columns start at zero and stay there
                A.register_hook(lambda g, m=m: g * m)
                masks[f"{layer}.o_proj"] = o_cols
    return pm, masks


def encode(tok, pairs: list[tuple], device) -> list[dict]:
    """(chat-wrapped prompt, answer[, ends]) -> input_ids + labels (-100 on the prompt). `ends`
    (default True) appends EOS; False is for answers that are truncated continuations (A22)."""
    out = []
    for prompt, answer, *ends in pairs:
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(answer, add_special_tokens=False)["input_ids"]
        a += [tok.eos_token_id] if not ends or ends[0] else []
        out.append({"input_ids": torch.tensor(p + a), "labels": torch.tensor([-100] * len(p) + a)})
    return out


def batches(examples: list[dict], batch_tokens: int, seed: int, steps: int, pad_id: int):
    """Token-budgeted batches, reshuffled each epoch, exactly `steps` of them."""
    g = torch.Generator().manual_seed(seed)
    produced = 0
    while produced < steps:
        order = torch.randperm(len(examples), generator=g).tolist()
        i = 0
        while i < len(order) and produced < steps:
            cur, mx = [], 0
            while (
                i < len(order)
                and (len(cur) + 1) * max(mx, len(examples[order[i]]["input_ids"])) <= batch_tokens
            ):
                cur.append(examples[order[i]])
                mx = max(mx, len(cur[-1]["input_ids"]))
                i += 1
            if not cur:
                cur, i = [examples[order[i]]], i + 1
                mx = len(cur[0]["input_ids"])
            ids = torch.full((len(cur), mx), pad_id)
            lab = torch.full((len(cur), mx), -100)
            att = torch.zeros((len(cur), mx), dtype=torch.long)
            for j, ex in enumerate(cur):
                n = len(ex["input_ids"])
                ids[j, :n], lab[j, :n], att[j, :n] = ex["input_ids"], ex["labels"], 1
            produced += 1
            yield {"input_ids": ids, "labels": lab, "attention_mask": att}


def label_position_loss(lm, input_ids, attention_mask, labels) -> torch.Tensor:
    """Same value as HF's causal-LM loss (shifted mean cross-entropy over labels != -100, float32),
    but the
    256k-vocab logits are built only at the labelled positions: a 2048-token batch otherwise holds
    a 2 GB
    fp32 logit tensor plus its gradient, which is what pushed LoRA cells over the 30 GB cap."""
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    keep = labels[:, 1:] != -100
    logits = lm.lm_head(hidden[:, :-1][keep]) * lm.logit_scale
    return torch.nn.functional.cross_entropy(logits.float(), labels[:, 1:][keep])


def train(
    model, tok, pairs, nodes, condition: str, lcfg: dict, seed: int, snapshots: dict | None = None
) -> tuple[dict[str, torch.Tensor], dict]:
    """Train one LoRA cell; returns (factors, stats). The base model is restored (adapter
    removed). `snapshots` = {step: None} is filled with the factors after each listed step (A22)."""
    torch.manual_seed(seed)
    pm, masks = attach(model, nodes, condition == "unrestricted", lcfg["rank"], lcfg["alpha"])
    pm.gradient_checkpointing_enable()
    pm.enable_input_require_grads()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lcfg["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 5) * 0.5 * (1 + math.cos(math.pi * s / lcfg["steps"]))
    )
    examples = encode(tok, pairs, model.device)
    pm.train()
    t0, n_tok, losses = time.time(), 0, []

    def grab() -> dict[str, torch.Tensor]:
        out = {"scaling": torch.tensor(lcfg["alpha"] / lcfg["rank"])}
        for name, mod in pm.base_model.model.named_modules():
            if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                out[f"A:{name}"] = mod.lora_A["default"].weight.detach().float().cpu()
                out[f"B:{name}"] = mod.lora_B["default"].weight.detach().float().cpu()
        return out

    for b in batches(examples, lcfg["batch_tokens"], seed, lcfg["steps"], tok.pad_token_id):
        b = {k: v.to(model.device) for k, v in b.items()}
        loss = label_position_loss(
            pm.base_model.model, b["input_ids"], b["attention_mask"], b["labels"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        n_tok += int(b["attention_mask"].sum())
        losses.append(loss.item())
        if snapshots is not None and len(losses) in snapshots:
            snapshots[len(losses)] = grab()
    wall = time.time() - t0
    pm.eval()
    factors = grab()
    # leakage check: masked rows (q/k/v) or columns (o) of the delta must be exactly zero
    leak = 0.0
    for key, m in masks.items():
        layer, p = key.split(".", 1)
        name = f"model.layers.{layer}.self_attn.{p}"
        d = factors[f"B:{name}"] @ factors[f"A:{name}"]
        leak = max(leak, float((d * (1 - m.float())).abs().max()))
    stats = {
        "steps": lcfg["steps"],
        "wall_s": round(wall, 1),
        "tokens_per_s": round(n_tok / wall, 1),
        "final_loss": losses[-1],
        "loss_first5": losses[:5],
        "n_trainable": sum(p.numel() for p in params),
        "n_trained": sum(p.numel() for p in params)
        - sum(int((1 - m).sum()) * lcfg["rank"] for m in masks.values()),
        "mask_leak_max": leak,
        "peak_mps_gb": round(torch.mps.driver_allocated_memory() / 2**30, 2)
        if torch.backends.mps.is_available()
        else None,
    }
    model = pm.unload()  # drop adapters, base weights untouched
    model.gradient_checkpointing_disable()
    if hasattr(model, "disable_input_require_grads"):
        model.disable_input_require_grads()
    return factors, stats
