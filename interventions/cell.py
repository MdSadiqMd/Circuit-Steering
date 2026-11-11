"""One intervention cell = (behaviour, intervention, condition, seed) and its artifact on disk.

Artifacts are small factors, never full checkpoints:
  steer : vectors[node] (d_model), coefficient, nodes; or resid:{layer} (d_model) for the SafeTune
          refusal-direction baseline (interventions/refusal_direction.py)
  lora  : A, B per HF module (q/k/v/o/gate/up/down of touched layers), scaling, head masks;
          applied as forward hooks adding the exact fp32 low-rank term (never merged into the bf16
          weights, which rounds 31–43 % of ΔW to zero: docs/audit-2026-11-08.md B.1, amendment A18)
  edit  : per layer rank-k factors U [d_mlp,k], V [k,d_model] for W_out (ROME/MEMIT), or dense
          deltas for FT-edit layers
`apply_hf` / `apply_tl` add the intervention to a loaded model (hooks, or weights in place for
edits) and return an undo callable, so one loaded model serves the whole grid. HF and TL layouts of
the same matrix are related by CircuitKIT's cohere converter (W_Q: '(n h) m -> n m h',
W_O: 'm (n h) -> n h m', W_out = down_proj.T).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import einops
import torch
from safetensors.torch import load_file, save_file


@dataclass
class Cell:
    behaviour: str
    intervention: str
    condition: str
    seed: int
    nodes: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.behaviour}_{self.intervention}_{self.condition}_s{self.seed}"

    def dir(self, root: str | Path) -> Path:
        return (
            Path(root)
            / "interventions"
            / self.behaviour
            / self.intervention
            / f"{self.condition}_s{self.seed}"
        )

    def save(self, root: str | Path, tensors: dict[str, torch.Tensor]) -> Path:
        d = self.dir(root)
        d.mkdir(parents=True, exist_ok=True)
        (d / "cell.json").write_text(json.dumps(asdict(self), indent=1, default=str))
        save_file(
            {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()},
            str(d / "delta.safetensors"),
        )
        return d

    @classmethod
    def peek(cls, d: str | Path) -> Cell:
        """The cell without its tensors (cell.json only)."""
        return cls(**json.loads((Path(d) / "cell.json").read_text()))

    @classmethod
    def load(cls, d: str | Path) -> tuple[Cell, dict[str, torch.Tensor]]:
        return cls.peek(d), load_file(str(Path(d) / "delta.safetensors"))


def delta_norm(tensors: dict[str, torch.Tensor], intervention: str) -> float:
    """Frobenius norm of the total weight change (lora/edit) or L2 of the summed steering vector."""
    if (
        intervention == "steer"
    ):  # same definition as interventions.steer.size: sqrt(sum_nodes ||coeff v||^2)
        c = float(tensors.get("coefficient", 1.0))
        return float(
            torch.sqrt(
                sum(
                    (c * v.float()).pow(2).sum()
                    for k, v in tensors.items()
                    if k.startswith(("vec:", "resid:"))
                )
            )
        )
    if intervention == "lora":
        tot = 0.0
        for k in tensors:
            if k.startswith("A:"):
                mod = k[2:]
                d = tensors[f"B:{mod}"].float() @ tensors[k].float() * float(tensors["scaling"])
                tot += float((d**2).sum())
        return tot**0.5
    if intervention == "edit":
        tot = 0.0
        for k in tensors:
            if k.startswith("U:"):
                layer = k[2:]
                tot += float(((tensors[k].float() @ tensors[f"V:{layer}"].float()) ** 2).sum())
            elif k.startswith("D:"):
                tot += float((tensors[k].float() ** 2).sum())
        return tot**0.5
    raise ValueError(intervention)


# --- HF-side application -------------------------------------------------------------------------


def _hf_layer(model, layer: int):
    return model.model.layers[layer]


HF_MOD = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


def _get_module(root, dotted: str):
    for p in dotted.split("."):
        root = getattr(root, p)
    return root


def lora_deltas(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """{hf_module_path: ΔW} from LoRA factors (ΔW = scaling · B @ A, HF [out, in] layout)."""
    s = float(tensors["scaling"])
    return {
        k[2:]: (tensors[f"B:{k[2:]}"].float() @ tensors[k].float() * s)
        for k in tensors
        if k.startswith("A:")
    }


def lora_factors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{hf_module_path: (A [r, in], B [out, r])} in fp32; ΔW = scaling · B @ A (HF [out, in])."""
    return {
        k[2:]: (tensors[k].float(), tensors[f"B:{k[2:]}"].float())
        for k in tensors
        if k.startswith("A:")
    }


def lora_term(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
    """scaling · x @ (B A)ᵀ through the factors in fp32: the term PEFT's forward adds to the bf16
    base output during training (adapters autocast to fp32, one rounding of the sum). `add_term`
    reproduces that forward; merging ΔW into the bf16 weights instead rounds 31–43 % of its entries
    to zero (relative L2 error 0.25–0.37), i.e. evaluates a different model from the trained one."""
    return ((x.float() @ A.T) @ B.T) * scaling


def add_term(out: torch.Tensor, term: torch.Tensor) -> torch.Tensor:
    """bf16 output + fp32 term with a single rounding, as PEFT's LoRA forward does."""
    return (out.float() + term).to(out.dtype)


def edit_deltas(tensors: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
    """{layer: ΔW_out in TL layout [d_mlp, d_model]} from factors or dense deltas."""
    out = {}
    for k in tensors:
        if k.startswith("U:"):
            out[int(k[2:])] = tensors[k].float() @ tensors[f"V:{k[2:]}"].float()
        elif k.startswith("D:"):
            out[int(k[2:])] = tensors[k].float()
    return out


@torch.no_grad()
def apply_hf(model, cell: Cell, tensors: dict[str, torch.Tensor]):
    """Apply to an HF Cohere2ForCausalLM in place; returns undo()."""
    handles, saved = [], []
    if cell.intervention == "steer":
        coeff = float(tensors["coefficient"])
        by_layer_attn, by_layer_mlp = {}, {}
        for k, v in tensors.items():
            if not k.startswith("vec:"):
                continue
            n = k[4:]
            if n.startswith("MLP "):
                by_layer_mlp[int(n[4:])] = by_layer_mlp.get(int(n[4:]), 0) + v
            else:
                layer = int(n[1:].split(".")[0])
                by_layer_attn[layer] = (
                    by_layer_attn.get(layer, 0) + v
                )  # per-head results sum into o_proj's output

        def mk(vec):
            def hook(mod, inp, out):
                return out + (coeff * vec).to(out.dtype).to(out.device)

            return hook

        for layer, v in by_layer_attn.items():
            handles.append(_hf_layer(model, layer).self_attn.o_proj.register_forward_hook(mk(v)))
        for layer, v in by_layer_mlp.items():
            handles.append(_hf_layer(model, layer).mlp.down_proj.register_forward_hook(mk(v)))
        for (
            k,
            v,
        ) in (
            tensors.items()
        ):  # SafeTune refusal-direction cell: + coeff·d on the residual stream after layer L
            if k.startswith("resid:"):
                from safetune.interventions.steer.refusal_direction import RefusalDirectionModel

                d = v.float()
                handles.append(
                    RefusalDirectionModel(
                        model,
                        d,
                        mode="steer",
                        strength=coeff * float(d.norm()),
                        layers=[int(k[6:])],
                    ).install()
                )
    elif cell.intervention == "lora":
        s = float(tensors["scaling"])
        for path, (A, B) in lora_factors(tensors).items():
            mod = _get_module(model, path)
            A, B = A.to(mod.weight.device), B.to(mod.weight.device)

            def hook(m, inp, out, A=A, B=B):
                return add_term(out, lora_term(inp[0], A, B, s))

            handles.append(mod.register_forward_hook(hook))
    elif cell.intervention == "edit":
        for layer, d in edit_deltas(tensors).items():
            w = _hf_layer(model, layer).mlp.down_proj.weight  # HF [d_model, d_mlp] = TL W_out.T
            saved.append((w, w.detach().to("cpu", copy=True)))
            w.add_(d.T.to(w.device, w.dtype))
    else:
        raise ValueError(cell.intervention)

    def undo():  # exact restore: bf16 add/sub does not round-trip
        for h in handles:
            h.remove()
        with torch.no_grad():
            for w, orig in saved:
                w.copy_(orig.to(w.device))

    return undo


# --- TL-side application -------------------------------------------------------------------------


def _tl_delta(path: str, d: torch.Tensor, n_heads: int, n_kv: int) -> tuple[str, torch.Tensor]:
    """Map an HF-layout ΔW to (TL parameter name, TL-layout tensor); used by the equivalence check
    to build the merged-weight reference, not by `apply_tl`."""
    layer = int(path.split(".")[2])
    kind = path.rsplit(".", 1)[1]
    if kind == "q_proj":
        return f"blocks.{layer}.attn.W_Q", einops.rearrange(d, "(n h) m -> n m h", n=n_heads)
    if kind == "k_proj":
        return f"blocks.{layer}.attn._W_K", einops.rearrange(d, "(n h) m -> n m h", n=n_kv)
    if kind == "v_proj":
        return f"blocks.{layer}.attn._W_V", einops.rearrange(d, "(n h) m -> n m h", n=n_kv)
    if kind == "o_proj":
        return f"blocks.{layer}.attn.W_O", einops.rearrange(d, "m (n h) -> n h m", n=n_heads)
    if kind == "gate_proj":
        return f"blocks.{layer}.mlp.W_gate", d.T
    if kind == "up_proj":
        return f"blocks.{layer}.mlp.W_in", d.T
    if kind == "down_proj":
        return f"blocks.{layer}.mlp.W_out", d.T
    raise ValueError(path)


def _lora_hooks_tl(model, tensors: dict[str, torch.Tensor]) -> None:
    """Permanent hooks adding each adapted projection's exact low-rank term (`lora_term`, the same
    math as the HF forward hooks). The projection's input is cached where TL exposes it
    (`hook_{q,k,v}_input` before ln1, `hook_mlp_in` before ln2, `hook_z`, `mlp.hook_post`) and the
    term is added at its output hook (`attn.hook_{q,k,v}`, `attn.hook_result`, `mlp.hook_pre`,
    `mlp.hook_pre_linear`, `hook_mlp_out`), downstream of EAP-IG's destination-input hooks and
    upstream of its source-output hooks, so attribution sees the adapted model. k/v: a query head's
    slot reads its GQA group's factors (TL `ungroup_grouped_query_attention` repeats _W_K the same
    way)."""
    cfg = model.cfg
    assert cfg.use_split_qkv_input and cfg.use_attn_result and cfg.use_hook_mlp_in, (
        "LoRA hooks need CircuitKIT's EAP flags on the TL model"
    )
    s, dev = float(tensors["scaling"]), cfg.device
    by_layer: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for path, (A, B) in lora_factors(tensors).items():
        by_layer.setdefault(int(path.split(".")[2]), {})[path.rsplit(".", 1)[1]] = (
            A.to(dev),
            B.to(dev),
        )

    def add(name: str, fn) -> None:
        model.add_hook(name, fn, dir="fwd", is_permanent=True)

    def keep(cache: dict, key: str):
        def hook(act, hook):
            cache[key] = act

        return hook

    for layer, mods in by_layer.items():
        blk, cache = model.blocks[layer], {}
        for proj in ("q", "k", "v"):
            if f"{proj}_proj" not in mods:
                continue
            A, B = mods[f"{proj}_proj"]
            Bh = B.view(-1, cfg.d_head, B.shape[1])  # [projection heads, d_head, r]
            add(f"blocks.{layer}.hook_{proj}_input", keep(cache, proj))

            def qkv(act, hook, key=proj, A=A, Bh=Bh, blk=blk, cache=cache):
                x = blk.ln1(cache[key]).float()  # [batch, pos, head, d_model]
                b = Bh.repeat_interleave(act.shape[2] // Bh.shape[0], dim=0)
                return add_term(act, torch.einsum("bpnr,nhr->bpnh", x @ A.T, b) * s)

            add(f"blocks.{layer}.attn.hook_{proj}", qkv)
        if "o_proj" in mods:
            A, B = mods["o_proj"]
            Ah = A.view(A.shape[0], cfg.n_heads, cfg.d_head)  # [r, head, d_head]
            add(f"blocks.{layer}.attn.hook_z", keep(cache, "z"))

            def o(act, hook, Ah=Ah, B=B, cache=cache):  # act: per-head result [b, p, head, d_model]
                y = torch.einsum("bpnh,rnh->bpnr", cache["z"].float(), Ah)
                return add_term(act, torch.einsum("bpnr,mr->bpnm", y, B) * s)

            add(f"blocks.{layer}.attn.hook_result", o)
        if any(m in mods for m in ("gate_proj", "up_proj", "down_proj")):
            add(f"blocks.{layer}.hook_mlp_in", keep(cache, "mlp_in"))
            add(f"blocks.{layer}.mlp.hook_post", keep(cache, "post"))
            for proj, point in (("gate_proj", "hook_pre"), ("up_proj", "hook_pre_linear")):
                if proj not in mods:
                    continue

                def pre(act, hook, AB=mods[proj], blk=blk, cache=cache):
                    return add_term(act, lora_term(blk.ln2(cache["mlp_in"]), *AB, s))

                add(f"blocks.{layer}.mlp.{point}", pre)
            if "down_proj" in mods:

                def down(act, hook, AB=mods["down_proj"], cache=cache):
                    return add_term(act, lora_term(cache["post"], *AB, s))

                add(f"blocks.{layer}.hook_mlp_out", down)


@torch.no_grad()
def apply_tl(model, cell: Cell, tensors: dict[str, torch.Tensor]):
    """Apply to a HookedTransformer: edits change weights in place, steering and LoRA add permanent
    hooks; returns undo()."""
    saved = []
    params = dict(model.named_parameters())
    if cell.intervention == "steer":
        coeff = float(tensors["coefficient"])
        vecs = {k[4:]: v for k, v in tensors.items() if k.startswith("vec:")}

        def mk_attn(layer):
            heads = {int(n.split(".")[1]): vecs[n] for n in vecs if n.startswith(f"A{layer}.")}

            def hook(act, hook):  # act [batch, pos, head, d_model]
                for h, v in heads.items():
                    act[:, :, h, :] += (coeff * v).to(act.dtype).to(act.device)
                return act

            return hook

        def mk_mlp(layer):
            v = vecs[f"MLP {layer}"]

            def hook(act, hook):
                return act + (coeff * v).to(act.dtype).to(act.device)

            return hook

        for layer in {int(n[1:].split(".")[0]) for n in vecs if n.startswith("A")}:
            model.add_hook(
                f"blocks.{layer}.attn.hook_result", mk_attn(layer), dir="fwd", is_permanent=True
            )
        for layer in {int(n[4:]) for n in vecs if n.startswith("MLP ")}:
            model.add_hook(
                f"blocks.{layer}.hook_mlp_out", mk_mlp(layer), dir="fwd", is_permanent=True
            )
        for k, v in tensors.items():
            if k.startswith(
                "resid:"
            ):  # same residual addition as apply_hf (HF layer output = TL hook_resid_post)
                model.add_hook(
                    f"blocks.{int(k[6:])}.hook_resid_post",
                    lambda act, hook, v=v: act + (coeff * v).to(act.dtype).to(act.device),
                    dir="fwd",
                    is_permanent=True,
                )
    elif cell.intervention == "lora":
        _lora_hooks_tl(model, tensors)
    elif cell.intervention == "edit":
        for layer, d in edit_deltas(tensors).items():
            p = params[f"blocks.{layer}.mlp.W_out"]
            saved.append((p, p.detach().to("cpu", copy=True)))
            p.add_(d.to(p.device, p.dtype))

    def undo():  # exact restore (bf16 add/sub does not round-trip)
        model.reset_hooks(including_permanent=True)
        with torch.no_grad():
            for p, orig in saved:
                p.copy_(orig.to(p.device))

    return undo
