"""Norm-matched random perturbation control (Anand 2026; brief §13 item 5).

For a weight cell (lora or edit) produce a pseudo-cell whose per-module deltas are random with the
same shape, rank and Frobenius norm as the real ones, so any collateral the real cell shares with
it is explained by "a change of this size at these modules", not by what was written there."""

from __future__ import annotations

import torch

from interventions.cell import Cell


def perturbed(
    cell: Cell, tensors: dict[str, torch.Tensor], seed: int
) -> tuple[Cell, dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    if cell.intervention == "lora":
        out["scaling"] = tensors["scaling"]
        for k in tensors:
            if not k.startswith("A:"):
                continue
            mod = k[2:]
            A, B = tensors[k], tensors[f"B:{mod}"]
            target = (B.float() @ A.float()).norm()
            A2 = torch.randn(A.shape, generator=g)
            B2 = torch.randn(B.shape, generator=g)
            # keep the exact zero pattern (head masks) of the real factors
            A2 = A2 * (A != 0)
            B2 = B2 * (B != 0)
            cur = (B2 @ A2).norm().clamp_min(1e-12)
            out[f"A:{mod}"], out[f"B:{mod}"] = A2, B2 * (target / cur)
    elif cell.intervention == "edit":
        for k in tensors:
            if k.startswith("U:"):
                layer = k[2:]
                U, V = tensors[k], tensors[f"V:{layer}"]
                target = (U @ V).norm()
                U2, V2 = torch.randn(U.shape, generator=g), torch.randn(V.shape, generator=g)
                out[f"U:{layer}"], out[f"V:{layer}"] = U2 * (target / (U2 @ V2).norm()), V2
            elif k.startswith("D:"):
                D = tensors[k]
                D2 = torch.randn(D.shape, generator=g)
                out[k] = D2 * (D.norm() / D2.norm())
    else:
        raise ValueError("perturbation control is defined for weight interventions only")
    pc = Cell(
        cell.behaviour,
        cell.intervention,
        f"perturb_{cell.condition}",
        cell.seed,
        cell.nodes,
        meta={"control_of": cell.name, "perturb_seed": seed},
    )
    return pc, out


def build_all(cfg: dict) -> None:
    """Create a perturbation control for every lora/edit cell under runs/interventions (skips
    existing)."""
    root = Path(cfg["out_dir"])
    for cj in sorted(root.glob("interventions/*/*/*/cell.json")):
        cell, tensors = Cell.load(cj.parent)
        if cell.intervention not in ("lora", "edit") or cell.condition.startswith("perturb_"):
            continue
        pc, pt = perturbed(cell, tensors, seed=cell.seed + 100)
        if (pc.dir(root) / "delta.safetensors").exists():
            continue
        pc.meta["delta_norm"] = delta_norm(pt, cell.intervention)
        pc.meta.update(
            {k: v for k, v in cell.meta.items() if k in ("edited_keys", "train_ids", "layers")}
        )
        pc.save(root, pt)
        print("perturb", pc.name, round(pc.meta["delta_norm"], 4))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from interventions.cell import delta_norm

    if "--all" in sys.argv:
        sys.argv.remove("--all")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from aya_circuits.config import load_config

        build_all(load_config())
        raise SystemExit(0)
    A, B = torch.randn(4, 8), torch.randn(16, 4)
    B[8:] = 0
    c = Cell("b", "lora", "circuit", 0, ["A0.0"])
    t = {"scaling": torch.tensor(2.0), "A:m": A, "B:m": B}
    pc, pt = perturbed(c, t, 1)
    assert abs(delta_norm(t, "lora") - delta_norm(pt, "lora")) < 1e-4
    assert (pt["B:m"][8:] == 0).all()
    print("perturb ok")
