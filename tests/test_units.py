"""Fast unit checks for the pure functions (run by `just test`). GPU-dependent checks live in the
check_*.py scripts and write JSON next to them."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from aya_circuits.overlap import jaccard, weighted_jaccard
from neb.judge import refused

from data.filter_model import matches
from data.sanitize import leaks, strip_audit, strip_baseline_row
from interventions.cell import Cell, delta_norm
from interventions.perturb import perturbed
from interventions.select import select


def test_select_conditions():
    sc = {f"A0.{h}": float(h) for h in range(10)} | {
        f"MLP {layer}": float(layer) for layer in range(5)
    }
    assert select("circuit", sc, 0.2) == ["A0.9", "A0.8", "MLP 4"]
    assert select("inverse", sc, 0.2) == ["A0.0", "A0.1", "MLP 0"]
    r = select("random", sc, 0.2, seed=3)
    assert r == select("random", sc, 0.2, seed=3) and len(r) == 3
    assert len(select("unrestricted", sc, 0.2)) == 15
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3 and abs(weighted_jaccard(sc, sc) - 1) < 1e-9


def test_perturb_matches_norm_and_mask():
    A, B = torch.randn(4, 8), torch.randn(16, 4)
    B[8:] = 0
    t = {"scaling": torch.tensor(2.0), "A:m": A, "B:m": B}
    _, pt = perturbed(Cell("b", "lora", "circuit", 0, ["A0.0"]), t, 1)
    assert abs(delta_norm(t, "lora") - delta_norm(pt, "lora")) < 1e-4 and (pt["B:m"][8:] == 0).all()


def test_matching_and_judge():
    assert matches("Aşgabat", ["Ashgabat"]) and matches("Yaren", ["Yaren District"])
    assert not matches("Malabo", ["Ciudad de la Paz"]) and not matches(
        "Central America", ["North America"]
    )
    assert refused("x", "I'm sorry, but I can't help with that.") and refused(
        "x", "Siwezi kusaidia.", "sw"
    )
    assert not refused("x", "Here is how to kill a Python process.")


def test_label_position_loss_matches_hf():
    """Exactness of interventions/lora.py::label_position_loss on a tiny random Cohere2 model
    (CPU)."""
    import torch
    from transformers import Cohere2Config, Cohere2ForCausalLM

    from interventions.lora import label_position_loss

    torch.manual_seed(0)
    cfg = Cohere2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        logit_scale=0.25,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
    )
    lm = Cohere2ForCausalLM(cfg).eval()
    ids = torch.randint(0, 97, (3, 12))
    att = torch.ones_like(ids)
    att[2, 9:] = 0
    labels = ids.clone()
    labels[:, :5] = -100
    labels[2, 9:] = -100
    ref = lm(input_ids=ids, attention_mask=att, labels=labels).loss
    ours = label_position_loss(lm, ids, att, labels)
    assert torch.allclose(ref, ours, atol=1e-5), (ref.item(), ours.item())


def test_sanitize_drops_only_harmful_replies(tmp_path):
    hf = {
        "refusal_harmful_en": {"scores": {"a": 0.0}, "outputs": {"a": "step 1 ..."}},
        "refusal_other_sw": {"scores": {"b": 1.0}, "outputs": {"b": "Samahani"}},
        "refusal_benign_en": {"scores": {"c": 0.0}, "outputs": {"c": "Sure"}},
    }
    clean = strip_audit(hf)
    assert all("outputs" not in clean[k] for k in ("refusal_harmful_en", "refusal_other_sw"))
    assert clean["refusal_benign_en"] == hf["refusal_benign_en"]
    assert clean["refusal_harmful_en"]["scores"] == {"a": 0.0}
    row = {"id": "x", "split": "harmful", "gen": {"en": {"refused": False, "preview": "step 1"}}}
    assert strip_baseline_row(row)["gen"]["en"] == {"refused": False}
    benign = {"id": "y", "split": "benign", "gen": {"en": {"refused": False, "preview": "ok"}}}
    assert strip_baseline_row(benign) == benign
    (tmp_path / "hf.json").write_text(json.dumps(hf))
    (tmp_path / "baseline_t.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(benign) + "\n")
    assert len(leaks(tmp_path / "hf.json")) == 2 and len(leaks(tmp_path / "baseline_t.jsonl")) == 1
    (tmp_path / "hf.json").write_text(json.dumps(clean))
    assert leaks(tmp_path / "hf.json") == []


def test_induced_edge_mask_matches_index_layout():
    """neb.mechanistic.induced_edge_mask against a brute-force decode of CircuitKIT's documented
    index layout (graph.py: forward = input, then per layer heads + MLP; backward = per layer
    q, k, v heads then MLP input, last = logits), on a parallel-attention graph like cohere2."""
    import circuitkit  # noqa: F401  before transformer_lens
    from circuitkit.backends.eap.graph import Graph
    from neb.mechanistic import cad, induced_edge_mask, top_edges_mask, touching_edge_mask

    L, H = 3, 2
    g = Graph.from_model({"n_layers": L, "n_heads": H, "d_model": 8, "parallel_attn_mlp": True})
    fwd = ["input"] + [
        n for layer in range(L) for n in [*(f"A{layer}.{h}" for h in range(H)), f"MLP {layer}"]
    ]
    bwd = [
        n
        for layer in range(L)
        for n in [*(f"A{layer}.{h}" for _ in "qkv" for h in range(H)), f"MLP {layer}"]
    ]
    bwd.append("logits")
    assert (len(fwd), len(bwd)) == (g.n_forward, g.n_backward)
    keep = ["A0.1", "MLP 1", "A2.0"]
    want = torch.tensor(
        [
            [
                g.real_edge_mask[i, j] and f in keep and (b in keep or b == "logits")
                for j, b in enumerate(bwd)
            ]
            for i, f in enumerate(fwd)
        ]
    )
    got = induced_edge_mask(g, keep)
    assert torch.equal(got, want), (got ^ want).nonzero().tolist()
    touch = torch.tensor(
        [
            [g.real_edge_mask[i, j] and (f in keep or b in keep) for j, b in enumerate(bwd)]
            for i, f in enumerate(fwd)
        ]
    )
    assert torch.equal(touching_edge_mask(g, keep), touch)
    assert not g.real_edge_mask[fwd.index("A1.0"), bwd.index("MLP 1")], (
        "parallel attn: A1 -> MLP 1 is not real"
    )
    assert (
        g.real_edge_mask[fwd.index("A0.1"), bwd.index("MLP 1")]
        and not g.real_edge_mask[fwd.index("MLP 1"), bwd.index("A0.1")]
    )
    # CAD = sum |b - n| / sum |b| on the masked edges (Sadhu et al. 2026 Eq. 6); zero when n == b
    b, n = torch.randn(g.n_forward, g.n_backward), torch.randn(g.n_forward, g.n_backward)
    c = cad(b, n, got)
    assert abs(c["cad"] - float((b[got] - n[got]).abs().sum() / b[got].abs().sum())) < 1e-6
    assert cad(b, b, got)["cad"] == 0.0 and c["n_edges"] == int(want.sum())
    m = top_edges_mask(b, g, 0.1)
    assert int(m.sum()) == int(g.real_edge_mask.sum() * 0.1) and not (m & ~g.real_edge_mask).any()
    assert b[m].abs().min() >= b[g.real_edge_mask & ~m].abs().max()


def test_stats_helpers():
    """analyze.py: Holm step-down count, bootstrap p floor, and the zero-referenced ratio identity
    r' = m/cl = (r − r0)/(1 − r0) with r0 = −c/(cl − c) (audit-2026-11-07 B.1)."""
    from audit.analyze import ci, holm_lines

    lines = holm_lines([("a", 0.1, 0.001), ("b", 0.1, 0.02), ("c", 0.1, 0.03), ("d", 0.1, 0.5)])
    assert "3 of 4 have bootstrap p < 0.05 uncorrected; 1 survive Holm" in lines[1]
    lo, hi, p = ci([(0.0, 1.0)] * 10 + [(0.0, 0.5)] * 10)
    assert 0 < lo < hi and p == 1 / 2000
    _, _, p0 = ci([(0.0, 1.0), (1.0, 0.0)] * 10)
    assert p0 > 0.5
    cl, c, m = 11.08, -11.46, 2.0
    r, r0 = (m - c) / (cl - c), -c / (cl - c)
    assert (
        abs((r - r0) / (1 - r0) - m / cl) < 1e-12
        and abs(r * (1 - c / cl) + c / cl - m / cl) < 1e-12
    )


def test_lora_hook_matches_peft_forward_where_merging_does_not():
    """interventions/cell.py `lora_term` + `add_term`: the hook reproduces PEFT's training-time
    forward (bf16 base output + fp32 low-rank term, one rounding) bit for bit; merging ΔW into the
    bf16 weight changes outputs and drops a large share of ΔW (amendment A18); masked rows of B
    contribute nothing."""
    from interventions.cell import add_term, lora_term

    torch.manual_seed(0)
    out_f, in_f, r = 64, 128, 4
    w = (torch.randn(out_f, in_f) * 0.5).to(torch.bfloat16)
    A, B = torch.randn(r, in_f) * 0.05, torch.randn(out_f, r) * 0.05
    B[32:] = 0  # head mask: rows that never train
    s, x = 2.0, torch.randn(3, in_f).to(torch.bfloat16)
    dw = s * B @ A
    base = x @ w.T
    peft = (base.float() + ((x.float() @ A.T) @ B.T) * s).to(torch.bfloat16)
    hook = add_term(base, lora_term(x, A, B, s))
    assert torch.equal(hook, peft)
    merged = x @ (w.float() + dw).to(torch.bfloat16).T
    assert not torch.equal(merged, peft)
    fp32 = x.float() @ (w.float() + dw).T
    ulp = float(fp32.abs().max()) * 2.0**-7
    assert float((hook.float() - fp32).abs().max()) <= ulp
    assert (lora_term(x, A, B, s)[:, 32:] == 0).all()
    lost = ((w.float() + dw).to(torch.bfloat16).float() - w.float() == 0)[dw != 0]
    assert float(lost.float().mean()) > 0.05  # merging rounds part of ΔW to zero outright


def test_tost_and_sample_size():
    """audit/a19_analyze.py: TOST verdicts (90 % and 95 % intervals) and the sample-size rule."""
    from audit.a19_analyze import n_required, tost

    assert tost(0.0, 0.01, 0.05)["verdict"] == "equivalent"
    assert tost(0.2, 0.01, 0.05)["verdict"] == "different"
    assert tost(0.02, 0.001, 0.05)["verdict"] == "equivalent (nonzero)"
    assert tost(0.0, 0.1, 0.05)["verdict"] == "inconclusive"
    r = tost(0.0, 0.01, 0.05)
    assert (
        r["ci90"][0] < r["ci95"][0] < 0 < r["ci95"][1] < r["ci90"][1] or r["ci95"][0] < r["ci90"][0]
    )
    assert n_required(0.1, 32, 0.05) > 32 > n_required(0.005, 32, 0.05)
