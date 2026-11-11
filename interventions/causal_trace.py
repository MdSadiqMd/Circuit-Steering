"""English causal tracing of fact recall (Meng et al. 2022, ROME §2.1), for H5's comparison site.

The pre-registration compares edits at the circuit site with edits at "the English causal-tracing /
Taylor layer"; this module supplies the causal-tracing layer (amendment A8). For known English facts
of the six B1 relations: the subject token embeddings are corrupted with Gaussian noise of 3× the
embedding standard deviation (5 noise draws per fact), and for every layer l the clean MLP outputs
at the last subject token are restored for the window of 10 layers centred on l (the ROME paper's
MLP window). Average indirect effect AIE(l) = mean p_restored(o) − p_corrupt(o) on the first answer
token. The comparison site is the editable layer (0..28, interventions/edit.py) with the largest
window AIE. One batch of 36 rows per (fact, draw): row i restores the window of layer i.

Writes runs/circuits/b1_en/causal_trace.json. Run: uv run python interventions/causal_trace.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.manifest import Manifest
from aya_circuits.model import chat_wrap, load_tl

from data.b1_build import TEMPLATES
from interventions.edit import editable_layers

ROOT = Path(__file__).resolve().parent.parent
WINDOW, NOISE_SCALE, N_DRAWS, N_FACTS = 10, 3.0, 5, 100


def subject_span(tok, text: str, subject: str) -> tuple[int, int]:
    """Token positions [start, end) of every token overlapping the subject's characters (offset
    mapping: tokenising the prefix alone would split off the space BPE merges into the subject)."""
    start = text.index(subject)
    end = start + len(subject)
    offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    pos = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    return pos[0], pos[-1] + 1


def main() -> None:
    cfg = load_config()
    out_path = ROOT / cfg["out_dir"] / "circuits/b1_en/causal_trace.json"
    if out_path.exists() and not cfg.get("force"):
        print("cached", out_path)
        return
    mf = Manifest("causal_trace", cfg, 0)
    model = load_tl(cfg)
    tok = model.tokenizer
    n_layers = model.cfg.n_layers
    ev_path = ROOT / "data/b1/model_eval_en.jsonl"
    mf.add_input(ev_path)
    ev = [json.loads(line) for line in ev_path.open()]
    rels = set(cfg["behaviour"]["b1_target_relations"])
    facts = [r for r in ev if r["known"] and r["relation"] in rels and r["p_gold"] >= 0.3]
    facts = sorted(facts, key=lambda r: (r["relation"], r["subject_qid"]))
    facts = facts[:: max(1, len(facts) // N_FACTS)][:N_FACTS]
    rows = []
    for r in facts:
        text = chat_wrap(tok, TEMPLATES[r["relation"]]["en"].format(s=r["subject"]), "minimal")
        assert text == r["text"], "prompt text differs from the filter pass"
        rows.append((r, subject_span(tok, text, r["subject"])))

    # noise scale: std of the embeddings of every subject token in the sample (ROME's 3σ rule)
    embs = []
    with torch.no_grad():
        for r, (a, b) in rows:
            toks = model.to_tokens(r["text"], prepend_bos=False)
            embs.append(model.embed(toks)[0, a:b].float().cpu())
    sigma = float(torch.cat(embs).std())
    g = torch.Generator().manual_seed(0)

    aie_win = torch.zeros(n_layers)
    aie_one = torch.zeros(n_layers)
    p_clean_all, p_corrupt_all = [], []
    with torch.no_grad():
        for r, (a, b) in rows:
            toks = model.to_tokens(r["text"], prepend_bos=False)
            last = b - 1
            gold = r["label"]
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_mlp_out"))
            clean_mlp = {
                lay: cache[f"blocks.{lay}.hook_mlp_out"][0, last].clone() for lay in range(n_layers)
            }
            p_clean = float(model(toks)[0, -1].float().softmax(-1)[gold])
            batch = toks.repeat(2 * n_layers + 1, 1)  # rows: window restore ×36, single ×36, none
            for _ in range(N_DRAWS):
                noise = torch.randn((b - a, model.cfg.d_model), generator=g) * NOISE_SCALE * sigma

                def corrupt(x, hook, noise=noise, a=a, b=b):
                    x[:, a:b] += noise.to(x.device, x.dtype)
                    return x

                hooks = [("hook_embed", corrupt)]
                for lay in range(n_layers):
                    win_rows = [
                        i for i in range(n_layers) if i - WINDOW // 2 <= lay < i + WINDOW // 2
                    ]
                    rows_ = [*win_rows, n_layers + lay]

                    def restore(x, hook, rows_=rows_, v=clean_mlp[lay], last=last):
                        x[rows_, last] = v.to(x.dtype)
                        return x

                    hooks.append((f"blocks.{lay}.hook_mlp_out", restore))
                logits = model.run_with_hooks(batch, fwd_hooks=hooks)[:, -1].float()
                p = logits.softmax(-1)[:, gold].cpu()
                p_corrupt = float(p[-1])
                aie_win += p[:n_layers] - p_corrupt
                aie_one += p[n_layers : 2 * n_layers] - p_corrupt
                p_corrupt_all.append(p_corrupt)
            p_clean_all.append(p_clean)
    n = len(rows) * N_DRAWS
    aie_win, aie_one = (aie_win / n).tolist(), (aie_one / n).tolist()
    ok = editable_layers(n_layers)
    layer = max(ok, key=lambda lay: aie_win[lay])
    res = {
        "method": "causal tracing (Meng et al. 2022): subject embeddings + 3σ noise, MLP restore",
        "n_facts": len(rows),
        "n_draws": N_DRAWS,
        "window": WINDOW,
        "sigma_embed": sigma,
        "p_clean_mean": sum(p_clean_all) / len(p_clean_all),
        "p_corrupt_mean": sum(p_corrupt_all) / len(p_corrupt_all),
        "aie_window": aie_win,
        "aie_single": aie_one,
        "layer": layer,
        "layer_single_argmax": max(ok, key=lambda lay: aie_one[lay]),
        "editable_layers": ok,
    }
    out_path.write_text(json.dumps(res, indent=1))
    mf.add_output(out_path)
    mf.write(out_path.with_name("manifest_causal_trace.json"))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("aie")}, indent=1))
    print("window AIE:", " ".join(f"{i}:{v:.3f}" for i, v in enumerate(aie_win)))


if __name__ == "__main__":
    main()
