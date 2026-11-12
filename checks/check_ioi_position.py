"""Where CircuitKIT's built-in IOI task reads its metric on this model.

`get_ioi_data_only` tokenises the full IOI sentence *including* the indirect object ("... gave a
bone to Mary") and marks `end_idxs` = the "to" position; the EAP path then writes
`model.to_string(tokens)` (answer and literal <PAD> text included) to a CSV and its metric reads
`input_length − 1`, the last non-pad token. If that token is the IO name itself, the attributed
quantity is logit(IO) − logit(S) for the token *after* the answer, not the IOI prediction. This
check tokenises the cached pairs exactly as the EAP path does (`tokenize_batch_pair`,
templated=False) and reports, per position, the mean clean and corrupt logit difference, whether
the token at the metric position is the IO id, and the BOS count. Runs on CPU (fp32 forward of
short prompts) so it never touches the MPS budget of a running chain.
Writes checks/results/ioi_position.json (checks/ioi_position_<model>.json for another model, e.g.
model=gpt2). Run: uv run python checks/check_ioi_position.py device=cpu dtype=float32
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import circuitkit as ck  # noqa: F401  before transformer_lens
import torch
from aya_circuits.config import load_config
from aya_circuits.faithfulness import discovery_cfg
from aya_circuits.model import load_tl
from aya_circuits.registry import resolve_task
from circuitkit.backends.eap.eap_utils import tokenize_batch_pair

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def main() -> None:
    cfg = load_config()
    n = int(cfg.get("n", 16))
    model = load_tl(cfg)
    spec = resolve_task("ioi", cfg)
    dl = spec.build_dataloader(model, discovery_cfg(cfg, spec, 0, "zero", n), cfg["device"])
    bos = model.tokenizer.bos_token_id
    rows = []
    for clean, corrupt, labels in dl:
        tc, tk, _, _, lens, _ = tokenize_batch_pair(
            model, clean, corrupt, pair_padding_side="right", templated=False
        )
        lc, lk = model(tc).float(), model(tk).float()
        for i in range(len(clean)):
            e = int(lens[i]) - 1  # CircuitKIT's metric position
            io, s = int(labels[i, 0]), int(labels[i, 1])
            row = {
                "clean": clean[i],
                "io": model.tokenizer.decode([io]),
                "s": model.tokenizer.decode([s]),
                "n_bos_prefix": int((tc[i, :2] == bos).sum()),
                "token_at_metric_pos": model.tokenizer.decode([int(tc[i, e])]),
                "metric_pos_is_io": bool(int(tc[i, e]) == io),
            }
            for name, pos in (("metric_pos", e), ("to_pos", e - 1)):
                row[f"clean_logit_diff_{name}"] = float(lc[i, pos, io] - lc[i, pos, s])
                row[f"corrupt_logit_diff_{name}"] = float(lk[i, pos, io] - lk[i, pos, s])
                row[f"clean_top1_{name}"] = model.tokenizer.decode([int(lc[i, pos].argmax())])
            rows.append(row)
    summary = {
        k: sum(r[k] for r in rows) / len(rows)
        for k in (
            "clean_logit_diff_metric_pos",
            "corrupt_logit_diff_metric_pos",
            "clean_logit_diff_to_pos",
            "corrupt_logit_diff_to_pos",
        )
    }
    summary["frac_metric_pos_is_io"] = sum(r["metric_pos_is_io"] for r in rows) / len(rows)
    summary["frac_double_bos"] = sum(r["n_bos_prefix"] == 2 for r in rows) / len(rows)
    summary["frac_clean_top1_is_io_at_to_pos"] = sum(
        r["clean_top1_to_pos"] == r["io"] for r in rows
    ) / len(rows)
    out = {"n": len(rows), "summary": summary, "rows": rows}
    tag = "" if "aya" in cfg["model"] else "_" + cfg["model"].split("/")[-1]
    (ROOT / f"checks/results/ioi_position{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    print(json.dumps(rows[0], indent=1))


if __name__ == "__main__":
    main()
