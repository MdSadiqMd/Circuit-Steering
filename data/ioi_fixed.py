"""IOI bystander task, specified correctly for this tokenizer (amendment A16).

CircuitKIT's built-in `ioi` EAP path serialises the whole sentence *including* the indirect object
("... gave a bone to Mary") plus literal <PAD> text, prepends a second BOS, and reads its metric at
the last non-pad token, i.e. at the answer itself (`checks/check_ioi_position.py`: metric position
is the IO token on 100 % of pairs, clean logit difference 2.0 there vs 11.2 at the "to" position;
the same holds on GPT-2). 2 of 4 pairs per batch also differ in token count on this tokenizer.

This builder keeps CircuitKIT's own generator (same ABBA template, same names, same ABC
corruption) and only fixes the specification: the prompt stops before the indirect object, the
text carries one BOS and is registered `templated=True` like every other task here, both names of
a pair are single tokens, and clean / corrupt have the same token count. Labels: correct_idx =
the IO token, incorrect_idx = the S token, so the metric is the IOI logit difference at the "to"
position. Output: data/bystanders/eap_ioi_fixed.csv (task `ioi_fixed`).
Run: uv run python data/ioi_fixed.py [n_rows=740 n_candidates=2000]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aya_circuits.config import load_config
from circuitkit.data.task_data.tasks.ioi.ioi_dataset import IOIDataset
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent / "bystanders" / "eap_ioi_fixed.csv"


def main() -> None:
    cfg = load_config()
    n_rows, n_cand = int(cfg.get("n_rows", 740)), int(cfg.get("n_candidates", 2000))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    mock = SimpleNamespace(  # IOIDataset / TokenIDGenerator need only the tokenizer
        tokenizer=tok,
        cfg=SimpleNamespace(model_name=cfg["model"]),
        to_tokens=lambda text, prepend_bos=False: torch.tensor(
            [tok.encode(text, add_special_tokens=prepend_bos)]
        ),
    )
    clean_ds = IOIDataset("ABBA", N=n_cand, model=mock, nb_templates=1, seed=0)
    abc_ds = (  # CircuitKIT's ABC corruption (get_ioi_data_only), seeds as there
        clean_ds.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )

    def ids(word: str) -> list[int]:
        return tok.encode(" " + word, add_special_tokens=False)

    def strip_io(p: dict) -> str:
        tail = " " + p["IO"]
        assert p["text"].endswith(tail), p
        return tok.bos_token + p["text"][: -len(tail)]

    rows, seen, dropped = [], set(), {"multi_token": 0, "length": 0, "dup": 0}
    for p, q in zip(clean_ds.ioi_prompts, abc_ds.ioi_prompts, strict=True):
        # the ABC prompt's third name is not in its dict; the token-count check below catches it
        if any(len(ids(n)) != 1 for n in (p["IO"], p["S"], q["IO"], q["S"])):
            dropped["multi_token"] += 1
            continue
        clean, corrupt = strip_io(p), strip_io(q)
        if len(tok.encode(clean, add_special_tokens=False)) != len(
            tok.encode(corrupt, add_special_tokens=False)
        ):
            dropped["length"] += 1
            continue
        if clean in seen:
            dropped["dup"] += 1
            continue
        seen.add(clean)
        rows.append(
            {
                "clean": clean,
                "corrupted": corrupt,
                "correct_idx": ids(p["IO"])[0],
                "incorrect_idx": ids(p["S"])[0],
            }
        )
        if len(rows) == n_rows:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} rows={len(rows)} dropped={dropped}")
    print(rows[0])


if __name__ == "__main__":
    main()
