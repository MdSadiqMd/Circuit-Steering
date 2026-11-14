---
pretty_name: "Circuit steering on Tiny Aya: audit results"
language: [en, hi, ar, fr, sw]
tags:
- mechanistic-interpretability
- ai-safety
- evaluation
configs:
- config_name: a22_b2
  data_files: a22_b2.csv
  default: true
- config_name: a24_b2
  data_files: a24_b2.csv
- config_name: b1_cells
  data_files: b1_cells.csv
- config_name: b2_cells
  data_files: b2_cells.csv
---

# Circuit steering on Tiny Aya: audit results

Every result of the "Nothing-Else-Broke" audit on `CohereLabs/tiny-aya-global`: one row per
intervention run, plus the per-item files behind each number. Part of the Lexsi hackathon submission
[Circuit-Steering](https://github.com/MdSadiqMd/Circuit-Steering) (Track 2b).

Replies to harmful prompts are not in this repo. For harmful prompts it holds only the judgment
(refused or complied); replies to harmless prompts are included.

## The tables in the viewer

| config | file | rows | content |
|---|---|---|---|
| `a22_b2` | `a22_b2.csv` | 25 | first adapter (A22): 24 training runs and the SafeTune steering baseline |
| `a24_b2` | `a24_b2.csv` | 9 | second adapter (A24): every training run, with its controls |
| `b1_cells` | `b1_cells.csv` | 231 | earlier runs on fact correction (steering, LoRA, MEMIT) |
| `b2_cells` | `b2_cells.csv` | 162 | earlier runs on refusal |

In the adapter tables, `condition` says which components were trained: `circuit30` is the top 30 %
by circuit score, `random30` a random set of the same size, `inverse30` the lowest-scoring 30 %,
`unrestricted` all components; `circuit10` and `random10` are the same at 10 %, and `safetune_rd`
is the SafeTune steering baseline. `comply_xl` is the share of harmful test requests the model helped
with in Hindi, Arabic, French and Swahili; `comply_en` is the same in English. The released
adapters are the rows `b2_lora_circuit30_s0`.

```python
from datasets import load_dataset

first = load_dataset("MdSadiqMd/circuit-steering-audit", "a22_b2", split="train")
```

## The other files

| path | content |
|---|---|
| `a22_b2.md`, `a24_b2.md`, `*.json` next to them | the adapter results as text, with confidence intervals and tests |
| `hypotheses_b1.md`, `hypotheses_b2.md` | the pre-registered tests on the 393 earlier runs |
| `standard_checks.md` | how many runs pass the usual side-effect check and are still damaged |
| `circuits.md` | faithfulness of every circuit |
| `a19_*`, `a20_*`, `a21_*`, `puget-a19-a21.md` | three follow-up studies, each with its own `.md` |
| `a22/`, `a24/` | per run: test judgments (`eval/`), judge labels (`judge/`, `judge_benign/`), fluency (`fluency/`), audit (`audit/`), logs (`logs/`) |
| `b1/<run>/`, `b2/<run>/` | the audit files of each earlier run; `base/` is the unchanged model |
| `figures/adapters.png` | a summary figure of both adapters |
| `article/` | the pictures, short clips and the 7-minute film used in the write-up |

Licence: these files are derived from `CohereLabs/tiny-aya-global` (CC-BY-NC 4.0).
