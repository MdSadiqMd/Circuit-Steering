---
pretty_name: "Circuit steering on Tiny Aya: circuits"
language: [en, hi, ar, fr, sw]
tags:
- mechanistic-interpretability
- circuit-discovery
- ai-safety
configs:
- config_name: component_scores
  data_files: component_scores.csv
---

# Circuit steering on Tiny Aya: circuits

Circuits found in `CohereLabs/tiny-aya-global` with CircuitKIT (EAP-IG, node level): 10 tasks,
3 seeds each, 30 runs. Part of the Lexsi hackathon submission
[Circuit-Steering](https://github.com/MdSadiqMd/Circuit-Steering) (Track 2b).

## The table in the viewer

`component_scores.csv` has one row per task, seed and component: 30 runs times 612 components
(576 attention heads, 36 MLPs) = 18,360 rows.

| column | meaning |
|---|---|
| `task` | which behaviour the circuit is for (see below) |
| `seed` | discovery seed (0, 1, 2) |
| `component` | `A<layer>.<head>` for an attention head, `MLP <layer>` for an MLP |
| `kind` | `head` or `mlp` |
| `layer` | 0 to 35 |
| `score` | CircuitKIT's node score (EAP-IG, absolute) |
| `signed_rowsum` | sum of the component's signed outgoing edge scores |
| `rank` | rank by `score` within the run (1 = highest) |

Tasks: `b1_en`, `b1_hi`, `b1_ar`, `b1_fr`, `b1_sw` (fact recall, per language), `b2_en` (refusal),
`b2m_en` (refusal, length-matched pairs), `ioi`, `ioi_fixed`, `langid_en` (other circuits, used to
check for side effects).

```python
from datasets import load_dataset

scores = load_dataset("MdSadiqMd/circuit-steering-circuits", "component_scores", split="train")
```

## The other files

Each run has a folder `<task>/seed<n>/`:

| file | content |
|---|---|
| `nodes.json` | the scores the table is built from |
| `edges.safetensors` | signed edge scores |
| `circuit.pt`, `circuit_scores.pt`, `circuit_scores.json` | the circuit as CircuitKIT saves it |
| `faithfulness.json`, `random_baseline.json` | faithfulness checks against random components |
| `cellset.json` | the top-10 % component set the earlier runs edited, with its controls |
| `manifest*.json` | config, code version and file hashes of the run |

`overlap.json` holds the overlap between tasks and between seeds.

Licence: these files are derived from `CohereLabs/tiny-aya-global` (CC-BY-NC 4.0).
