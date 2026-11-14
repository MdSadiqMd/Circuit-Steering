---
base_model: CohereLabs/tiny-aya-global
language: [en, hi, ar, fr, sw]
tags: [mechanistic-interpretability, circuitkit, steering, lora, knowledge-editing, safety]
license: cc-by-nc-4.0
---

# Circuit steering on Tiny Aya: every intervention run

The raw record of the [Circuit-Steering](https://github.com/MdSadiqMd/Circuit-Steering) study
(Lexsi hackathon, Track 2b): 427 changes made to
[`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global), each stored as a
small file of weight changes with a description of what was changed. It is an archive for checking
and reusing our results, not a model to load.

Looking for a model to use? Take the
[recommended adapter](https://huggingface.co/MdSadiqMd/tiny-aya-global-refusal-circuit-lora-fluent).

![Runs in this repo by behaviour and method, split into weights published and weights withheld](https://huggingface.co/MdSadiqMd/circuit-steering-cells/resolve/main/figures/cells_contents.png)

## What is inside

| Folder | What the runs do | Runs | With weights |
|---|---|---|---|
| `b1/` | Correct false facts the model believes | 231 | 77 |
| `b2/` | Make the model refuse harmful requests more often | 162 | 81 |
| `a22/` | Training runs of the [first adapter](https://huggingface.co/MdSadiqMd/tiny-aya-global-refusal-circuit-lora), with its controls and the SafeTune steering baseline | 25 | 25 |
| `a24/` | Training runs of the [second adapter](https://huggingface.co/MdSadiqMd/tiny-aya-global-refusal-circuit-lora-fluent), with its controls | 9 | 9 |

`b1/` and `b2/` are the 393 runs of the pre-registered study. 213 of them got the full audit; the
other 180 are steering runs at other strengths, with a shorter audit.

Each run is one folder, `<behaviour>/<method>/<condition>_s<seed>/`:

| File | Content |
|---|---|
| `cell.json` | What was changed: behaviour, method, condition, seed, the list of components, and training or editing details |
| `delta.safetensors` | The change itself, as small factors (never a full checkpoint). Missing when withheld |
| `scorecard.json` | The audit numbers of the run (`b1/` and `b2/`) |
| `manifest.json` | Config, code version and file hashes, where the run wrote one |

Methods: `steer` adds a vector to the outputs of the chosen components at inference, `lora` trains a rank-8 adapter
on them, `edit` rewrites MLP weights with MEMIT.

## The conditions

A condition says which components a run was allowed to change. Comparing them is the point of the
study: if the circuit matters, `circuit` should beat the others.

| Condition | Components changed |
|---|---|
| `circuit` | The top 10 % by circuit score (CircuitKIT, EAP-IG), heads and MLPs ranked separately |
| `random` | The same number of random components |
| `inverse` | The lowest-scoring components |
| `taylor` | The components with the largest gradient |
| `unrestricted` | All components |
| `perturb_<condition>` | Random noise of the same size on the same components, as a control |
| `circuit_deep`, `causal_trace` | Editing only: the circuit's top MLP outside layers 0–3, and the layer found by causal tracing |
| `circuit_joint`, `random_joint`, `inverse_joint` | A second circuit definition, with heads and MLPs ranked together |
| `<condition>@<strength>` | Steering at 0.25, 0.5, 1 or 2 times the base strength (for example `circuit@0.25`) |
| `safetune_rd` | SafeTune's refusal-direction steering, the baseline |
| `circuit30`, `random30`, `inverse30`, `circuit10`, `random10` | Adapter runs (`a22/`, `a24/`): the top, random or lowest 30 % or 10 % |

## What the runs showed

Changing only the circuit did not keep a change more local.

![Fact correction: accuracy gain on the edited facts and side effects in other languages, by method and by which components were changed](https://huggingface.co/MdSadiqMd/circuit-steering-cells/resolve/main/figures/cells_locality.png)

Bars are means over seeds, dots are single runs. The picture shows fact correction (`b1/`).

- **The circuit did not reduce side effects.** In no method did the circuit runs pass our
  pre-registered test of having fewer side effects than the random and lowest-scoring runs.
- **It often cost effect.** LoRA on the circuit raised accuracy on the edited facts by 41 points,
  on random components by 64, on all components by 99.
- **Editing the circuit was the worst case.** The circuit's top MLP is in layer 0, and MEMIT
  there moved other languages by about 20 points, against 7 for random components.
- **Steering barely corrected facts** in any condition.

The refusal runs (`b2/`) give the same answer. All tables:
[circuit-steering-audit](https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit)
(`hypotheses_b1.md`, `hypotheses_b2.md`).

## Why some weights are missing

We do not publish a change that could make the model refuse less. A weight file is withheld when
the audit shows a lower refusal rate than the base model on any harmful prompt set in any
language. 235 of the 393 files are withheld, listed in `WITHHELD.md`. `cell.json` and the audit
numbers are published for every run.

## Load a run

```python
import json

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

repo, run = "MdSadiqMd/circuit-steering-cells", "b2/lora/circuit_s0"
cell = json.load(open(hf_hub_download(repo, f"{run}/cell.json")))
delta = load_file(hf_hub_download(repo, f"{run}/delta.safetensors"))
print(cell["condition"], len(cell["nodes"]), "components,", len(delta), "tensors")
```

To apply a run to the model, use `Cell.load` and `apply_hf` in `interventions/cell.py` of the
[code repo](https://github.com/MdSadiqMd/Circuit-Steering). They add the change (forward hooks, or
weights in place for edits) and return a function that undoes it.

## Links and licence

- Results as charts: https://huggingface.co/spaces/MdSadiqMd/circuit-steering
- Every result file: https://huggingface.co/datasets/MdSadiqMd/circuit-steering-audit
- The circuits: https://huggingface.co/datasets/MdSadiqMd/circuit-steering-circuits

These files are derived from `CohereLabs/tiny-aya-global` and carry that model's licence
(CC-BY-NC 4.0) and Cohere's acceptable-use terms. Code: BSD-3-Clause.
