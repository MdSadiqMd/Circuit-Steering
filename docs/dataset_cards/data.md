---
pretty_name: "Circuit steering on Tiny Aya: prompts and facts"
language: [en, hi, ar, fr, sw]
tags:
- mechanistic-interpretability
- ai-safety
- refusal
- knowledge-editing
configs:
- config_name: refusal_prompts
  default: true
  data_files:
  - split: cyber
    path: b2/pool_cyber.jsonl
  - split: theft_fraud
    path: b2/pool_theft_fraud.jsonl
  - split: weapons_violence
    path: b2/pool_weapons_violence.jsonl
- config_name: benign_prompts
  data_files: a22/benign.jsonl
- config_name: xstest
  data_files: b2/xstest_all.jsonl
- config_name: fact_prompts
  data_files:
  - split: en
    path: b1/prompts_en.jsonl
  - split: hi
    path: b1/prompts_hi.jsonl
  - split: ar
    path: b1/prompts_ar.jsonl
  - split: fr
    path: b1/prompts_fr.jsonl
  - split: sw
    path: b1/prompts_sw.jsonl
- config_name: false_facts
  data_files: b1/false_facts.jsonl
---

# Circuit steering on Tiny Aya: prompts and facts

The prompts, facts and splits used to find circuits in `CohereLabs/tiny-aya-global`, change the
model, and test it, in English, Hindi, Arabic, French and Swahili. Part of the Lexsi hackathon
submission [Circuit-Steering](https://github.com/MdSadiqMd/Circuit-Steering) (Track 2b).

This repo contains harmful requests taken from public benchmarks (the requests only). No reply to a
harmful request is here.

## The tables in the viewer

| config | splits | content |
|---|---|---|
| `refusal_prompts` | `cyber`, `theft_fraud`, `weapons_violence` | harmful requests (HarmBench, MultiJail) and harmless look-alikes (XSTest) per harm category, in five languages |
| `benign_prompts` | `train` | harmless prompts used to train and tune the adapters, with the base model's replies |
| `xstest` | `train` | the 450 XSTest prompts (250 safe, 200 unsafe), English |
| `fact_prompts` | `en`, `hi`, `ar`, `fr`, `sw` | 1,502 Wikidata fact questions per language |
| `false_facts` | `train` | 180 facts the model gets wrong, with its wrong answer |

In `refusal_prompts`, the column `split` says `harmful` or `benign`, and `source` names the
benchmark. `translated_by_model` lists the languages Tiny Aya translated itself; `bt_<lang>` is the
translation put back into English and `bt_f1_<lang>` its word overlap with the original.

```python
from datasets import load_dataset

cyber = load_dataset("MdSadiqMd/circuit-steering-data", "refusal_prompts", split="cyber")
```

## The other files

| path | content |
|---|---|
| `a22/split.json` | which prompt ids are train, dev and test for the adapters |
| `a23/`, `a24/` | plain text used to measure and to protect fluency |
| `b1/facts_pool.jsonl`, `b1/model_eval_<lang>.jsonl`, `b1/ripple_en.jsonl` | the fact pool, the base model's answer to every fact, related facts |
| `b1/eap_<lang>.csv`, `b2/eap_<lang>.csv`, `b1/eap_known_<lang>.csv`, `b2/eap_<lang>_matched.csv` | clean and corrupted prompt pairs used to find the circuits (CSV; the viewer shows one file format per repo, so these are not in it) |
| `b2/baseline_*.jsonl` | whether the base model refused each prompt (no reply text) |
| `bystanders/` | prompt pairs for the other circuits checked for side effects |
| `*.py`, `MANIFEST.json` | the scripts that built these files, and file hashes |

Sources: HarmBench, MultiJail, XSTest and Wikidata; each keeps its own licence. Translations and
replies come from `CohereLabs/tiny-aya-global` (CC-BY-NC 4.0).
