#!/usr/bin/env bash
# The whole pipeline once, on one seed: fetch the data, find the refusal circuit, train the second
# adapter on the circuit and on random components of the same size, test both on the held-out
# prompts, audit them and write the comparison. It is the study's own stages at their real
# settings, so its circuit30 row should match seed 0 of docs/results/a24_b2.md (regex columns).
# About 75 minutes on a 48 GB Apple-silicon Mac. Writes runs/circuits/b2_en/seed0/, runs/demo/
# and docs/results/demo_b2.md; replies to harmful prompts stay in .cache/private/demo/. The model
# is gated: accept its licence on the Hub and log in (`hf auth login`) or set HF_TOKEN. There is
# no LLM judge here, so harmful refusals are counted by phrases. Extra key=value arguments go to
# every model stage, e.g. `just chain demo device=cuda`.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh

run "fetch data" uv run python data/fetch.py
run "discover the refusal circuit" uv run python -m aya_circuits.discover task=b2_en seed=0 "$@"
loop "train adapters" uv run python interventions/run_a22.py \
  out_dir=runs/demo amendment=A24 plain=data/a24/plain.jsonl draws=1 "seeds=[0]" \
  "conditions=[circuit30,random30]" grid.cells_per_process=1 "$@"
loop "test refusal" uv run python audit/a22_eval.py \
  out_dir=runs/demo private=.cache/private/demo audit.cells_per_process=4 "$@"
loop "audit" uv run python audit/run_audit.py \
  grid.behaviour=b2 cells=lora side=hf out_dir=runs/demo audit.cells_per_process=4 "$@"
run "fluency" uv run python audit/a23_fluency.py out_dir=runs/demo "$@"
run "analyze" uv run python audit/a22_analyze.py out_dir=runs/demo private=.cache/private/demo
echo "=== done $(date): read docs/results/demo_b2.md"
