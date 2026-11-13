#!/usr/bin/env bash
# Amendment A22 (docs/preregistration.md): close the cross-lingual refusal gap with a
# circuit-restricted LoRA and a retain objective. Splits, translations and retain targets, the 24
# cells with their dev-selected checkpoints, refusal on the held-out test sets, the Nothing-Else-
# Broke audit (L1/L2/L3/L5, hf side), the LLM judge and the analysis. Writes runs/a22/ and
# data/a22/; reads the main study's refusal circuits (runs/circuits/b2_en). The judge stage needs
# llama.cpp's `llama-server` and JUDGE_GGUF=<path to a GGUF chat model>; without it the stage is
# skipped and the analysis reports the regex judge only. Extra key=value arguments go to every
# model stage.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

[[ -e data/a22/benign.jsonl ]] || run "data" uv run python data/a22_build.py "$@"
# One cell per training process and four per evaluation process: the MPS allocator grows across
# cells, and on a nearly full disk the swap it causes can fill the volume.
loop "cells" uv run python interventions/run_a22.py \
  out_dir=runs/a22 grid.cells_per_process=1 "$@"
loop "test refusal" uv run python audit/a22_eval.py \
  out_dir=runs/a22 audit.cells_per_process=4 "$@"
loop "audit hf" uv run python audit/run_audit.py \
  grid.behaviour=b2 cells=lora,steer side=hf out_dir=runs/a22 audit.cells_per_process=4 "$@"
if [[ -n ${JUDGE_GGUF:-} ]]; then
  run "llm judge" uv run python audit/a22_judge.py out_dir=runs/a22 judge_gguf="$JUDGE_GGUF"
fi
run analyze uv run python audit/a22_analyze.py out_dir=runs/a22 private=.cache/private/a22
echo "=== done $(date)"
