#!/usr/bin/env bash
# Amendment A24 (docs/preregistration.md): A23 with the plain-text retain targets drawn from the
# base model at temperature 1 instead of decoded greedily (A23 was stopped after one cell: greedy
# targets sharpen the adapter and raised dev perplexity). 9 cells (circuit30, random30,
# unrestricted x 3 seeds), refusal on the A22 test sets, the Nothing-Else-Broke audit, the
# five-language fluency check for A22 and A24, the LLM judge and both analyses. Writes runs/a24/;
# reads data/a24/plain.jsonl and data/a23/fluency.jsonl (data/a23_build.py) and A22's data and
# judge cache (scripts/chains/17_a22.sh). JUDGE_GGUF as in 17_a22.sh. Extra key=value arguments
# go to every model stage. A23's one cell (runs/a23) came from the same cell command with
# out_dir=runs/a23 amendment=A23 plain=data/a23/plain.jsonl.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

P=.cache/private/a24
if [[ ! -e data/a24/plain.jsonl ]]; then # the two public datasets are fetched once
  HF_HUB_OFFLINE=0 run "data" uv run python data/a23_build.py \
    variant=sampled out=data/a24/plain.jsonl "$@"
fi
loop "cells" uv run python interventions/run_a22.py \
  out_dir=runs/a24 amendment=A24 plain=data/a24/plain.jsonl draws=1 \
  "conditions=[circuit30,random30,unrestricted]" grid.cells_per_process=1 "$@"
loop "test refusal" uv run python audit/a22_eval.py \
  out_dir=runs/a24 private="$P" audit.cells_per_process=4 "$@"
loop "audit hf" uv run python audit/run_audit.py \
  grid.behaviour=b2 cells=lora side=hf out_dir=runs/a24 audit.cells_per_process=4 "$@"
run "fluency a22" uv run python audit/a23_fluency.py out_dir=runs/a22 "$@"
run "fluency a24" uv run python audit/a23_fluency.py out_dir=runs/a24 "$@"
if [[ -n ${JUDGE_GGUF:-} ]]; then
  [[ -e $P/judge_cache.json ]] || run "judge cache" cp .cache/private/a22/judge_cache.json "$P/"
  run "llm judge" uv run python audit/a22_judge.py \
    out_dir=runs/a24 private="$P" judge_gguf="$JUDGE_GGUF"
fi
run "analyze a22" uv run python audit/a22_analyze.py \
  out_dir=runs/a22 private=.cache/private/a22
run "analyze a24" uv run python audit/a22_analyze.py out_dir=runs/a24 private="$P"
echo "=== done $(date)"
