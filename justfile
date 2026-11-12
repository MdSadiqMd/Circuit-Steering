# Recipes for the circuit-steering study; `just` lists them. Every recipe runs from this directory,
# with the two sibling repos next to it (README, "Repository layout").

set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments := true

siblings := "../tiny-aya-circuits ../nothing-else-broke"

# List the recipes
default:
    @just --list --unsorted

# Install the locked environment: this repo, the two sibling packages (editable) and dev tools
sync:
    uv sync --locked

# Every fast check, no model needed: lint, tests, dry acceptance check, leak scan, shell, chain plans
check: lint test smoke-dry scan shell dry-chains

# Ruff lint and format check over the three repos
lint:
    uv run ruff check . {{ siblings }}
    uv run ruff format --check . {{ siblings }}

# Apply ruff fixes and formatting to the three repos and shfmt to the shell scripts
fmt:
    uv run ruff check --fix . {{ siblings }}
    uv run ruff format . {{ siblings }}
    uv run shfmt -i 2 -ci -w scripts

# Unit tests and entry-point tests of the three repos
test:
    uv run pytest -q --import-mode=importlib tests ../tiny-aya-circuits/tests ../nothing-else-broke/tests

# The Lexsi stack (CircuitKIT, AuditKIT, SafeTune) imports and exposes its entry points (no model)
smoke-dry:
    uv run python checks/smoke_lexsi_stack.py --dry

# Real-model acceptance check (needs HF_TOKEN), writes checks/results/stack_check.json
smoke:
    uv run python checks/smoke_lexsi_stack.py

# No harmful prompt text in any file meant for release
scan:
    uv run python data/sanitize.py scan

# shellcheck and shfmt over every script
shell:
    uv run shellcheck -x scripts/*.sh scripts/chains/*.sh scripts/puget/*.sh
    uv run shfmt -i 2 -ci -d scripts

# Every stage of every chain, printed instead of run
dry-chains:
    for f in scripts/chains/*.sh; do DRY_RUN=1 HF_TOKEN="${HF_TOKEN:-dry}" bash "$f" >/dev/null; done

# Run one chain by name, e.g. `just chain a21 device=cuda`; logs to .cache/logs/<name>.log
chain name *args:
    #!/usr/bin/env bash
    set -euo pipefail
    f=(scripts/chains/[0-9][0-9]_"$1".sh)
    [[ -e $f ]] || f=scripts/chains/$1.sh
    mkdir -p .cache/logs
    "$f" "${@:2}" 2>&1 | tee -a ".cache/logs/$1.log"

# Every numbered chain in order, as first run (each stage skips outputs that already exist)
reproduce:
    for f in scripts/chains/[0-9][0-9]_*.sh; do n=${f##*/}; n=${n#[0-9][0-9]_}; just chain "${n%.sh}"; done

# Every table and analysis regenerated from the stored runs (CPU, a few minutes)
analyze:
    uv run python -m aya_circuits.table
    for b in b1 b2 b1; do uv run python audit/analyze.py grid.behaviour="$b"; done
    for b in b1 b2; do uv run python audit/a19_analyze.py grid.behaviour="$b"; done
    uv run python audit/a20_analyze.py out_dir=runs/a20
    uv run python audit/a21_analyze.py out_dir=runs/a21
    uv run python audit/standard_checks.py
    uv run python audit/a22_analyze.py out_dir=runs/a22
    uv run python audit/a22_analyze.py out_dir=runs/a24
