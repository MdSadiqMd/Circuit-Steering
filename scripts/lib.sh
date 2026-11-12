# shellcheck shell=bash
# Helpers shared by the pipeline chains in scripts/chains/. Each chain does
# `cd "$(dirname "$0")/../.."` and then sources this file, so every command runs from the
# repository root.
#
# DRY_RUN=1 prints each stage's command instead of running it, which checks a chain's plan without
# a model or GPU. Every stage is resumable on its own (it skips outputs that exist), so a failed
# chain is fixed and rerun from the top.

export CIRCUITKIT_TL_MEMORY_PATCH=1

# Drops the progress noise of transformers / CircuitKIT from a stage's output.
quiet() {
  grep --line-buffered -vF -e 'Redirects' -e 'Loading weights' -e 'it/s]' -e '•'
}

# Under DRY_RUN, prints the command and succeeds; otherwise fails, so the caller runs it.
dry_print() {
  [[ -n ${DRY_RUN:-} ]] || return 1
  printf '   '
  printf ' %q' "$@"
  printf '\n'
}

# run <label> <command...>: runs one stage; a non-zero exit stops the chain.
run() {
  local label=$1
  shift
  echo "=== $label $(date)"
  dry_print "$@" && return 0
  "$@" 2>&1 | quiet
  local rc=${PIPESTATUS[0]}
  if ((rc != 0)); then
    echo "=== FAILED $label (exit $rc)"
    exit 1
  fi
}

# try <label> <command...>: like run, but a failure is logged and the chain moves on (for
# stand-alone checks that later stages do not read).
try() {
  local label=$1
  shift
  echo "=== $label $(date)"
  dry_print "$@" && return 0
  "$@" 2>&1 | quiet
  local rc=${PIPESTATUS[0]}
  if ((rc != 0)); then
    echo "=== FAILED $label (exit $rc), continuing"
  fi
}

# loop <label> <command...>: re-invokes a stage while it exits 3, the code run_grid.py and
# run_audit.py use for "stopped after grid/audit.cells_per_process cells, more work left" (they
# bound the MPS allocator's growth per process); any other non-zero exit stops the chain.
loop() {
  local label=$1
  shift
  while true; do
    echo "=== $label $(date)"
    dry_print "$@" && return 0
    "$@" 2>&1 | quiet
    local rc=${PIPESTATUS[0]}
    if ((rc == 0)); then
      return 0
    elif ((rc != 3)); then
      echo "=== FAILED $label (exit $rc)"
      exit 1
    fi
  done
}

# Both behaviours' hypothesis tests. The pooled H3 block of each reads the other behaviour's
# h3_points file, so B1 runs again after B2 to pool with B2's final points.
analyze_both() {
  local beh
  for beh in b1 b2 b1; do
    run "analyze $beh" uv run python audit/analyze.py grid.behaviour="$beh"
  done
}
