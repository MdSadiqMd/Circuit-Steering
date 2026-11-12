#!/usr/bin/env bash
# Run on the Mac: copy every A19–A21 output and log from puget into this checkout. Additive only:
# A19 adds runs/audit/a19/ and a19.{json,safetensors} next to the main audit files (never
# overwriting anything else there); A20/A21 have their own roots; puget's logs go to
# .cache/logs/puget/. Safe to repeat (rsync only transfers what changed).
cd "$(dirname "$0")/../.." || exit 1
R=puget:/dev/shm/cs/circuit-steering
set -e
rsync -a $R/runs/audit/a19/ runs/audit/a19/
rsync -a --include='*/' --include='a19.json' --include='a19.safetensors' --exclude='*' --prune-empty-dirs $R/runs/audit/ runs/audit/
for d in runs_a20 runs_a21 runs_a20_repro runs_a21_repro; do
  if ssh -n puget "test -d /dev/shm/cs/circuit-steering/$d"; then rsync -a $R/$d/ $d/; fi
done
rsync -a --include='quadrature_tinyaya.json' --include='a19_repro.json' --include='a20_a21_repro_*.json' --exclude='*' $R/checks/ checks/
mkdir -p .cache/logs/puget && rsync -a $R/logs_puget/ .cache/logs/puget/
echo "pulled $(date)"
