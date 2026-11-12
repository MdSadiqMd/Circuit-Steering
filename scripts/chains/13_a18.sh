#!/usr/bin/env bash
# Amendment A18 (2026-11-08): LoRA cells are applied as exact low-rank hooks (fp32 factors) instead
# of being merged into the bf16 weights (fifth audit B.1: 31-43 % of ΔW rounded to zero). Every
# LoRA cell of both behaviours is re-audited on all five layers (HF layers, L4 target + bystanders +
# ioi_fixed in one pass, pillar and shared masks, ripple, arbitration, relearning, the L4 power
# check), then the checks, tables and analyses are regenerated. Superseded outputs are moved, not
# deleted, to runs/_superseded/2026-11-08_A18_lora_bf16_merge/ (idempotent). Runs after
# 12_a16.sh.
set -u
cd "$(dirname "$0")/../.." || exit 1
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
export HF_HUB_OFFLINE=1

run supersede uv run python - <<'PY'
import json, shutil
from pathlib import Path
sup = Path("runs/_superseded/2026-11-08_A18_lora_bf16_merge")
for beh in ("b1", "b2"):
    for d in sorted(Path(f"runs/audit/{beh}").glob(f"{beh}_lora_*")):
        dst = sup / beh / d.name
        if dst.exists():
            continue  # already moved; a partial re-run keeps its new files in place
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), dst)
        print("moved", d)
    for sc in sorted(Path(f"runs/interventions/{beh}/lora").glob("*/scorecard.json")):
        dst = sup / beh / "scorecards" / f"{sc.parent.name}.json"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sc, dst)
p = Path("runs/audit/b1/power/l4_power.json")
if p.exists():
    d = json.loads(p.read_text())
    lora = [k for k in d if "_lora_" in k]
    if lora:
        (sup / "b1").mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, sup / "b1" / "l4_power.json")
        for k in lora:
            d.pop(k)
        p.write_text(json.dumps(d, indent=1))
        print("dropped power entries", lora)
PY
run "apply equivalence" uv run python checks/check_apply_equivalence.py
for b in b1 b2; do
  loop "audit $b lora" uv run python audit/run_audit.py grid.behaviour="$b" cells=lora
done
run ripple uv run python audit/ripple.py
run arbitration uv run python audit/arbitration.py
run relearn uv run python audit/relearn.py
for b in b1 b2; do
  run "l4 pillar mask $b" uv run python audit/l4_pillarmask.py grid.behaviour="$b"
  run "l4 shared $b" uv run python audit/l4_shared.py grid.behaviour="$b"
done
run "l4 power (lora cells)" uv run python audit/l4_power.py cells=lora/circuit_s0,lora/unrestricted_s0
run "eap vs patching ioi_fixed" uv run python checks/check_eap_vs_patching.py task=ioi_fixed seed=0
run "lora masks" uv run python checks/check_lora_masks.py
run tables uv run python -m aya_circuits.table
analyze_both
echo "=== done $(date)"
