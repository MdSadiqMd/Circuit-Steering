#!/bin/bash
# End-to-end dry run of the A20 / A21 code paths at tiny sizes into scratch roots ($CS/tmp/dry_*).
# Only exit status and errors are read; the numbers are discarded.
# shellcheck source=scripts/puget/env.sh
# shellcheck disable=SC2086  # $A, $SMALL and $R are split into key=value words on purpose
source /dev/shm/cs/env.sh
cd /dev/shm/cs/circuit-steering || exit 1
set -o pipefail
D=$CS/tmp/dry
rm -rf $D
mkdir -p $D/a20/circuits $D/a20/interventions $D/a21/interventions
cp -r runs/circuits/b1_en $D/a20/circuits/
cp runs/interventions/b2_split.json $D/a20/interventions/
cp runs/interventions/b2_split.json $D/a21/interventions/
SMALL="audit.l1.n_neighbourhood=3 audit.l1.n_paraphrase=2 audit.l1.n_mmlu=2 audit.l1.n_ifeval=1 audit.l1.n_gsm8k=1 audit.l1.n_wikitext=2 audit.l2.n_harmful=2 audit.l2.n_benign=2 audit.l5.n_never_edited=3 audit.max_new_tokens=8"
step() {
  echo "--- $1"
  shift
  "$@" >$D/last.log 2>&1 || {
    echo "DRY FAILED"
    grep -v Warning $D/last.log | tail -25
    exit 1
  }
}
step certify .venv/bin/python -m aya_circuits.certify task=b1_en seed=0 certify.n_heldout=8 device=cuda out_dir=$D/a20 cache_dir=$D/cache
cp $D/a20/circuits/b1_en/seed0/certified.json $D/a20/circuits/b1_en/seed1/
cp $D/a20/circuits/b1_en/seed0/certified.json $D/a20/circuits/b1_en/seed2/
python3 -c "import json;d=json.load(open('$D/a20/circuits/b1_en/seed0/certified.json'));print('certified present:', d['certified'] is not None, 'grid points', len(d['curve']))"
step steer .venv/bin/python interventions/run_a20.py grid.intervention=steer "only=[cert_s0,cert_strat_s3,unrestricted_s0]" device=cuda out_dir=$D/a20
step lora .venv/bin/python interventions/run_a20.py grid.intervention=lora "only=[cert_s0,cert_strat_s3]" interventions.lora.steps=2 device=cuda out_dir=$D/a20
step audit_shards bash -c "CS_GPU_CAP_GB=15 .venv/bin/python audit/run_audit.py grid.behaviour=b1 cells=steer,lora side=hf device=cuda audit.cells_per_process=0 out_dir=$D/a20 shard=0/2 $SMALL & p=\$!; CS_GPU_CAP_GB=15 .venv/bin/python audit/run_audit.py grid.behaviour=b1 cells=steer,lora side=hf device=cuda audit.cells_per_process=0 out_dir=$D/a20 shard=1/2 $SMALL; wait \$p"
ls $D/a20/audit/b1/
step a21_edit .venv/bin/python interventions/run_a21.py "a21.sites=[28]" "a21.folds=[0]" device=cuda out_dir=$D/a21
ls $D/a21/interventions/b1/edit/
step a21_audit .venv/bin/python audit/run_audit.py grid.behaviour=b1 cells=edit side=hf device=cuda audit.cells_per_process=0 out_dir=$D/a21 $SMALL
step a21_analyze .venv/bin/python audit/a21_analyze.py out_dir=$D/a21
echo "DRY OK"
