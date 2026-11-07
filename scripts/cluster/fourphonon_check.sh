#!/bin/bash
# Reproduce one of FourPhonon's shipped examples and compare with its reference output, using
# mga's parser (checks the build and parse_shengbte_kappa on real output in one go).
#
#   sbatch -p CPU-amd -N1 -n 16 -c 8 -t 12:00:00 -J mga-4ph-si  scripts/cluster/fourphonon_check.sh Example-Si
#   sbatch -p CPU-amd -N1 -n 16 -c 8 -t 12:00:00 -J mga-4ph-bas scripts/cluster/fourphonon_check.sh Example-BAs
set -euo pipefail
EX=$1
SRC=/share/yujie.cen/mga/src/MLIP_generative_agent
REF=/share/yujie.cen/mga/opt/FourPhonon/$EX/output
RUN=/share/yujie.cen/mga/fourphonon_tests/$EX
rm -rf "$RUN" && mkdir -p "$RUN" && cp /share/yujie.cen/mga/opt/FourPhonon/$EX/input/* "$RUN"/
cd "$RUN"
start=$(date +%s)
bash "$SRC/scripts/cluster/fourphonon_run.sh" > log.out 2>&1
echo "wall_s $(( $(date +%s) - start ))  ranks ${SLURM_NTASKS:-1} x threads ${SLURM_CPUS_PER_TASK:-1}"

# conda activate scripts reference unset variables (NVCC_PREPEND_FLAGS, job 963770)
set +u
source ~/modules/ll/etc/profile.d/conda.sh && conda activate /share/yujie.cen/mga/envs/dpa
PYTHONPATH=$SRC/src python - "$RUN" "$REF" <<'EOF'
import sys
from pathlib import Path
from mga.workflows.ladder import parse_shengbte_kappa
run, ref = (parse_shengbte_kappa(Path(p)) for p in sys.argv[1:3])
print("run", run)
print("ref", ref)
for k in ("kappa_avg", "kappa_rta_avg"):
    if run and ref and run.get(k) and ref.get(k):
        print(f"{k}: run {run[k]:.3f}  ref {ref[k]:.3f}  rel.dev {run[k] / ref[k] - 1:+.2e}")
EOF
