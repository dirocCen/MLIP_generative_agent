#!/bin/bash
# Run ShengBTE_cpu (FourPhonon) in the current directory (CONTROL + FORCE_CONSTANTS_* there).
#   fourphonon_run.sh [n_mpi]      n_mpi defaults to SLURM_NTASKS or 1; OpenMP threads = SLURM_CPUS_PER_TASK
# Used as $MGA_FOURPHONON by the L3 job and by the example check (fourphonon_check.sh).
set -euo pipefail
E=/share/yujie.cen/mga/envs/fourphonon
BIN=/share/yujie.cen/mga/opt/FourPhonon/ShengBTE_cpu
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OMP_STACKSIZE=1G
ulimit -s unlimited || true
NP=${1:-${SLURM_NTASKS:-1}}
# single node: shared memory only; do not let OpenMPI bind every rank to one core
exec mpirun --bind-to none --mca btl self,vader -np "$NP" "$BIN"
