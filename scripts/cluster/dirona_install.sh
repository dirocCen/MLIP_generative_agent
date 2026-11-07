#!/bin/bash
# Build the mga worker environment on dirona (VASP side: phono3py_fd displacement generation,
# FC fitting and kappa run here; DESIGN §2.3), then test it on a compute node.
#
# dirona compute nodes have no internet (checked 11-03: pypi.org does not resolve), so the
# package download/install runs here on the login node (network + unpacking only, no
# computation), and the tests go to the queue as a batch job (DESIGN §11).
#   bash dirona_install.sh          # on the dirona login node
# The env is a venv on top of ~/miniconda3's Python 3.12; the conda base and the existing
# conda envs are not touched.
set -euo pipefail
ROOT=$HOME/mga
SRC=$ROOT/src/MLIP_generative_agent
E=$ROOT/envs/mga
LOG=$ROOT/install_logs
UV=~/miniconda3/bin/uv
export UV_CACHE_DIR=$ROOT/uv_cache UV_LINK_MODE=copy
mkdir -p "$LOG"

[ -x "$E/bin/python" ] || "$UV" venv --python ~/miniconda3/bin/python3.12 "$E"
"$UV" pip install --python "$E/bin/python" -e "$SRC[worker,dft,test]"
"$UV" pip freeze --python "$E/bin/python" > "$LOG/mga-freeze.txt"

sbatch --parsable --partition=normal --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:30:00 \
  --job-name=mga-install-test --output="$LOG/%x-%j.out" --chdir="$SRC" --wrap "
echo host=\$(hostname) job=\$SLURM_JOB_ID start=\$(date -Is)
$E/bin/python -c 'import phono3py, phonopy, symfc, atomate2, jobflow_remote, mga; print(\"phono3py\", phono3py.__version__, \"phonopy\", phonopy.__version__)'
$E/bin/python -m pytest -q tests/test_phono3py_fd.py tests/test_vasp_flows.py 2>&1 | tail -5
echo done \$(date -Is)"
