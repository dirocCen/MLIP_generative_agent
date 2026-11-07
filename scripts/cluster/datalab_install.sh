#!/bin/bash
# Build one mga worker environment on datalab, then run the calculator smoke test.
# Runs as a batch job on a GPU compute node (compiling and testing are not for the login node,
# DESIGN §11). Usage:  sbatch datalab_install.sh {dpa|tace|pet}
# Versions follow the leaderboard runs (matbench-discovery c70947f, models/*/*.yml):
#   DPA-4.1-Max-OAM : deepmd-kit @ 1ded5d24 (pt backend, CUDA ops), torch 2.13.0+cu130
#   TECE-OAM-RRA-1.0: tace @ 81f65a4c, torch 2.13.0
#   PET-OAM-XL 1.0.0: upet + metatomic-ase (SymmetrizedCalculator), float64
# Python 3.12 for all: phono3py==3.30.0 (kappa_SRME's MS-SMM19 solver) has wheels for it.
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/share/yujie.cen/mga/install_logs/%x-%j.out
set -euo pipefail

WHICH=${1:?usage: datalab_install.sh dpa|tace|pet}
ROOT=/share/yujie.cen/mga
SRC=$ROOT/src/MLIP_generative_agent
E=$ROOT/envs/$WHICH
CK=$ROOT/checkpoints
LOG=$ROOT/install_logs
TORCH_INDEX=https://download.pytorch.org/whl/cu130
export UV_CACHE_DIR=$ROOT/uv_cache UV_LINK_MODE=copy
# Own conda package cache per env, on /share: the three jobs run at the same time and must not
# race on the shared ~/modules/ll/pkgs (11-03 first attempt: InvalidArchiveError / .partial renames),
# and /home is 91% full.
export CONDA_PKGS_DIRS=$ROOT/conda_pkgs/$WHICH
# /tmp on the GPU nodes is not writable for us (uv: "Permission denied ... /tmp/.tmpXXXX")
export TMPDIR=$ROOT/tmp/${SLURM_JOB_ID:-$$}
mkdir -p "$CK" "$LOG" "$CONDA_PKGS_DIRS" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

source ~/modules/ll/etc/profile.d/conda.sh
export PATH=$E/bin:$PATH  # console scripts of the env (dp, mtt) for the calculator loaders
# pip torch loads libstdc++ by name and gets /lib64's (CXXABI <= 1.3.13); conda's libicu (via
# sqlite3 <- mendeleev <- deepmd) then needs 1.3.15 (11-04, dpa job 954836: "dp freeze" failed)
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
UV=~/modules/ll/bin/uv
pipi() { "$UV" pip install --python "$E/bin/python" "$@"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} env=$E start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

[ -d "$E" ] && [ ! -x "$E/bin/python" ] && rm -rf "$E"  # leftover of a failed conda create

case $WHICH in
dpa)
  [ -x "$E/bin/python" ] || conda create -y -p "$E" -c conda-forge python=3.12
  # full CUDA 13.0 toolkit: torch's Caffe2Config (public/cuda.cmake) needs cublas/cufft/curand/
  # cusolver/cusparse/nvtx dev files, not only nvcc + cudart (11-03: "cannot find the CUDA libraries")
  conda install -y -p "$E" -c conda-forge "cuda-toolkit=13.0" cmake ninja
  pipi "torch==2.13.0" --index-url $TORCH_INDEX
  # build-system.requires of deepmd-kit (--no-build-isolation: they must already be in the env)
  pipi numpy setuptools-scm hatch-fancy-pypi-readme "scikit-build-core>=1" packaging dependency-groups "cmake>=3.24" ninja wheel
  # legacy FindCUDA (used by torch's public/cuda.cmake and by deepmd) only looks in $E/include on a
  # native build; conda keeps the headers in targets/x86_64-linux/include (11-03, 5th attempt:
  # "Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS)")
  CUDA_INC=$E/targets/x86_64-linux/include
  export CUDA_HOME=$E CUDAToolkit_ROOT=$E CUDA_PATH=$E CUDA_INC_PATH=$CUDA_INC
  DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=1 DP_ENABLE_PADDLE=0 DP_VARIANT=cuda \
  CMAKE_BUILD_PARALLEL_LEVEL=16 \
    pipi --no-build-isolation -C cmake.define.CMAKE_CUDA_ARCHITECTURES="80;89;90" \
      -C cmake.define.CUDA_TOOLKIT_ROOT_DIR="$E" -C cmake.define.CUDA_TOOLKIT_INCLUDE="$CUDA_INC" \
      "deepmd-kit @ git+https://github.com/deepmodeling/deepmd-kit@1ded5d2429467fb16cb1d6c3d3059fac881f82a7"
  pipi "e3nn==0.6.0"
  CKPT=$CK/dpa-4.1-max-oam-inference.pt
  [ -s "$CKPT" ] || curl -L --fail -o "$CKPT" https://ndownloader.figshare.com/files/70537500
  NAME=dpa41_max_oam LOADER=deepmd
  ;;
tace)
  [ -x "$E/bin/python" ] || conda create -y -p "$E" -c conda-forge python=3.12
  pipi "torch==2.13.0" --index-url $TORCH_INDEX
  pipi "tace @ git+https://github.com/xvzemin/tace@81f65a4c188bd09cec8d1419388f7afdcc1b6fd0"
  CKPT=$CK/TECE-OAM-RRA-1.0.pt
  [ -s "$CKPT" ] || curl -L --fail -o "$CKPT" https://huggingface.co/xvzemin/tace-foundations/resolve/main/TECE-OAM-RRA-1.0.pt
  NAME=tece_oam_rra LOADER=tace
  ;;
pet)
  [ -x "$E/bin/python" ] || conda create -y -p "$E" -c conda-forge python=3.12
  pipi "torch==2.13.0" --index-url $TORCH_INDEX
  pipi upet metatomic-ase metatrain
  CKPT=$CK/pet-oam-xl-v1.0.0.ckpt
  [ -s "$CKPT" ] || curl -L --fail -o "$CKPT" https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-oam-xl-v1.0.0.ckpt
  NAME=pet_oam_xl LOADER=upet
  ;;
*) echo "unknown env $WHICH"; exit 2 ;;
esac

# mga itself + worker extras (phono3py 3.30.0, phonopy 3.5.x, symfc), without touching torch
pipi -e "$SRC[worker]"
"$E/bin/python" - <<'EOF'
import torch, phono3py, symfc, mga
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("phono3py", phono3py.__version__, "symfc", symfc.__version__)
EOF
"$UV" pip freeze --python "$E/bin/python" > "$LOG/$WHICH-freeze.txt"

cd "$LOG"
"$E/bin/python" "$SRC/scripts/cluster/smoke_calculator.py" --name "$NAME" --loader "$LOADER" \
  --checkpoint "$CKPT" --out "$LOG/$WHICH-smoke.json"
echo "done $(date -Is)"
