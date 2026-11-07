#!/bin/bash
# Build FourPhonon (CPU: MPI + OpenMP) on datalab with a conda-forge GNU toolchain.
# datalab has no Fortran compiler or MPI outside conda; FourPhonon's arch.make assumes Intel.
#
#   sbatch -p CPU-amd -N1 -c 16 -t 01:00:00 -J mga-fourphonon-build scripts/cluster/fourphonon_install.sh
#
# Result: $OPT/FourPhonon/ShengBTE_cpu, run with scripts/cluster/fourphonon_run.sh
set -euo pipefail
OPT=/share/yujie.cen/mga/opt
E=/share/yujie.cen/mga/envs/fourphonon
SPGLIB_TAG=v2.5.0
export TMPDIR=/share/yujie.cen/mga/tmp && mkdir -p "$TMPDIR"

source ~/modules/ll/etc/profile.d/conda.sh
if [ ! -x "$E/bin/mpifort" ]; then
  conda create -y -p "$E" -c conda-forge --override-channels \
    compilers "openmpi>=4,<6" openblas cmake make
fi
conda activate "$E"
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
mpifort --version | head -1

# spglib C library (static): ShengBTE binds the old C API (spg_get_symmetry, ...) directly
if [ ! -f "$OPT/spglib/lib/libsymspg.a" ]; then
  rm -rf "$TMPDIR/spglib-src"
  git clone -q --depth 1 -b "$SPGLIB_TAG" https://github.com/spglib/spglib.git "$TMPDIR/spglib-src"
  cmake -S "$TMPDIR/spglib-src" -B "$TMPDIR/spglib-build" -DCMAKE_BUILD_TYPE=Release \
    -DSPGLIB_SHARED_LIBS=OFF -DSPGLIB_WITH_Fortran=OFF -DSPGLIB_WITH_TESTS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="$OPT/spglib" -DCMAKE_INSTALL_LIBDIR=lib
  cmake --build "$TMPDIR/spglib-build" -j "${SLURM_CPUS_PER_TASK:-8}"
  cmake --install "$TMPDIR/spglib-build"
fi

cd "$OPT/FourPhonon/Src"
git -C "$OPT/FourPhonon" log --oneline -1
# gfortran rejects `x.eq. .false.` on logicals (an Intel extension). Not `.eqv.`: it binds looser
# than `.and.`, so `a.and.b.eq. .false.` would change meaning. Rewrite per identifier instead:
# `b.eq. .false.` -> `(.not.b)`, `b.eq. .true.` -> `(b)` (10 places in config.f90 / ShengBTE.f90).
sed -i -E -e 's/([A-Za-z_][A-Za-z0-9_]*)\.eq\.[[:space:]]*\.false\./(.not.\1)/Ig' \
          -e 's/([A-Za-z_][A-Za-z0-9_]*)\.eq\.[[:space:]]*\.true\./(\1)/Ig' ./*.f90
# Off-by-one in the six 4ph sampling loops (processes.f90 @ 5b814b3): `iter=0` with
# `matrix_iter = (iter-1)*5` reads rand_matrix(-4:0) on the first pass, a garbage q index follows and
# gfortran builds segfault (job 963763, Example-Si-sampling). Starting at 1 gives exactly N samples,
# which is also what the N_x / P_x normalisation divides by. These are the only `iter=0` lines.
sed -i -E 's/^([[:space:]]*)iter=0[[:space:]]*$/\1iter=1/' processes.f90
git -C "$OPT/FourPhonon" diff --stat
git -C "$OPT/FourPhonon" diff | grep -E "^[-+] " || true
make clean >/dev/null 2>&1 || true
# gfortran instead of mpiifort; -fallow-argument-mismatch for the untyped MPI buffers (gfortran >= 10)
make cpu -j1 \
  CPU_COMPILER=mpifort \
  CPU_FFLAGS="-fopenmp -O2 -cpp -DCPU_VERSION -ffree-line-length-none -fallow-argument-mismatch" \
  LDFLAGS="-L$OPT/spglib/lib -lsymspg" \
  LIBS="-L$E/lib -lopenblas"
ls -l "$OPT/FourPhonon/ShengBTE_cpu"
ldd "$OPT/FourPhonon/ShengBTE_cpu" | grep -E "openblas|mpi|gomp|not found" || true
echo BUILD_OK
