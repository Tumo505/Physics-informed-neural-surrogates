#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PBS_JOBID:-}" && "${CMBBE_ALLOW_GATEWAY_INSTALL:-0}" != 1 ]]; then
  echo 'Submit this script as a PBS setup job; no installation or compilation on the gateway by default.' >&2
  exit 2
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'These locks require Linux x86_64.' >&2; exit 2; }
mkdir -p .tools/bin .hpc/cache .hpc/setup-records experiments/logs
export MAMBA_ROOT_PREFIX="$PWD/.hpc/mamba"
export XDG_CACHE_HOME="$PWD/.hpc/cache"
export MPLCONFIGDIR="$PWD/.hpc/cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ ! -x .tools/bin/micromamba ]]; then
  curl --fail --location https://micro.mamba.pm/api/micromamba/linux-64/latest --output .hpc/micromamba.tar.bz2
  tar -xjf .hpc/micromamba.tar.bz2 -C .tools bin/micromamba
fi
.tools/bin/micromamba --version
getconf GNU_LIBC_VERSION
for pair in 'native .fenics' 'modern .hpc/modern'; do
  read -r kind prefix <<< "$pair"
  if [[ ! -d "$prefix/conda-meta" ]]; then
    .tools/bin/micromamba create --yes --prefix "$PWD/$prefix" --file "hpc/$kind-linux64-glibc217.lock"
  fi
  .tools/bin/micromamba list --prefix "$PWD/$prefix" --explicit > ".hpc/setup-records/$kind-installed.lock"
done
if [[ ! -e .venv ]]; then ln -s .hpc/modern .venv; fi
.venv/bin/python hpc/check_environment_locks.py
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
# Repository external entries are relative symlinks, not portable submodules.
parent="$(dirname "$PWD")"
for entry in 'cardiac_benchmark https://github.com/finsberg/cardiac_benchmark.git c7d2d1ba6d680627dc465598dbf6b0363a75d604' 'passive-lv-gnn-emul https://github.com/dodaltuin/passive-lv-gnn-emul.git 449cc22459ba5ada344fe0ff238e6b586d5e9319'; do
  read -r name url revision <<< "$entry"
  if [[ ! -d "$parent/$name/.git" ]]; then
    git clone "$url" "$parent/$name"
    git -C "$parent/$name" checkout --detach "$revision"
  fi
  [[ "$(git -C "$parent/$name" rev-parse HEAD)" == "$revision" ]] || { echo "Wrong external revision: $name" >&2; exit 2; }
done
.tools/bin/micromamba run -p "$PWD/.fenics" python hpc/preflight_native.py > .hpc/setup-records/native-preflight.json
.venv/bin/python -c 'import sys,numpy,scipy,matplotlib,h5py,torch; print(sys.version); print("torch",torch.__version__,"CUDA available",torch.cuda.is_available())' > .hpc/setup-records/modern-preflight.txt
.venv/bin/python hpc/registry.py setup --root "$PWD"
echo 'Environment installed. Native numerical equivalence must still pass on the compute node.'
