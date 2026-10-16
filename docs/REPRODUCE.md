# Reproduction (tested on macOS arm64)

From /Users/tumokgabeng/Projects/CMBBE/cardiac-lv-surrogate:

```sh
# Existing isolated environment is ready.
make smoke
make reproduce
make train
make evaluate
make benchmark
make figures
make report
```

Fresh environment (Python 3.12; do not replace system Python):

```sh
/path/to/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

`make audit` verifies both original archive MD5s, scans arrays, extracts only LV data after path-traversal validation, records hashes and partitions. It requires the already downloaded lvData.zip and benchmark_article_data.zip one directory above. Expanded LV footprint ~2.58 GB. `make inspect-benchmark` extracts selected 1.17 MB ellipsoid geometry, not all simulation snapshots. Source symlinks point to sibling user checkouts; exact SHAs in audit. Never modify or update those checkouts silently.

`make reproduce` evaluates all 567 available varying-geometry test cases using the unedited author model on modern Flax, and all 246 fixed-geometry parameter cases using a numerically cross-checked NumPy adapter. The original Python 3.9/JAX 0.3/Flax 0.3 environment has not been reproduced. CPU/JAX package versions differ; report numerical compatibility rather than identical legacy software reproduction.

`make train` overwrites pilot checkpoints/predictions with the prespecified five-seed run (~two minutes on this host; timing depends on concurrent work). It trains real graph networks on 64 real LV FEM cases. `make evaluate` requires all five checkpoints/prediction files. Calibration is disjoint from validation and test. No node-level independent-case statistics.

`make mechanics-verify` tests synthetic unit mechanics and affine kinematics on the authentic benchmark mesh. Passing this target does NOT pass cardiac equilibrium release gates. Trusted cardiac solver checks are tracked separately.

Checkpoints and large predictions are excluded from Git but retained locally. Scripts regenerate them from pinned data/source and saved configs; manifests record hashes. Plot sources are saved prediction arrays, not generated illustrations. Small JSON metrics and plotted example fields are versioned. Commands use local relative paths only after entering the project directory.

Native cardiac probe (experimental, may require host approval for MPI startup): `make solver-probe`. Separate conda explicit package lock is fenics-environment.lock; extra Python packages in fenics-pip-lock.txt. Recreate with project-local micromamba `create -p "$PWD/.fenics" --file fenics-environment.lock`, then install the locked pip-only additions. Do not mix .fenics compiled packages into .venv.

For the bootstrap used here, micromamba was downloaded from https://micro.mamba.pm/api/micromamba/osx-arm64/latest and placed in .tools/bin. Exact resolved FEniCS packages are in the explicit lock rather than relying on that rolling URL. No system package installation was performed. Gmsh is absent; probe loads archived mesh rather than regenerating a geometry. Full activated environment and separate compiler cache are used to avoid PATH/ABI ambiguity. Check reports/solver_probe.md for attained outcome and native failures.
