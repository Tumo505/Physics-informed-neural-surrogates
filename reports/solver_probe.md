# Native cardiac solver probe — 2026-10-16

**Outcome: incomplete / release gate failed.** No converged cardiac displacement/residual was produced, and no physics-guided training result exists.

Installed a separate native osx-arm64 FEniCS environment: fenics-dolfin 2019.1.0, Python 3.11, 198 resolved conda packages (~333 MB downloads). Explicit lock: fenics-environment.lock; Python additions: fenics-pip-lock.txt. Corrected the outdated README assumption that native Apple Silicon packages do not exist. Dataset/solver sources are present; blocker is native execution, not absence of mesh data.

The probe loads the archived ellipsoid (4577 nodes / 17625 tetrahedra), facet labels 1/2/3, and invokes the pinned benchmark's Laplace-based fiber construction. It uses original constitutive, pressure waveform, Robin boundary, inertia/viscosity and generalized-alpha code, with explicitly chosen P1 displacement/fibers, zero activation, dt=.001 s, pressure evaluated at .0006 s. This would be a first-step diagnostic, not reproduction of the full published P2 trajectory.

## Commands and observed errors

1. Direct isolated Python import: `OSError: pkg-config probably not installed` because environment bin directory was not on PATH. Corrected environment PATH.
2. Sandbox MPI import: `MPI_Init_thread ... OFI call ep_enable failed (default nic=utun6: Operation not permitted)`. Retried outside sandbox with host-approved execution.
3. Initial mesh/fiber run progressed to `IndexError: too many indices for array: array is 0-dimensional` in pressure function handling. Corrected with ndarray reshape: benchmark pressure function squeezes single-time results.
4. Corrected pressure run: native process exit 139, last log `Solving linear variational problem.` Tested without MPI-provider override; same exit 139. Python faulthandler did not yield a Python traceback. No unsupported numerical result recorded.
5. Full micromamba activation and separate compiler cache (`make solver-probe`): mesh loaded, native JIT forms compiled, Laplace solve entered, further forms compiled; again exit 139. This does not prove a particular ABI/compiler or solver-library cause. A platform/native-library incompatibility is suspected, not established.

Raw logs retained locally in experiments/logs/cardiac_probe*.log; compiler output retained in ignored .fenics-cache*/. Failures were not counted as successful solves. No physics loss was trained. Initial installed-package detection in benchmark_audit describes the modern .venv; isolated .fenics installation is separate.

Next diagnostic: reduce to a minimal FEniCS Laplace/interpolation example under the explicit environment and obtain a native debugger traceback; then fix/pin the failing native component or run the pinned solver in its recommended Linux container on a suitable host. Only after a converged solve should independent FE residual/Jacobian assembly and pressure/Robin validation proceed. Do not substitute a synthetic cube for cardiac validation.
