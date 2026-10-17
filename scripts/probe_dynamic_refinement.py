"""Bounded third-mesh capacity probe, never a trajectory accuracy result."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

P = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--worker', action='store_true')
ap.add_argument('--timeout', type=float, default=300)
ap.add_argument('--max-rss-gib', type=float, default=30)
ap.add_argument('--degree', type=int, choices=[1, 2], default=1)
ap.add_argument('--refinement', type=int, choices=[0, 1, 2], default=2)
args = ap.parse_args()
tag = '' if (args.degree, args.refinement) == (1, 2) else f'_p{args.degree}_r{args.refinement}'
stem = 'dynamic_refinement_capacity' + tag
progress = P / f'experiments/logs/{stem}_progress.json'
output = P / f'experiments/metrics/{stem}.json'
source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

if args.worker:
    def record(**kwargs):
        progress.write_text(json.dumps(dict(pid=os.getpid(), **kwargs), indent=2))

    record(phase='initializing')
    sys.path.insert(0, str(P / 'src'))
    from lv_surrogate.physics.reference_fem import setup, residual, df, LVProblem
    import numpy as np

    # MUMPS allocation cap supplements the parent process's total-RSS guard.
    df.PETScOptions.set('mat_mumps_icntl_23', 20000)
    start = time.perf_counter()
    geo, material, problem = setup(refinement_level=args.refinement, dt=.0025,
                                   geometry_scales=[1, 1])
    if args.degree == 2:
        parameters = dict(problem.parameters)
        parameters['function_space'] = 'P_2'
        problem = LVProblem(geo, material, parameters=parameters,
                            solver_parameters={'linear_solver': 'mumps',
                                               'petsc': {'ksp_type': 'preonly',
                                                         'pc_type': 'none'}})
    description = dict(tetrahedra=geo.mesh.num_cells(), vertices=geo.mesh.num_vertices(),
                       displacement_dofs=problem.u_space.dim())
    record(phase='solving', **description)
    problem.solver._solver.parameters['line_search'] = 'bt'
    problem.parameters['p'].assign(1000.)
    solve_start = time.perf_counter()
    iterations, _ = problem.solver.solve()
    seconds = time.perf_counter() - solve_start
    reason = int(problem.solver._snes.getConvergedReason())
    force = float(residual(problem).norm('l2'))
    record(phase='finished', **description, nonlinear_seconds=seconds,
           setup_and_solve_seconds=time.perf_counter() - start,
           iterations=int(iterations), snes_reason=reason, residual_l2_N=force,
           solve_accepted=bool(reason > 0 and np.isfinite(force) and force < 1e-6))
    sys.exit(0 if reason > 0 and np.isfinite(force) and force < 1e-6 else 2)

progress.parent.mkdir(parents=True, exist_ok=True)
if progress.exists():
    progress.unlink()
env = os.environ.copy()
env.update(FI_PROVIDER='tcp', FI_TCP_IFACE='en0', OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2',
           XDG_CACHE_HOME=str(P / '.cache'), MAMBA_ROOT_PREFIX=str(P / '.mamba'),
           DIJITSO_CACHE_DIR=str(P / '.fenics-cache-activated'))
cmd = [str(P / '.tools/bin/micromamba'), 'run', '-p', str(P / '.fenics'),
       'python', str(Path(__file__).resolve()), '--worker', '--degree', str(args.degree),
       '--refinement', str(args.refinement)]
start = time.perf_counter()
peak_rss = 0
stopped = None
with (progress.parent / f'{stem}.log').open('w') as log:
    child = subprocess.Popen(cmd, cwd=P, env=env, stdout=log, stderr=subprocess.STDOUT)
    worker = {}
    while child.poll() is None:
        try:
            worker = json.loads(progress.read_text())
            rss = int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(worker['pid'])],
                                             text=True).strip()) * 1024
            peak_rss = max(peak_rss, rss)
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError,
                ValueError):
            pass
        if time.perf_counter() - start > args.timeout:
            stopped = 'predeclared_capacity_probe_timeout'
        if peak_rss > args.max_rss_gib * 2**30:
            stopped = 'predeclared_total_RSS_limit'
        if stopped:
            # Signal only the worker launched by this process, then its wrapper.
            if worker.get('pid'):
                try:
                    os.kill(worker['pid'], 15)
                except ProcessLookupError:
                    pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if worker.get('pid'):
                    try:
                        os.kill(worker['pid'], 9)
                    except ProcessLookupError:
                        pass
                child.kill()
                child.wait()
            break
        time.sleep(2)
try:
    worker = json.loads(progress.read_text())
except (FileNotFoundError, json.JSONDecodeError):
    worker = {}
result = dict(engineering_only=True, complete=True, worker=worker,
              executed_source_sha256=source_hash, displacement_degree=args.degree,
              refinement_level=args.refinement,
              process_exit_code=child.returncode, stop_reason=stopped,
              measured_wall_seconds=time.perf_counter() - start,
              sampled_peak_worker_RSS_bytes=peak_rss, RSS_sample_interval_s=2,
              declared_limits=dict(wall_seconds=args.timeout, RSS_GiB=args.max_rss_gib,
                                   MUMPS_MB_per_process=20000),
              scope='One 1000 Pa startup query from zero histories at dt=2.5 ms; '
                    'same benchmark mechanics; resource feasibility only. '
                    'Does not establish full-cycle convergence, stability or accuracy.')
output.write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps(result, indent=2), flush=True)
