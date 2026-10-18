"""Bounded third-mesh capacity probe, never a trajectory accuracy result."""
from pathlib import Path
import argparse
import fcntl
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
ap.add_argument('--solver', choices=['mumps','gamg'], default='mumps')
ap.add_argument('--dt', type=float, default=.0025)
ap.add_argument('--history-reference')
ap.add_argument('--history-step', type=int)
ap.add_argument('--inexact-newton', action='store_true')
ap.add_argument('--linear-iterations', type=int, default=1000)
ap.add_argument('--gmres-restart', type=int, default=100)
ap.add_argument('--wait-for-probe')
ap.add_argument('--resume-loaded-probe', help='Prior timed-out EW probe stem; replay its saved fixed histories and unaccepted iterate')
args = ap.parse_args()
assert args.dt > 0 and (args.solver == 'mumps' or args.degree == 2)
assert (args.history_reference is None) == (args.history_step is None)
tag = '' if (args.degree, args.refinement) == (1, 2) else f'_p{args.degree}_r{args.refinement}'
if args.solver != 'mumps':
    tag += '_'+args.solver
if args.history_reference:
    tag += f'_loaded_{args.history_step}'
if args.inexact_newton:
    tag += '_ew'
if args.resume_loaded_probe:
    assert args.inexact_newton and args.history_reference and args.degree == 2
    assert Path(args.resume_loaded_probe).name == args.resume_loaded_probe
    tag += '_resume_' + hashlib.sha256(args.resume_loaded_probe.encode()).hexdigest()[:8]
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
    geo, material, problem = setup(refinement_level=args.refinement, dt=args.dt,
                                   geometry_scales=[1, 1])
    if args.degree == 2:
        parameters = dict(problem.parameters)
        parameters['function_space'] = 'P_2'
        from lv_surrogate.physics.quadratic_solver import make_quadratic_problem
        problem = make_quadratic_problem(geo, material, parameters, args.solver,
            linear_maximum_iterations=args.linear_iterations,gmres_restart=args.gmres_restart)
        if args.solver == 'gamg':
            problem.solver._solver.parameters['absolute_tolerance'] = 1e-8
            problem.solver._solver.parameters['relative_tolerance'] = 0.
    history_description = {}
    pressure = 1000.
    if args.resume_loaded_probe:
        prior_path = P/f'experiments/metrics/{args.resume_loaded_probe}.json'
        prior = json.loads(prior_path.read_text())
        assert prior['stop_reason'] == 'predeclared_capacity_probe_timeout'
        for key, expected in [('displacement_degree',args.degree),('refinement_level',args.refinement),
                              ('native_solver',args.solver),('dt_s',args.dt),('inexact_newton',True),
                              ('linear_maximum_iterations',args.linear_iterations),('gmres_restart',args.gmres_restart)]:
            assert prior[key] == expected, key
        history_description = prior['worker']['history_probe'].copy()
        assert history_description['source_reference'] == args.history_reference
        assert history_description['source_step'] == args.history_step
        hp = P/f'experiments/logs/{args.resume_loaded_probe}_fixed_histories.npz'
        ip = P/f'experiments/logs/{args.resume_loaded_probe}_UNACCEPTED_iterate.npy'
        provenance = {str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [prior_path,hp,ip]}
        with np.load(hp) as saved:
            assert float(saved['dt_s']) == args.dt
            pressure = float(saved['pressure_Pa'])
            assert pressure == prior['worker']['pressure_Pa']
            for name in ['u_old','v_old','a_old']:
                value = saved[name]
                assert value.shape == (problem.u_space.dim(),) and np.isfinite(value).all()
                getattr(problem,name).vector().set_local(value)
                getattr(problem,name).vector().apply('insert')
        iterate = np.load(ip)
        assert iterate.shape == (problem.u_space.dim(),) and np.isfinite(iterate).all()
        problem.u.vector().set_local(iterate); problem.u.vector().apply('insert')
        problem.parameters['p'].assign(pressure)
        replay_R = float(residual(problem).norm('l2'))
        expected_R = prior['worker']['nonlinear_history'][-1]['residual_N']
        np.testing.assert_allclose(replay_R,expected_R,rtol=1e-7,atol=1e-10)
        history_description['resume'] = dict(input_sha256=provenance,replayed_residual_N=replay_R,
            recorded_residual_N=expected_R,scope='Unaccepted initial guess only; fixed mechanics and final force criterion retained')
    elif args.history_reference:
        source_root = P/'data/processed'/args.history_reference
        source_manifest = json.loads((source_root/'manifest.json').read_text())
        cfg = source_manifest['config']; step = args.history_step
        assert 0 <= step < source_manifest['completed_steps']
        assert cfg['geometry_scales'] == [1.,1.] and cfg['dt_s'] == args.dt and cfg['quadrature_degree'] == 6
        values = np.load(source_root/'displacement_coefficients.npy', mmap_mode='r')
        inputs = np.load(source_root/'inputs.npz')
        sg, sm, sb = setup(refinement_level=cfg['refinement_level'],dt=args.dt,geometry_scales=[1,1])
        sp = dict(sb.parameters); sp['function_space'] = 'P_2'
        source = make_quadratic_problem(sg,sm,sp,'gamg')
        np.testing.assert_allclose(source.u_space.tabulate_dof_coordinates(),inputs['dof_coordinates'],atol=1e-14,rtol=0)
        old = np.zeros(values.shape[1]); velocity = old.copy(); acceleration = old.copy()
        beta, gamma = float(source._beta), float(source._gamma)
        prefix_hash = hashlib.sha256()
        for i in range(step):
            current = np.array(values[i],copy=True); prefix_hash.update(current.tobytes())
            new_a = (current-(old+args.dt*velocity+(.5-beta)*args.dt**2*acceleration))/(beta*args.dt**2)
            velocity = velocity+(1-gamma)*args.dt*acceleration+gamma*args.dt*new_a
            acceleration = new_a; old = current
        for name,value in [('u_old',old),('v_old',velocity),('a_old',acceleration),('u',values[step])]:
            f = getattr(source,name); f.vector().set_local(value); f.vector().apply('insert')
        pressure = float(inputs['pressure'][step]); source.parameters['p'].assign(pressure)
        source_R = float(residual(source).norm('l2'))
        assert np.isfinite(source_R) and source_R<1e-6
        for name in ['u_old','v_old','a_old']:
            getattr(problem,name).assign(df.interpolate(getattr(source,name),problem.u_space))
        manufactured = df.Expression(('x[0]*x[0]+x[1]','x[1]*x[2]','x[2]*x[2]-x[0]'),degree=2)
        transferred = df.interpolate(df.interpolate(manufactured,source.u_space),problem.u_space)
        exact = df.interpolate(manufactured,problem.u_space)
        transfer_error = float(np.max(abs(transferred.vector().get_local()-exact.vector().get_local())))
        assert transfer_error<1e-10
        history_description = dict(source_reference=args.history_reference,source_step=step,
            source_time_s=float(inputs['time'][step]),source_native_residual_N=source_R,
            source_config_sha256=source_manifest['config_sha256'],
            source_history_prefix_sha256=prefix_hash.hexdigest(),
            source_current_field_sha256=hashlib.sha256(np.ascontiguousarray(values[step]).tobytes()).hexdigest(),
            manufactured_transfer_maximum_error=transfer_error)
        problem.parameters['p'].assign(pressure)
        previous = problem.u_old.vector().get_local()
        v = problem.v_old.vector().get_local(); a = problem.a_old.vector().get_local()
        best = previous; best_R = float('inf')
        for guess in [previous,previous+args.dt*v,previous+args.dt*v+.5*args.dt**2*a]:
            problem.u.vector().set_local(guess); problem.u.vector().apply('insert')
            R = float(residual(problem).norm('l2'))
            if np.isfinite(R) and R<best_R:
                best = guess; best_R = R
        problem.u.vector().set_local(best); problem.u.vector().apply('insert')
        del source, sg, sm, sb, transferred, exact
    description = dict(tetrahedra=geo.mesh.num_cells(), vertices=geo.mesh.num_vertices(),
                       pressure_Pa=pressure, history_probe=history_description,
                       displacement_dofs=problem.u_space.dim())
    record(phase='solving', **description)
    problem.solver._solver.parameters['line_search'] = 'bt'
    problem.parameters['p'].assign(pressure)
    nonlinear_history = []; linear_history = []
    if args.inexact_newton:
        snes = problem.solver._snes
        snes.setUseEW(True,version=2,rtol_0=.01,rtol_max=.1)
        history_path = P/f'experiments/logs/{stem}_fixed_histories.npz'
        np.savez_compressed(history_path,pressure_Pa=pressure,dt_s=args.dt,
            **{n:getattr(problem,n).vector().get_local() for n in ['u_old','v_old','a_old']})
        def nonlinear_monitor(snes,iteration,norm):
            nonlinear_history.append(dict(iteration=int(iteration),residual_N=float(norm) if np.isfinite(norm) else None,
                KSP_rtol=float(snes.getKSP().getTolerances()[0])))
            path = P/f'experiments/logs/{stem}_UNACCEPTED_iterate.npy'
            temporary = path.with_suffix('.pending.npy')
            np.save(temporary,snes.getSolution().getArray(readonly=True))
            temporary.replace(path)
            record(phase='solving',**description,nonlinear_history=nonlinear_history,
                   linear_history=linear_history,EW_effective=bool(snes.getUseEW()))
            print('LOADED INEXACT NEWTON',iteration,norm,flush=True)
        def linear_monitor(ksp,iteration,norm):
            if iteration%25 == 0:
                linear_history.append(dict(nonlinear_iteration=len(nonlinear_history)-1,
                    linear_iteration=int(iteration),reported_residual=float(norm) if np.isfinite(norm) else None,
                    KSP_rtol=float(ksp.getTolerances()[0])))
                record(phase='solving',**description,nonlinear_history=nonlinear_history,
                       linear_history=linear_history,EW_effective=bool(snes.getUseEW()))
        snes.setMonitor(nonlinear_monitor); snes.getKSP().setMonitor(linear_monitor)
    solve_start = time.perf_counter()
    iterations, _ = problem.solver.solve()
    seconds = time.perf_counter() - solve_start
    reason = int(problem.solver._snes.getConvergedReason())
    force = float(residual(problem).norm('l2'))
    if args.inexact_newton:
        assert problem.solver._snes.getUseEW(), 'Requested inexact-Newton method was not effective'
    record(phase='finished', **description, nonlinear_seconds=seconds,
           setup_and_solve_seconds=time.perf_counter() - start,
           iterations=int(iterations), snes_reason=reason, residual_l2_N=force,
           solve_accepted=bool(reason > 0 and np.isfinite(force) and force < 1e-6),
           nonlinear_history=nonlinear_history,linear_history=linear_history,
           EW_effective=bool(problem.solver._snes.getUseEW()))
    sys.exit(0 if reason > 0 and np.isfinite(force) and force < 1e-6 else 2)

progress.parent.mkdir(parents=True, exist_ok=True)
lock = (progress.parent/f'{stem}.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if output.exists():
    raise RuntimeError('Preserve existing capacity result; use a separately named protocol for another probe')
if args.wait_for_probe:
    progress.write_text(json.dumps(dict(phase='waiting_for_prior_probe',prerequisite=args.wait_for_probe)))
    prerequisite = (progress.parent/f'{args.wait_for_probe}.lock').open('w')
    fcntl.flock(prerequisite,fcntl.LOCK_EX)
    fcntl.flock(prerequisite,fcntl.LOCK_UN); prerequisite.close()
if progress.exists():
    progress.unlink()
env = os.environ.copy()
env.update(FI_PROVIDER='tcp', FI_TCP_IFACE='en0', OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2',
           XDG_CACHE_HOME=str(P / '.cache'), MAMBA_ROOT_PREFIX=str(P / '.mamba'),
           DIJITSO_CACHE_DIR=str(P / '.fenics-cache-activated'))
cmd = [str(P / '.tools/bin/micromamba'), 'run', '-p', str(P / '.fenics'),
       'python', str(Path(__file__).resolve()), '--worker', '--degree', str(args.degree),
       '--refinement', str(args.refinement), '--solver', args.solver, '--dt', str(args.dt)]
if args.history_reference:
    cmd += ['--history-reference',args.history_reference,'--history-step',str(args.history_step)]
cmd += ['--linear-iterations',str(args.linear_iterations),'--gmres-restart',str(args.gmres_restart)]
if args.inexact_newton:
    cmd += ['--inexact-newton']
if args.resume_loaded_probe:
    cmd += ['--resume-loaded-probe',args.resume_loaded_probe]
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
              native_solver=args.solver, dt_s=args.dt,
              inexact_newton=args.inexact_newton, linear_maximum_iterations=args.linear_iterations,
              gmres_restart=args.gmres_restart,
              process_exit_code=child.returncode, stop_reason=stopped,
              measured_wall_seconds=time.perf_counter() - start,
              sampled_peak_worker_RSS_bytes=peak_rss, RSS_sample_interval_s=2,
              declared_limits=dict(wall_seconds=args.timeout, RSS_GiB=args.max_rss_gib,
                                   MUMPS_MB_per_process=20000),
              scope=(f'One loaded-state query using interpolated accepted histories from {args.history_reference}, step{args.history_step}; '
                     if args.history_reference else 'One1000Pa startup query from zero histories; ')+f'dt={args.dt:g}s using {args.solver}; '
                    'same benchmark mechanics; resource feasibility only. '
                    'Does not establish full-cycle convergence, stability or accuracy.')
output.write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps(result, indent=2), flush=True)
