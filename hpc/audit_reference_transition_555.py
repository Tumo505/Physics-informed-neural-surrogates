"""Trace force contributions and dynamic histories through accepted states 550–554."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'hpc'), str(ROOT / 'src')]
from registry import artifact, atomic, event, sha
from submit_probe_profile import assert_idle, registered

TARGET = ROOT / 'data/processed/cardiac_hpc_cmbbe-cofactor-ref-03'
ASSEMBLY_AUDIT = ROOT / 'experiments/hpc/cmbbe-residual-audit-555-01'
NATIVE = ROOT / '.hpc/native-v2'
ELEMENT = 813578


def load(path):
    return json.loads(path.read_text())


def submit(run):
    with (ROOT / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(ROOT)
        for path in (ASSEMBLY_AUDIT / 'plan.json', ASSEMBLY_AUDIT / 'result.json'):
            registered(ROOT, ASSEMBLY_AUDIT, path)
        previous = load(ASSEMBLY_AUDIT / 'plan.json')
        assert load(ASSEMBLY_AUDIT / 'result.json')['passed']
        assert load(TARGET / 'manifest.json')['completed_steps'] == 554
        assert sha(TARGET / 'restart.npz') == previous['checkpoint_sha256']
        frozen = dict(previous['frozen'])
        for path, digest in frozen.items():
            assert sha(Path(path)) == digest, path
        for path in (Path(__file__), ASSEMBLY_AUDIT / 'result.json',
                     ROOT / 'hpc/audit_reference_cell_555.py'):
            frozen[str(path)] = sha(path)
        run.mkdir()
        atomic(run / 'plan.json', dict(seed=previous['seed'], protocol=previous['protocol'],
                                      checkpoint_sha256=previous['checkpoint_sha256'], frozen=frozen))
        artifact(ROOT, run, run / 'plan.json', 'Force and history transition audit plan')
        script = run / 'job.pbs'
        command = [str(ROOT / '.venv/bin/python'), str(Path(__file__).resolve()),
                   'execute', '--run-id', run.name]
        script.write_text('#!/bin/bash\nset -eu\ncd ' + shlex.quote(str(ROOT))
                          + '\nexec ' + shlex.join(command) + '\n')
        artifact(ROOT, run, script, 'PBS transition audit command')
        atomic(run / 'state.json', dict(status='submitted'))
        reply = subprocess.run(
            ['qsub', '-N', 'lv_transition', '-q', 'nvidia',
             '-l', 'select=1:ncpus=28:mem=240gb', '-l', 'place=excl',
             '-l', 'software=PBSJobApp', '-l', 'walltime=168:00:00',
             '-j', 'oe', '-o', str(run / 'pbs.log'), str(script)],
            text=True, capture_output=True)
        event(run, 'job_submission', return_code=reply.returncode,
              stdout=reply.stdout.strip(), stderr=reply.stderr.strip())
        reply.check_returncode()
        print('Transition audit job:', reply.stdout.strip())


def execute(run):
    assert os.environ.get('PBS_JOBID'), 'Use a PBS compute job'
    plan = load(run / 'plan.json')
    state = dict(status='running', job=os.environ['PBS_JOBID'])
    with (run / 'execution.lock').open('a') as execution, (TARGET / 'generation.lock').open('a') as generation:
        for lock in (execution, generation):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            registered(ROOT, run, run / 'plan.json')
            for path, digest in plan['frozen'].items():
                assert sha(Path(path)) == digest, path
            atomic(run / 'state.json', state)
            env = os.environ.copy()
            for key in ('LD_PRELOAD', 'PETSC_DIR', 'PETSC_ARCH', 'PETSC_OPTIONS', 'PKG_CONFIG_SYSROOT_DIR'):
                env.pop(key, None)
            pkg = f'{NATIVE}/lib/pkgconfig:{NATIVE}/share/pkgconfig'
            env.update(PATH=f'{NATIVE}/bin:{env["PATH"]}', LD_LIBRARY_PATH=f'{NATIVE}/lib',
                       PKG_CONFIG=str(NATIVE / 'bin/pkg-config'), PKG_CONFIG_PATH=pkg, PKG_CONFIG_LIBDIR=pkg,
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='', DIJITSO_CACHE_DIR=str(ROOT / '.hpc/fenics-cache'),
                       XDG_CACHE_HOME=str(ROOT / '.hpc/cache'))
            with (run / 'audit.log').open('x') as log:
                subprocess.run([str(NATIVE / 'bin/python'), str(Path(__file__).resolve()),
                                'audit', '--run-id', run.name], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            report = load(run / 'result.json')
            assert report['complete']
            assert sha(TARGET / 'restart.npz') == plan['checkpoint_sha256']
            assert sha(TARGET / 'manifest.json') == plan['frozen'][str(TARGET / 'manifest.json')]
            artifact(ROOT, run, run / 'result.json', 'Force and history transition measurements')
            state['status'] = 'audit_complete'
        except Exception as exc:
            state.update(status='blocked', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            atomic(run / 'state.json', state)


def component_forms(problem):
    from lv_surrogate.physics.reference_fem import ufl

    from cardiac_benchmark.problem import interpolate
    w = problem.u_test
    p = problem.parameters
    a_new = problem.a(problem.u, problem.u_old, problem.v_old, problem.a_old)
    v_new = problem.v(a_new, problem.v_old, problem.a_old)
    a = interpolate(problem.a_old, a_new, p['alpha_m'])
    u = interpolate(problem.u_old, problem.u, p['alpha_f'])
    v = interpolate(problem.v_old, v_new, p['alpha_f'])
    F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
    elastic = ufl.diff(problem.material.strain_energy(F), F)
    total_material = problem._first_piola(F, v)
    epi = ufl.dot(p['alpha_epi'] * u, problem.N) + ufl.dot(p['beta_epi'] * v, problem.N)
    top = p['alpha_top'] * u + p['beta_top'] * v
    return dict(
        inertia=problem._acceleration_form(a, w),
        elastic=ufl.inner(elastic, ufl.grad(w)) * ufl.dx,
        viscous=ufl.inner(total_material - elastic, ufl.grad(w)) * ufl.dx,
        pressure=problem._pressure_term(F, w),
        epicardial_boundary=ufl.inner(epi * w, problem.N) * problem.ds(problem.epi),
        basal_boundary=ufl.inner(top, w) * problem.ds(problem.top),
    )


def vector_summary(values, total, coordinates, cell_dofs):
    import numpy as np

    values = np.asarray(values)
    count = min(8, len(values))
    ids = np.argpartition(abs(values), len(values) - count)[-count:]
    ids = ids[np.argsort(abs(values[ids]))[::-1]]
    norm = float(np.linalg.norm(values))
    total_norm = float(np.linalg.norm(total))
    return dict(l2_N=norm, max_abs_N=float(np.max(abs(values))),
                dot_total_N2=float(values @ total),
                projection_on_total_N=float(values @ total / total_norm) if total_norm else 0.,
                limiting_cell_dof_l2_N=float(np.linalg.norm(values[cell_dofs])),
                largest_entries=[dict(dof=int(i), value_N=float(values[i]),
                                      coordinate=coordinates[i].tolist(),
                                      on_limiting_cell=bool(i in cell_dofs)) for i in ids])


def audit(run):
    import hashlib
    import numpy as np
    from lv_surrogate.physics.parallel_fem import ParallelP2
    from lv_surrogate.physics.reference_fem import df, residual
    from lv_surrogate.physics.bernstein import BernsteinAudit
    from audit_reference_cell_555 import cartesian_deformation, inspect_field

    plan = load(run / 'plan.json')
    manifest = load(TARGET / 'manifest.json')
    assert manifest['completed_steps'] == 554
    protocol = load(Path(plan['protocol']))
    adapter = ParallelP2(ROOT, protocol['source_reference'], manifest['config'])
    assert adapter.comm.size == 1
    problem = adapter.problem
    with np.load(TARGET / 'inputs.npz', allow_pickle=False) as inputs:
        pressures, times = inputs['pressure'], inputs['time']
        coordinates = inputs['dof_coordinates']
        cell_map = inputs['canonical_cell_dofs'][ELEMENT]
        cell_dofs = cell_map.ravel()
        vertices = inputs['coords'][inputs['cells'][ELEMENT]]
    local_audit = BernsteinAudit(dict(coords=vertices, cells=np.array([[0, 1, 2, 3]]),
                                      canonical_cell_dofs=cell_map[None]))
    fields = np.load(TARGET / 'displacement_coefficients.npy', mmap_mode='r')
    with np.load(plan['seed'], allow_pickle=False) as seed:
        warm = seed['u']
        assert int(seed['step']) == 554 and seed['checkpoint_sha256'].item() == plan['checkpoint_sha256']
    checkpoint = np.load(TARGET / 'restart.npz', allow_pickle=False)
    histories = ('u_old', 'v_old', 'a_old')
    for name in ('u', *histories):
        getattr(problem, name).vector().zero()
    dt, beta, gamma = map(float, (problem.parameters['dt'], problem._beta, problem._gamma))
    forms = component_forms(problem)
    result = dict(complete=False, accepted_states=554, first_state=550, last_state=555,
                  dt_s=dt, beta=beta, gamma=gamma, force_target_N=1e-8,
                  reference_state_accepted=False, records=[], history_updates=[],
                  scope='Serial force decomposition using the cached mesh and original forms. '
                        'Histories are replayed from the initial zero histories through every accepted displacement. '
                        'Force components must reconstruct the full assembled vector.')

    def canonical(vector):
        out = np.empty(adapter.ndof)
        out[adapter.mapping] = vector.get_local()
        return out

    def state_stats():
        return {name: dict(max_abs=float(np.max(abs(getattr(problem, name).vector().get_local()))),
                           l2=float(getattr(problem, name).vector().norm('l2')))
                for name in histories}

    def evaluate(state, label):
        full = canonical(residual(problem))
        parts = {name: canonical(df.assemble(form)) for name, form in forms.items()}
        summed = sum(parts.values())
        error = float(np.linalg.norm(summed - full))
        scale = max(1., sum(float(np.linalg.norm(v)) for v in parts.values()))
        assert error <= 1e-10 * scale, (state, label, error, scale)
        row = dict(state=state, candidate=label, time_s=float(times[state-1]),
                   pressure_Pa=float(pressures[state-1]), full_force_N=float(np.linalg.norm(full)),
                   decomposition_error_N=error, component_sum_norm_N=float(np.linalg.norm(summed)),
                   histories=state_stats(), total=vector_summary(full, full, coordinates, cell_dofs),
                   components={name: vector_summary(values, full, coordinates, cell_dofs)
                               for name, values in parts.items()})
        canonical_u = adapter.gather('u')
        canonical_old = adapter.gather('u_old')
        alpha_f = float(problem.parameters['alpha_f'])
        row['limiting_cell'] = {}
        for stage, displacement in (
                ('endpoint', canonical_u),
                ('stress', alpha_f*canonical_old + (1-alpha_f)*canonical_u)):
            nodal = displacement[cell_map]
            deformation = cartesian_deformation(vertices, nodal, vertices)
            measurement = inspect_field(local_audit, deformation)
            row['limiting_cell'][stage] = dict(
                minimum_estimate=measurement['optimized_minimum_estimate'],
                barycentric=measurement['minimum_barycentric'],
                singular_values=measurement['singular_values_at_minimum'],
                bernstein=measurement['bernstein'])
        row['component_norm_sum_N'] = sum(v['l2_N'] for v in row['components'].values())
        row['cancellation_ratio'] = row['component_norm_sum_N'] / max(row['full_force_N'], 1e-30)
        result['records'].append(row)
        atomic(run / 'result.json', result)
        print('TRANSITION FORCE', state, label, row['full_force_N'],
              {name: component['l2_N'] for name, component in row['components'].items()}, flush=True)

    for index in range(554):
        if index >= 549:
            journal = load(TARGET / f'step_journal/{index:06d}.json')
            assert journal['record'] == manifest['steps'][index]
            assert hashlib.sha256(np.asarray(fields[index]).tobytes()).hexdigest() == journal['coefficient_sha256']
            old = problem.u_old.vector().get_local().copy()
            prior_v = problem.v_old.vector().get_local().copy()
            prior_a = problem.a_old.vector().get_local().copy()
            adapter.assign('u', fields[index-1])
            problem.parameters['p'].assign(float(pressures[index]))
            evaluate(index + 1, 'previous_accepted_displacement')
        adapter.assign('u', fields[index])
        if index >= 549:
            evaluate(index + 1, 'accepted_displacement')
            u = problem.u.vector().get_local().copy()
            expected_a = (u - old - dt*prior_v - (.5-beta)*dt**2*prior_a) / (beta*dt**2)
            expected_v = prior_v + (1-gamma)*dt*prior_a + gamma*dt*expected_a
        problem._update_fields()
        if index >= 549:
            a_error = float(np.max(abs(problem.a_old.vector().get_local() - expected_a)))
            v_error = float(np.max(abs(problem.v_old.vector().get_local() - expected_v)))
            result['history_updates'].append(dict(
                state=index+1, displacement_increment_max_abs=float(np.max(abs(u-old))),
                acceleration_max_abs=float(np.max(abs(expected_a))),
                velocity_max_abs=float(np.max(abs(expected_v))),
                independent_acceleration_max_abs_error=a_error, independent_velocity_max_abs_error=v_error))
        if (index+1) % 64 == 0:
            print('HISTORY REPLAY', index+1, '/', 554, flush=True)
    differences = {}
    for name in histories:
        actual = adapter.gather(name)
        diff = actual - checkpoint[name]
        differences[name] = dict(max_abs=float(np.max(abs(diff))),
                                 relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(checkpoint[name]),1e-30)),
                                 matches=bool(np.allclose(actual,checkpoint[name],rtol=1e-7,atol=1e-10)))
    result['replayed_final_history_vs_checkpoint'] = differences
    # Examine state 555 with the exact durable histories, even if replay reveals a discrepancy.
    for name in histories:
        adapter.assign(name, checkpoint[name])
    problem.parameters['p'].assign(float(pressures[554]))
    adapter.assign('u', checkpoint['u_old'])
    evaluate(555, 'checkpoint_displacement')
    adapter.assign('u', warm)
    evaluate(555, 'original_warm_start')
    result['accepted_force_failures'] = [
        dict(state=row['state'], force_N=row['full_force_N'])
        for row in result['records'] if row['candidate'] == 'accepted_displacement' and row['full_force_N'] >= 1e-8
    ]
    result['first_replayed_accepted_force_failure'] = (
        result['accepted_force_failures'][0] if result['accepted_force_failures'] else None)
    result['complete'] = True
    atomic(run / 'result.json', result)
    print('TRANSITION AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'execute', 'audit'))
    parser.add_argument('--run-id', default='cmbbe-transition-audit-555-01')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('Invalid run ID')
    run = ROOT / 'experiments/hpc' / args.run_id
    if args.action == 'submit':
        submit(run)
    elif args.action == 'execute':
        execute(run)
    else:
        audit(run)
