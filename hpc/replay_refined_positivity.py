"""Replay an early rejected refined trial and independently audit its limiting P2 cell."""
import argparse
import fcntl
from pathlib import Path
import re
import shutil

import launch_refined_production as production
import compare_reference_timesteps_555 as refined
from registry import artifact, atomic, read, sha
from submit_probe_profile import assert_idle, registered

ROOT = production.ROOT
ELEMENT = 813578
REASON = 'Whole-element positivity remains unresolved at maximum depth'


def replay_spec(trace, substep):
    assert trace['substep'] == substep
    reason = f'{REASON} (stage=endpoint, element={ELEMENT})'
    rows = trace['custom_newton_steps']
    for index, row in enumerate(rows):
        assert row['iteration'] == index
        for trial_index, trial in enumerate(row['trials']):
            if trial['armijo_passed'] and trial.get('positivity_failure') == reason:
                keys = ('iteration', 'regularization', 'next_regularization', 'initial_force_N',
                        'accepted_custom_update')
                trial_keys = ('alpha', 'native_force_N', 'armijo_passed', 'whole_element_admissible',
                              'positivity_failure')
                prefix = [dict({key: prior[key] for key in keys}, trials=[
                    {key: item.get(key) for key in trial_keys} for item in prior['trials']])
                    for prior in rows[:index+1]]
                assert all(item['accepted_custom_update'] for item in prefix[:-1])
                return dict(substep=substep, element=ELEMENT, stage='endpoint', iteration=index,
                            trial_index=trial_index, alpha=trial['alpha'], reason=reason,
                            selected_predictor=trace['selected_predictor'],
                            initial_forces_N=trace['initial_forces_N'], prefix=prefix,
                            original_depth=20, retry_depth=24, depths=[24, 28, 32, 36])
    raise AssertionError('No unresolved endpoint trial found for the limiting cell')


def replay_mismatches(spec, trace):
    import numpy as np
    errors = []
    rows = trace.get('custom_newton_steps', [])
    if len(rows) != spec['iteration']+1:
        return ['Replay stopped at a different Newton iteration']
    if trace.get('selected_predictor') != spec['selected_predictor']:
        errors.append('Initializer selection changed')

    def near(actual, expected):
        if actual is None or expected is None:
            return actual is expected
        return bool(np.isclose(actual, expected, rtol=1e-6, atol=1e-10))

    if len(trace['initial_forces_N']) != len(spec['initial_forces_N']) or not all(
            near(a, b) for a, b in zip(trace['initial_forces_N'], spec['initial_forces_N'])):
        errors.append('Initializer residuals changed')
    for index, (actual, expected) in enumerate(zip(rows, spec['prefix'], strict=True)):
        for key in ('iteration', 'regularization'):
            if actual[key] != expected[key]:
                errors.append(f'Newton {index}: {key} changed')
        if not near(actual['initial_force_N'], expected['initial_force_N']):
            errors.append(f'Newton {index}: starting residual changed')
        prior = index < spec['iteration']
        if prior:
            for key in ('accepted_custom_update', 'next_regularization'):
                if actual[key] != expected[key]:
                    errors.append(f'Newton {index}: {key} changed')
        count = len(expected['trials']) if prior else spec['trial_index']+1
        if (prior and len(actual['trials']) != count) or len(actual['trials']) < count:
            errors.append(f'Newton {index}: backtracking sequence changed')
            continue
        for trial_index in range(count):
            a, b = actual['trials'][trial_index], expected['trials'][trial_index]
            for key in ('alpha', 'armijo_passed', 'whole_element_admissible', 'positivity_failure'):
                if a.get(key) != b[key]:
                    errors.append(f'Newton {index} trial {trial_index}: {key} changed')
            if not near(a['native_force_N'], b['native_force_N']):
                errors.append(f'Newton {index} trial {trial_index}: force changed')
    return errors


class TrialCapture:
    def __init__(self, audit, spec, trace):
        self.audit, self.spec, self.trace = audit, spec, trace
        self.captured = None
        self.nodal = None

    def track_field(self):
        import numpy as np
        field = self.audit.field
        mapping = self.audit.maps[self.spec['element']]

        def tracked(values):
            self.nodal = np.asarray(values[mapping]).copy()
            return field(values)

        self.audit.field = tracked

    def on_failure(self, stage, element, deformation, vertices, reason):
        trace = self.trace()
        if ('custom_newton_steps' in trace and len(trace['custom_newton_steps']) == self.spec['iteration']
                and stage == self.spec['stage'] and element == self.spec['element']
                and reason == REASON and self.captured is None):
            self.captured = dict(deformation=deformation.copy(), vertices=vertices.copy(),
                                 nodal_displacement=self.nodal.copy(),
                                 canonical_dofs=self.audit.maps[element].copy())


def analyze_cell(values, depths):
    import numpy as np
    from audit_reference_cell_555 import cartesian_deformation, inspect_field
    from lv_surrogate.physics.bernstein import BernsteinAudit

    vertices, nodal = values['vertices'], values['nodal_displacement']
    assert vertices.shape == (4,3) and nodal.shape == (10,3)
    assert np.isfinite(vertices).all() and np.isfinite(nodal).all()
    audit = BernsteinAudit(dict(coords=vertices, cells=np.array([[0,1,2,3]]),
                                canonical_cell_dofs=np.arange(30).reshape(1,10,3)))

    def independent(points):
        return cartesian_deformation(vertices, nodal, points)

    captured = values['deformation']
    primary = inspect_field(audit, captured, independent)
    if any(item['depth'] == 24 for item in primary['bernstein']):
        primary['bernstein'] = [item for item in primary['bernstein'] if item['depth'] >= 24]
    if primary['bernstein'] and primary['bernstein'][-1]['outcome'] == 'unresolved':
        for depth in depths:
            if depth <= 24:
                continue
            audit.max_depth = depth
            try:
                proof = audit.resolve(captured, vertices)
                entry = dict(depth=depth, outcome='certified_positive', **proof)
            except ValueError as exc:
                entry = dict(depth=depth, outcome='nonpositive_witness' if 'nonpositive' in str(exc)
                             else 'unresolved', reason=str(exc))
            primary['bernstein'].append(entry)
            if entry['outcome'] != 'unresolved':
                break
    independent_vertices = independent(vertices)
    # A separate minimum search uses derivatives of the Cartesian P2 interpolant.
    other = inspect_field(audit, independent_vertices)
    difference = float(np.max(abs(independent_vertices-captured)))
    return dict(captured_gradient=primary, independent_cartesian=other,
                independent_vertex_gradient_max_abs_difference=difference,
                independent_gradient_agrees=bool(np.allclose(independent_vertices, captured, rtol=1e-10, atol=1e-10)),
                optimized_minimum_difference=abs(primary['optimized_minimum_estimate']-other['optimized_minimum_estimate']),
                scope='Independent scaled Cartesian quadratic interpolation of the saved ten P2 nodal '
                      'displacements. Numerical minima are estimates; only successful Bernstein '
                      'subdivision supplies a whole-cell positive lower bound. Margin remains 1e-9.')


def finish_replay(run, spec, trace, capture):
    errors = replay_mismatches(spec, trace)
    if capture.captured is None:
        errors.append('The requested rejected cell was not captured')
    report = dict(substep=spec['substep'], element=spec['element'], stage=spec['stage'],
                  iteration=spec['iteration'], alpha=spec['alpha'], margin=1e-9,
                  original_depth=spec['original_depth'], retry_depth=spec['retry_depth'],
                  reproduced=not errors, mismatches=errors, reference_state_accepted=False)
    if capture.captured is not None:
        refined.save_npz(run / 'rejected-cell.npz', **capture.captured)
    if not errors:
        report['cell_audit'] = analyze_cell(capture.captured, spec['depths'])
        entries = report['cell_audit']['captured_gradient']['bernstein']
        if not entries or entries[0]['depth'] != 24 or entries[0]['outcome'] != 'unresolved':
            report['reproduced'] = False
            errors.append('Captured cell did not reproduce unresolved certification at depth 24')
    atomic(run / 'positivity-replay.json', report)
    print('REFINED POSITIVITY REPLAY reproduced', report['reproduced'], flush=True)
    if report.get('cell_audit'):
        cell = report['cell_audit']
        print('INDEPENDENT DETERMINANT minimum_estimate',
              cell['independent_cartesian']['optimized_minimum_estimate'],
              'gradient_agrees', cell['independent_gradient_agrees'], flush=True)
        for result in cell['captured_gradient']['bernstein']:
            print('CELL CERTIFICATION', result, flush=True)


def initialize(run, failed):
    registered(ROOT, failed, failed / 'plan.json')
    registered(ROOT, failed, failed / 'pressure.json')
    state, plan = production.load(failed / 'state.json'), production.load(failed / 'plan.json')
    assert state['status'] == 'blocked' and plan['refined_production'] and not plan.get('positivity_replay')
    assert any(item['kind'] == 'job_submission' and item['return_code'] == 0 and item['stdout'] == state['job']
               for item in read(failed))
    records = production.production_records(failed)
    substep = len(records)+1
    trace_path = failed / f'factor-4/substep-{substep:03d}/trace-slice-{state["slice"]:03d}.json'
    log_path = failed / f'factor-4/command-{state["slice"]:03d}.log'
    assert 'Regularization ceiling without admissible descent' in log_path.read_text()
    spec = replay_spec(production.load(trace_path), substep)
    assert substep == 21 and len(records) == 20
    parent = ROOT / 'experiments/hpc' / plan['parent_refinement']
    qualified = refined.accepted_prefix(parent, 4, completed=True)
    assert [r['checkpoint_sha256'] for r in qualified] == [r['checkpoint_sha256'] for r in records]
    production.initialize(run, parent)
    # Freeze the measured failed path and the full pressure schedule used by production.
    shutil.copyfile(failed / 'pressure.json', run / 'pressure.json')
    artifact(ROOT, run, run / 'pressure.json', 'Preserved production pressure schedule')
    diagnostic = production.load(run / 'plan.json')
    diagnostic.update(positivity_replay=spec, diagnostic_parent=failed.name,
                      scope='Replay one early rejected refined production trial; no reference admission or continuation.')
    for path in (Path(__file__).resolve(), ROOT / 'hpc/audit_reference_cell_555.py',
                 failed / 'plan.json', trace_path, log_path, run / 'pressure.json'):
        diagnostic['frozen'][str(path)] = sha(path)
    atomic(run / 'plan.json', diagnostic)
    artifact(ROOT, run, run / 'plan.json', 'Focused refined positivity replay plan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit',))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--from-production', default='cmbbe-refined-production-01')
    args = parser.parse_args()
    for value in (args.run_id, args.from_production):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Invalid run ID')
    run = ROOT / 'experiments/hpc' / args.run_id
    with (ROOT / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(ROOT)
        initialize(run, ROOT / 'experiments/hpc' / args.from_production)
        state = production.load(run / 'state.json')
        state.update(status='submitted', next_job=production.submit_slice(run, 1))
        atomic(run / 'state.json', state)
