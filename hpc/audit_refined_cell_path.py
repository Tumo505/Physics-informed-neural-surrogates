"""Track the limiting cell through a preserved refined Newton path without advancing production."""
import argparse
import fcntl
from pathlib import Path
import re
import shutil
import threading

import launch_refined_production as production
import compare_reference_timesteps_555 as refined
from replay_refined_positivity import REASON, ELEMENT, analyze_cell, replay_mismatches
from registry import artifact, atomic, read, sha
from submit_probe_profile import assert_idle, registered

ROOT = production.ROOT


def path_spec(trace, maximum_shift):
    assert trace['substep'] == 21
    rows = trace['custom_newton_steps']
    # Stop before the known terminal ceiling rejection. Earlier unsuccessful
    # directions remain part of the path and retain their original policy.
    end = next((i for i, row in enumerate(rows) if not row['accepted_custom_update']
                and row['next_regularization'] >= maximum_shift), len(rows))
    assert end >= 8, 'Require a measured sequence of complete Newton directions'
    rows = rows[:end]
    keys = ('iteration', 'regularization', 'next_regularization', 'initial_force_N', 'accepted_custom_update')
    trial_keys = ('alpha', 'native_force_N', 'armijo_passed', 'whole_element_admissible', 'positivity_failure')
    prefix = []
    for index, row in enumerate(rows):
        assert row['iteration'] == index and row['trials']
        prefix.append(dict({key: row[key] for key in keys}, trials=[
            {key: trial.get(key) for key in trial_keys} for trial in row['trials']]))
    return dict(substep=21, element=ELEMENT, directions=end, iteration=end-1,
                trial_index=len(prefix[-1]['trials'])-1, prefix=prefix,
                selected_predictor=trace['selected_predictor'], initial_forces_N=trace['initial_forces_N'],
                margin=1e-9, maximum_certification_depth=32)


def matching_prefix(spec, trace):
    rows = trace.get('custom_newton_steps', [])
    matched = 0
    for count in range(1, min(len(rows), len(spec['prefix']))+1):
        expected = spec['prefix'][count-1]
        partial_spec = dict(spec, iteration=count-1, prefix=spec['prefix'][:count],
                            trial_index=len(expected['trials'])-1)
        partial_trace = dict(trace, custom_newton_steps=rows[:count])
        if replay_mismatches(partial_spec, partial_trace):
            break
        actual = rows[count-1]
        if len(actual['trials']) != len(expected['trials']):
            break
        if any(actual[key] != expected[key] for key in ('accepted_custom_update', 'next_regularization')):
            break
        matched = count
    return matched


class CellPath:
    def __init__(self, run, audit, spec, trace):
        import numpy as np
        self.run, self.audit, self.spec, self.trace = run, audit, spec, trace
        self.element = spec['element']
        self.mapping = audit.maps[self.element].copy()
        self.vertices = audit.X[self.element].copy()
        gl = audit.gl[self.element]
        self.gradients = np.stack([
            np.concatenate(((4*b[:,None]-1)*gl,
                            np.array([4*(b[i]*gl[j]+b[j]*gl[i]) for i,j in audit.edges])), axis=0)
            for b in np.eye(4)])
        self.frames = []
        self.rejected = set()
        self.lock = threading.Lock()
        self.folder = run / 'cell-frames'
        self.folder.mkdir()
        self.current_nodal = self.old = None

    def add(self, kind, iteration, stage, nodal, force=None, deformation=None, **metadata):
        import numpy as np
        nodal = np.asarray(nodal).copy()
        if deformation is None:
            deformation = np.eye(3)+np.einsum('ki,vkj->vij', nodal, self.gradients)
        with self.lock:
            path = self.folder / f'frame-{len(self.frames):03d}.npz'
            refined.save_npz(path, vertices=self.vertices, nodal_displacement=nodal,
                             deformation=deformation, canonical_dofs=self.mapping)
            row = dict(kind=kind, iteration=iteration, stage=stage, force_N=force,
                       path=str(path.relative_to(self.run)), sha256=sha(path), **metadata)
            self.frames.append(row)
            atomic(self.folder / 'index.json', dict(element=self.element, margin=1e-9, frames=self.frames))

    def start(self, current, previous, force):
        self.old = current[self.mapping].copy()
        self.add('checkpoint', -2, 'endpoint', self.old, force,
                 force_context='Accepted preceding physical substep')
        self.add('checkpoint', -2, 'stress', .4*previous[self.mapping]+.6*self.old)

    def pair(self, kind, iteration, current, force, **metadata):
        nodal = current[self.mapping]
        self.add(kind, iteration, 'endpoint', nodal, force, **metadata)
        self.add(kind, iteration, 'stress', .4*self.old+.6*nodal, force, **metadata)

    def initializer(self, current, force):
        self.pair('initializer', -1, current, force)

    def accepted(self, iteration, current, force, shift):
        self.pair('accepted_newton', iteration, current, force, next_regularization=shift)

    def track_field(self):
        import numpy as np
        original = self.audit.field

        def tracked(values):
            self.current_nodal = np.asarray(values[self.mapping]).copy()
            return original(values)

        self.audit.field = tracked

    def on_failure(self, stage, element, deformation, vertices, reason):
        trace = self.trace()
        if ('custom_newton_steps' not in trace or stage != 'endpoint'
                or element != self.element or reason != REASON):
            return
        iteration = len(trace['custom_newton_steps'])
        if iteration in self.rejected:
            return
        self.rejected.add(iteration)
        self.add('first_unresolved_trial', iteration, stage, self.current_nodal,
                 deformation=deformation.copy(), rejection=reason)

    def finish(self, trace, stop_reason):
        import numpy as np
        edge = (self.vertices[1:]-self.vertices[0]).T
        matched = matching_prefix(self.spec, trace)
        report = dict(complete=True, element=self.element, substep=self.spec['substep'], margin=1e-9,
                      time_s=trace.get('time_s'), pressure_Pa=trace.get('pressure_Pa'),
                      source_trace_sha256=self.spec.get('source_trace_sha256'),
                      requested_directions=self.spec['directions'],
                      replayed_directions=len(trace.get('custom_newton_steps', [])),
                      matched_prefix_directions=matched,
                      reproduced_exactly=matched == self.spec['directions'] == len(trace.get('custom_newton_steps', [])),
                      stop_reason=stop_reason or 'direction_budget', reference_state_accepted=False,
                      geometry=dict(vertices=self.vertices.tolist(), canonical_dofs=self.mapping.tolist(),
                                    signed_reference_volume=float(np.linalg.det(edge)/6),
                                    edge_matrix_condition=float(np.linalg.cond(edge))), frames=[],
                      scope='Independent determinant estimates and existing depth-32 certification along '
                            'the recorded replay. Exact-source path agreement is reported separately. '
                            'No changes to histories, physical acceptance or production depth limits.')
        rows = trace.get('custom_newton_steps', [])
        accepted_iterations = [row['iteration'] for row in rows if row['accepted_custom_update']]
        captured_iterations = [frame['iteration'] for frame in self.frames
                               if frame['kind'] == 'accepted_newton' and frame['stage'] == 'endpoint']
        assert accepted_iterations == captured_iterations, 'Missing accepted Newton cell snapshot'
        for frame in self.frames:
            path = self.run / frame['path']
            assert sha(path) == frame['sha256']
            with np.load(path, allow_pickle=False) as saved:
                values = {name:saved[name] for name in saved.files}
            result = analyze_cell(values, [32])
            row = dict(frame, cell_audit=result)
            iteration = frame['iteration']
            if frame['kind'] == 'first_unresolved_trial':
                assert 0 <= iteration < len(rows), 'Captured trial has no recorded Newton direction'
                reason = f'{REASON} (stage=endpoint, element={self.element})'
                trial = next((t for t in rows[iteration]['trials'] if t.get('positivity_failure') == reason), None)
                assert trial is not None, 'Captured unresolved cell does not match a recorded rejection'
                row.update(force_N=trial['native_force_N'], alpha=trial['alpha'],
                           regularization=rows[iteration]['regularization'])
            report['frames'].append(row)
            atomic(self.run / 'cell-path.json', dict(report, complete=False))
            independent = result['independent_cartesian']
            print('CELL PATH', frame['kind'], 'iteration', iteration, 'stage', frame['stage'],
                  'determinant_estimate', independent['optimized_minimum_estimate'],
                  'smallest_singular_value', independent['singular_values_at_minimum'][-1],
                  'gradient_agrees', result['independent_gradient_agrees'], flush=True)
        atomic(self.run / 'cell-path.json', report)
        print('CELL PATH AUDIT COMPLETE matched_prefix_directions', matched, flush=True)


def initialize(run, failed):
    for name in ('plan.json', 'pressure.json'):
        registered(ROOT, failed, failed / name)
    state, plan = production.load(failed / 'state.json'), production.load(failed / 'plan.json')
    assert state['status'] in ('running', 'blocked') and not plan.get('cell_path_audit')
    assert plan['certified_depth_retry']['depth'] == 32 and not plan.get('positivity_replay')
    assert any(item['kind'] == 'job_submission' and item['return_code'] == 0 and item['stdout'] == state['job']
               for item in read(failed))
    records = production.production_records(failed)
    assert len(records) == 20
    trace = failed / f'factor-4/substep-021/trace-slice-{state["slice"]:03d}.json'
    protocol = production.load(Path(plan['protocol']))
    spec = path_spec(production.load(trace), protocol['admissible_newton']['maximum_regularization'])
    spec['source_trace_sha256'] = sha(trace)
    parent = ROOT / 'experiments/hpc' / plan['parent_refinement']
    qualified = refined.accepted_prefix(parent, 4, completed=True)
    assert [row['checkpoint_sha256'] for row in qualified] == [row['checkpoint_sha256'] for row in records]
    production.initialize(run, parent)
    shutil.copyfile(failed / 'pressure.json', run / 'pressure.json')
    artifact(ROOT, run, run / 'pressure.json', 'Preserved cell-path replay pressure schedule')
    diagnostic = production.load(run / 'plan.json')
    diagnostic.update(cell_path_audit=spec, diagnostic_parent=failed.name,
                      certified_depth_retry=plan['certified_depth_retry'],
                      scope='Independent limiting-cell path audit; no production continuation or state admission.')
    for path in (Path(__file__).resolve(), Path(__file__).with_name('replay_refined_positivity.py'),
                 Path(__file__).with_name('audit_reference_cell_555.py'), failed / 'plan.json',
                 failed / 'state.json', trace, run / 'pressure.json'):
        diagnostic['frozen'][str(path)] = sha(path)
    atomic(run / 'plan.json', diagnostic)
    artifact(ROOT, run, run / 'plan.json', 'Frozen limiting-cell path replay plan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit',))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--from-production', default='cmbbe-refined-production-02')
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
