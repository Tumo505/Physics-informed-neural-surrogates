import copy
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import replay_refined_positivity as replay
from test_hpc_refined_production import production_run
from registry import atomic, event, sha


@pytest.fixture
def failed_trace():
    def trial(alpha, force, reason=None):
        return dict(alpha=alpha, native_force_N=force, armijo_passed=True,
                    whole_element_admissible=reason is None, positivity_failure=reason)

    reason = f'{replay.REASON} (stage=endpoint, element={replay.ELEMENT})'
    return dict(substep=21, selected_predictor=2, initial_forces_N=[1., .8, .6, 1.2],
                custom_newton_steps=[
                    dict(iteration=0, regularization=1., next_regularization=1., initial_force_N=.6,
                         accepted_custom_update=True, trials=[trial(1., .4)]),
                    dict(iteration=1, regularization=1., next_regularization=4., initial_force_N=.4,
                         accepted_custom_update=True, trials=[trial(1., .3, reason), trial(.5, .35)]),
                    dict(iteration=2, regularization=4., next_regularization=16., initial_force_N=.35,
                         accepted_custom_update=True, trials=[trial(1., .34, reason), trial(.5, .345)])])


def test_replay_selects_early_rejection_instead_of_terminal_iterate(failed_trace):
    spec = replay.replay_spec(failed_trace, 21)
    assert spec['iteration'] == 1 and spec['trial_index'] == 0 and spec['alpha'] == 1.
    assert spec['depths'] == [24, 28, 32, 36]
    actual = copy.deepcopy(failed_trace)
    actual['custom_newton_steps'] = actual['custom_newton_steps'][:2]
    assert replay.replay_mismatches(spec, actual) == []
    actual['custom_newton_steps'][0]['trials'][0]['native_force_N'] *= 1.01
    assert replay.replay_mismatches(spec, actual)


def test_replay_rejects_changed_backtracking_and_initializer(failed_trace):
    spec = replay.replay_spec(failed_trace, 21)
    actual = copy.deepcopy(failed_trace)
    actual['custom_newton_steps'] = actual['custom_newton_steps'][:2]
    actual['selected_predictor'] = 0
    actual['custom_newton_steps'][1]['trials'][0]['alpha'] = .5
    errors = replay.replay_mismatches(spec, actual)
    assert 'Initializer selection changed' in errors
    assert any('alpha changed' in item for item in errors)


def test_replay_rejects_a_direct_inversion_as_the_requested_unresolved_trial(failed_trace):
    trace = copy.deepcopy(failed_trace)
    for row in trace['custom_newton_steps']:
        for trial in row['trials']:
            if trial['positivity_failure']:
                trial['positivity_failure'] = 'Direct nonpositive determinant witness'
    with pytest.raises(AssertionError, match='No unresolved endpoint trial'):
        replay.replay_spec(trace, 21)


def test_independent_cartesian_derivatives_and_minimum_of_quadratic_displacement():
    vertices = np.array([[0.,0.,0.],[2.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    nodes = np.vstack((vertices, [(vertices[i]+vertices[j])/2 for i,j in combinations(range(4),2)]))
    nodal = np.column_stack((.1*nodes[:,0]**2, .2*nodes[:,1], -.1*nodes[:,2]))
    gradient = np.broadcast_to(np.eye(3), (4,3,3)).copy()
    gradient[:,0,0] = 1+.2*vertices[:,0]
    gradient[:,1,1] = 1.2
    gradient[:,2,2] = .9
    values = dict(vertices=vertices, nodal_displacement=nodal, deformation=gradient)
    result = replay.analyze_cell(values, [24,28,32,36])
    assert result['independent_gradient_agrees']
    assert result['independent_cartesian']['optimized_minimum_estimate'] == pytest.approx(1.08, abs=1e-12)
    assert result['optimized_minimum_difference'] < 1e-12
    values['deformation'] = gradient*1.01
    assert not replay.analyze_cell(values, [24,28,32,36])['independent_gradient_agrees']


def test_capture_keeps_nodal_values_from_the_rejected_field_only(failed_trace):
    class Audit:
        maps = np.arange(30).reshape(1,10,3)

        def field(self, values):
            return None

    spec = replay.replay_spec(failed_trace, 21)
    spec['element'] = 0
    trace = dict(custom_newton_steps=[failed_trace['custom_newton_steps'][0]])
    capture = replay.TrialCapture(Audit(), spec, lambda: trace)
    capture.track_field()
    nodal = np.arange(30, dtype=float)
    capture.audit.field(nodal)
    capture.on_failure('stress', 0, np.zeros((4,3,3)), np.zeros((4,3)), replay.REASON)
    assert capture.captured is None
    capture.on_failure('endpoint', 0, np.zeros((4,3,3)), np.zeros((4,3)), replay.REASON)
    nodal[:] = -1
    np.testing.assert_array_equal(capture.captured['nodal_displacement'], np.arange(30).reshape(10,3))


def test_diagnostic_uses_preserved_production_pressure_and_never_submits_a_successor(
        production_run, failed_trace, monkeypatch):
    from types import SimpleNamespace
    production = replay.production
    failed, _, target = production_run
    monkeypatch.setattr(replay, 'ROOT', production.ROOT)
    helper = production.ROOT / 'hpc/audit_reference_cell_555.py'
    helper.parent.mkdir()
    helper.write_bytes((Path(replay.__file__).parent / helper.name).read_bytes())
    event(failed, 'job_submission', return_code=0, stdout='12344.pbs')
    atomic(failed / 'state.json', dict(status='blocked', slice=1, job='12344.pbs'))
    step = failed / 'factor-4/substep-021'
    step.mkdir()
    atomic(step / 'trace-slice-001.json', failed_trace)
    (failed / 'factor-4/command-001.log').write_text('RuntimeError: Regularization ceiling without admissible descent')
    original_hash = sha(target / 'restart.npz')
    run = failed.parent / 'focused-replay'
    replay.initialize(run, failed)
    plan = production.load(run / 'plan.json')
    assert plan['positivity_replay']['iteration'] == 1
    assert sha(run / 'pressure.json') == sha(failed / 'pressure.json')
    assert len(production.production_records(run)) == 20
    nodes = run / 'nodes'
    nodes.write_text('vis01\n')
    monkeypatch.setenv('PBS_JOBID', '12345.pbs')
    monkeypatch.setenv('PBS_NODEFILE', str(nodes))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'a,b,c,d')
    commands = []

    def worker(command, **kwargs):
        commands.append(command)
        atomic(run / 'positivity-replay.json', dict(reproduced=True, reference_state_accepted=False))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production.subprocess, 'run', worker)
    production.execute(run)
    assert len(commands) == 1 and commands[0][0] != 'qsub'
    state = production.load(run / 'state.json')
    assert state['status'] == 'diagnostic_complete' and state['completed_substeps'] == 20
    assert not (run / 'factor-4/substep-021/accepted.json').exists()
    assert sha(target / 'restart.npz') == original_hash
