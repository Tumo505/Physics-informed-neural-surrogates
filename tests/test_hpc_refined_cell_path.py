import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import audit_refined_cell_path as cell_path
from lv_surrogate.physics.bernstein import BernsteinAudit
from registry import artifact, atomic, event, sha
from test_hpc_refined_production import production_run


@pytest.fixture
def trace():
    rows = []
    reason = f'{cell_path.REASON} (stage=endpoint, element={cell_path.ELEMENT})'
    for index in range(9):
        rows.append(dict(iteration=index, regularization=1., next_regularization=1.,
                         initial_force_N=.3-.001*index, accepted_custom_update=True,
                         trials=[dict(alpha=1., native_force_N=.299-.001*index,
                                      armijo_passed=True, whole_element_admissible=True, positivity_failure=None)]))
    rows[-1]['trials'].insert(0, dict(alpha=1., native_force_N=.29, armijo_passed=True,
                                     whole_element_admissible=False, positivity_failure=reason))
    rows[-1]['trials'][-1]['alpha'] = .5
    rows.append(dict(iteration=9, regularization=1e8, next_regularization=1e8, initial_force_N=.29,
                     accepted_custom_update=False, trials=[dict(alpha=1., native_force_N=.29,
                         armijo_passed=False, whole_element_admissible=False, positivity_failure=None)]))
    return dict(substep=21, selected_predictor=1, initial_forces_N=[1., .3, .4, .5], custom_newton_steps=rows)


def test_cell_path_uses_completed_directions_before_terminal_ceiling(trace):
    spec = cell_path.path_spec(trace, 1e8)
    assert spec['directions'] == 9 and spec['maximum_certification_depth'] == 32
    actual = dict(trace, custom_newton_steps=trace['custom_newton_steps'][:9])
    assert cell_path.matching_prefix(spec, actual) == 9
    actual = copy.deepcopy(actual)
    actual['custom_newton_steps'][7]['trials'][0]['alpha'] = .5
    assert cell_path.matching_prefix(spec, actual) == 7


def test_path_preserves_available_directions_after_cancellation(trace):
    trace['custom_newton_steps'] = trace['custom_newton_steps'][:8]
    assert cell_path.path_spec(trace, 1e8)['directions'] == 8


def test_cell_frames_preserve_checkpoint_initial_and_newton_fields(tmp_path, trace):
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    audit = BernsteinAudit(dict(coords=vertices, cells=np.array([[0,1,2,3]]),
                                canonical_cell_dofs=np.arange(30).reshape(1,10,3)))
    spec = cell_path.path_spec(trace, 1e8)
    spec['element'] = 0
    current_trace = dict(custom_newton_steps=[])
    path = cell_path.CellPath(tmp_path, audit, spec, lambda: current_trace)
    old = np.zeros(30)
    path.start(old, old, 2e-10)
    candidate = np.arange(30, dtype=float)*1e-5
    path.initializer(candidate, .3)
    path.accepted(0, candidate, .2, 1.)
    path.track_field()
    audit.field(candidate)
    deformation = np.eye(3)+np.einsum('ki,vkj->vij', candidate.reshape(10,3), path.gradients)
    path.on_failure('endpoint', 0, deformation, vertices, cell_path.REASON)
    path.on_failure('endpoint', 0, deformation, vertices, cell_path.REASON)
    path.on_failure('stress', 0, deformation, vertices, cell_path.REASON)
    candidate[:] = 10
    assert len(path.frames) == 7
    with np.load(tmp_path / path.frames[-1]['path']) as saved:
        np.testing.assert_array_equal(saved['nodal_displacement'], np.arange(30).reshape(10,3)*1e-5)
    with np.load(tmp_path / path.frames[5]['path']) as saved:
        np.testing.assert_array_equal(saved['nodal_displacement'], .6*(np.arange(30).reshape(10,3)*1e-5))
    assert path.frames[0]['kind'] == 'checkpoint' and path.frames[2]['kind'] == 'initializer'


def test_independent_path_report_distinguishes_partial_replay_from_source(tmp_path, trace):
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    audit = BernsteinAudit(dict(coords=vertices, cells=np.array([[0,1,2,3]]),
                                canonical_cell_dofs=np.arange(30).reshape(1,10,3)))
    spec = cell_path.path_spec(trace, 1e8)
    spec['element'] = 0
    actual = dict(trace, custom_newton_steps=trace['custom_newton_steps'][:1])
    path = cell_path.CellPath(tmp_path, audit, spec, lambda: actual)
    zero = np.zeros(30)
    path.start(zero, zero, 2e-10)
    path.initializer(zero, .3)
    path.accepted(0, zero, .299, 1.)
    path.finish(actual, 'Regularization ceiling without admissible descent')
    result = cell_path.production.load(tmp_path / 'cell-path.json')
    assert result['complete'] and not result['reference_state_accepted'] and not result['reproduced_exactly']
    assert result['matched_prefix_directions'] == 1 and result['replayed_directions'] == 1
    assert len(result['frames']) == 6
    for frame in result['frames']:
        independent = frame['cell_audit']['independent_cartesian']
        assert independent['optimized_minimum_estimate'] == pytest.approx(1.)
        assert independent['singular_values_at_minimum'][-1] == pytest.approx(1.)
        assert frame['cell_audit']['independent_gradient_agrees']


def test_cancelled_production_becomes_separate_diagnostic_without_advancing_state(production_run, trace, monkeypatch):
    from types import SimpleNamespace
    production = cell_path.production
    failed, _, target = production_run
    monkeypatch.setattr(cell_path, 'ROOT', production.ROOT)
    plan = production.load(failed / 'plan.json')
    plan['certified_depth_retry'] = dict(depth=32)
    atomic(Path(plan['protocol']), dict(admissible_newton=dict(maximum_regularization=1e8)))
    atomic(failed / 'plan.json', plan)
    artifact(production.ROOT, failed, failed / 'plan.json', 'Depth-32 source fixture')
    atomic(failed / 'state.json', dict(status='running', slice=1, job='12344.pbs'))
    event(failed, 'job_submission', return_code=0, stdout='12344.pbs')
    atomic(failed / 'factor-4/substep-021/trace-slice-001.json', trace)
    source_hash = sha(target / 'restart.npz')
    run = failed.parent / 'cell-path'
    cell_path.initialize(run, failed)
    diagnostic = production.load(run / 'plan.json')
    assert diagnostic['cell_path_audit']['directions'] == 9
    assert diagnostic['certified_depth_retry']['depth'] == 32
    assert sha(run / 'pressure.json') == sha(failed / 'pressure.json')
    nodes = run / 'nodes'
    nodes.write_text('vis01\n')
    monkeypatch.setenv('PBS_JOBID', '12345.pbs')
    monkeypatch.setenv('PBS_NODEFILE', str(nodes))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'a,b,c,d')
    commands = []

    def worker(command, **kwargs):
        commands.append(command)
        atomic(run / 'cell-path.json', dict(complete=True, reference_state_accepted=False, reproduced_exactly=False))
        atomic(run / 'cell-frames/index.json', dict(frames=[]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(production.subprocess, 'run', worker)
    production.execute(run)
    state = production.load(run / 'state.json')
    assert state['status'] == 'diagnostic_complete' and state['completed_substeps'] == 20
    assert len(commands) == 1 and commands[0][0] != 'qsub'
    assert not (run / 'factor-4/substep-021/accepted.json').exists()
    assert sha(target / 'restart.npz') == source_hash
