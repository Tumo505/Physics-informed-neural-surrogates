import copy
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import launch_refined_production as production
from refined_positivity_policy import extend_unresolved_depth
from lv_surrogate.physics.bernstein import BernsteinAudit
from registry import artifact, atomic, sha
from test_hpc_refined_production import production_run


@pytest.fixture
def evidence():
    spec = dict(substep=21, element=813578, stage='endpoint', iteration=6, alpha=1.,
                original_depth=20, retry_depth=24)
    report = dict(spec, reproduced=True, mismatches=[], reference_state_accepted=False, margin=1e-9,
                  cell_audit=dict(independent_gradient_agrees=True,
                      independent_cartesian=dict(optimized_minimum_estimate=3.201362035041934e-7),
                      captured_gradient=dict(bernstein=[
                          dict(depth=24, outcome='unresolved'), dict(depth=28, outcome='unresolved'),
                          dict(depth=32, outcome='certified_positive', lower_bound=2.2905681658968454e-7,
                               maximum_depth=30, leaf_tetrahedra=47)])))
    return report, spec


def test_observed_replay_authorizes_depth_32(evidence):
    report, spec = evidence
    policy = production.check_depth_evidence(report, spec)
    assert policy['depth'] == 32 and policy['margin'] == 1e-9 and policy['substep'] == 21


@pytest.mark.parametrize('change', ['unreproduced', 'gradient_disagrees', 'weak_bound', 'unresolved_32', 'different_margin'])
def test_unqualified_evidence_cannot_authorize_retry(evidence, change):
    report, spec = copy.deepcopy(evidence)
    if change == 'unreproduced':
        report['reproduced'] = False
    elif change == 'gradient_disagrees':
        report['cell_audit']['independent_gradient_agrees'] = False
    elif change == 'weak_bound':
        report['cell_audit']['captured_gradient']['bernstein'][-1]['lower_bound'] = 1e-10
    elif change == 'unresolved_32':
        report['cell_audit']['captured_gradient']['bernstein'][-1]['outcome'] = 'unresolved'
    else:
        report['margin'] = 1e-12
    with pytest.raises(AssertionError):
        production.check_depth_evidence(report, spec)


def make_cell():
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    cell = BernsteinAudit(dict(coords=vertices, cells=np.array([[0,1,2,3]]),
                               canonical_cell_dofs=np.arange(30).reshape(1,10,3)))
    cell.max_depth = 20
    return cell, vertices


def test_only_unresolved_proofs_reach_the_deeper_arithmetic():
    child, vertices = make_cell()

    def unresolved(*args):
        raise ValueError('Whole-element positivity remains unresolved at maximum depth')

    child.resolve = unresolved
    restore = extend_unresolved_depth(SimpleNamespace(margin=1e-9, shards=[(0, child)]), 32)
    proof = child.resolve(np.broadcast_to(np.eye(3), (4,3,3)), vertices)
    assert proof['lower_bound'] == pytest.approx(1.) and proof['subdivision_retry_depth'] == 32
    assert child.max_depth == 20
    restore()
    assert child.resolve is unresolved


@pytest.mark.parametrize('reason', ['Direct nonpositive determinant witness', 'Nonfinite determinant'])
def test_direct_witnesses_and_other_errors_are_not_retried(reason):
    child, vertices = make_cell()

    def reject(*args):
        raise ValueError(reason)

    child.resolve = reject
    extend_unresolved_depth(SimpleNamespace(margin=1e-9, shards=[(0, child)]), 32)
    # This deformation would pass if the direct rejection were swallowed.
    with pytest.raises(ValueError, match=reason):
        child.resolve(np.broadcast_to(np.eye(3), (4,3,3)), vertices)


def test_successful_base_proof_is_retained_without_extra_subdivision():
    child, vertices = make_cell()
    expected = dict(lower_bound=.5, leaf_tetrahedra=1, maximum_depth=0)
    child.resolve = lambda *args: expected
    extend_unresolved_depth(SimpleNamespace(margin=1e-9, shards=[(0, child)]), 32)
    assert child.resolve(None, vertices) is expected


def test_depth_32_can_still_reject_a_nonpositive_cell():
    child, vertices = make_cell()

    def unresolved(*args):
        raise ValueError('Whole-element positivity remains unresolved at maximum depth')

    child.resolve = unresolved
    extend_unresolved_depth(SimpleNamespace(margin=1e-9, shards=[(0, child)]), 32)
    with pytest.raises(ValueError, match='Direct nonpositive determinant witness'):
        child.resolve(np.broadcast_to(np.diag([-1.,1.,1.]), (4,3,3)), vertices)


def test_retry_starts_from_accepted_checkpoint_and_preserves_pressure(production_run, evidence):
    failed, _, target = production_run
    atomic(failed / 'state.json', dict(status='blocked'))
    trace = failed / 'factor-4/substep-021/trace-slice-001.json'
    atomic(trace, dict(unaccepted=True))
    replay = failed.parent / 'certified-replay'
    replay.mkdir()
    report, spec = evidence
    atomic(replay / 'plan.json', dict(positivity_replay=spec, diagnostic_parent=failed.name,
                                     frozen={str(failed / 'plan.json'): sha(failed / 'plan.json'), str(trace): sha(trace)}))
    atomic(replay / 'positivity-replay.json', report)
    atomic(replay / 'state.json', dict(status='diagnostic_complete'))
    (replay / 'rejected-cell.npz').write_bytes(b'registered captured cell')
    last = replay / 'factor-4/substep-020/checkpoint.npz'
    last.parent.mkdir(parents=True)
    shutil.copyfile(failed / 'factor-4/substep-020/checkpoint.npz', last)
    for name in ('plan.json', 'positivity-replay.json', 'rejected-cell.npz'):
        artifact(production.ROOT, replay, replay / name, 'Completed replay evidence')
    retry = failed.parent / 'depth-32-production'
    production.initialize_certified_retry(retry, replay)
    plan = production.load(retry / 'plan.json')
    assert not plan.get('positivity_replay')
    assert plan['certified_depth_retry']['depth'] == 32 and plan['requested_substeps'] == 4200
    assert plan['resume']['completed_substeps'] == 20 and plan['walltime_seconds'] == 168*3600
    assert sha(retry / 'factor-4/substep-020/checkpoint.npz') == sha(last)
    assert sha(retry / 'pressure.json') == sha(failed / 'pressure.json')
    assert not (retry / 'factor-4/substep-021').exists()
    assert production.load(target / 'manifest.json')['completed_steps'] == 554
