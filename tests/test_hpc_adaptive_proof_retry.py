from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import launch_refined_production as production
import compare_reference_timesteps_555 as refined
from replay_refined_positivity import analyze_cell
from registry import artifact, atomic, sha
from test_hpc_refined_production import production_run


@pytest.fixture
def saved_diagnostic(production_run):
    failed, _, _ = production_run
    atomic(failed / 'state.json', dict(status='running'))
    source = production.load(failed / 'plan.json')
    source['certified_depth_retry'] = dict(depth=32)
    atomic(failed / 'plan.json', source)
    artifact(production.ROOT, failed, failed / 'plan.json', 'Depth-32 source')
    diagnostic = failed.parent / 'completed-path-audit'
    folder = diagnostic / 'cell-frames'
    folder.mkdir(parents=True)
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    nodes = np.vstack((vertices, [(vertices[i]+vertices[j])/2 for i,j in combinations(range(4),2)]))
    s = np.sqrt(1e-8)
    f = np.zeros((4,3,3))
    f[:,0,0] = f[:,1,1] = vertices[:,0]-.37
    f[:,0,1] = -s
    f[:,1,0] = s+vertices[:,1]
    f[:,2,2] = 1
    nodal = np.column_stack((.5*nodes[:,0]**2-1.37*nodes[:,0]-s*nodes[:,1],
                            s*nodes[:,0]+(nodes[:,0]-1.37)*nodes[:,1], np.zeros(10)))
    frames = []
    for index, (kind, iteration, displacement, deformation) in enumerate((
            ('checkpoint', -2, np.full((10,3),20.), np.broadcast_to(np.eye(3),(4,3,3))),
            ('first_unresolved_trial', 7, nodal, f))):
        path = folder / f'frame-{index:03d}.npz'
        values = dict(vertices=vertices, nodal_displacement=displacement, deformation=deformation)
        refined.save_npz(path, **values)
        frames.append(dict(kind=kind, iteration=iteration, stage='endpoint', path=str(path.relative_to(diagnostic)),
                           sha256=sha(path), cell_audit=analyze_cell(values,[32])))
    assert frames[-1]['cell_audit']['captured_gradient']['bernstein'][-1]['outcome'] == 'unresolved'
    atomic(diagnostic / 'cell-path.json', dict(complete=True, reference_state_accepted=False,
                element=813578, substep=21, margin=1e-9, matched_prefix_directions=9, frames=frames))
    atomic(folder / 'index.json', dict(frames=[{k:v for k,v in row.items() if k != 'cell_audit'} for row in frames]))
    atomic(diagnostic / 'plan.json', dict(cell_path_audit=dict(substep=21), certified_depth_retry=dict(depth=32),
                                        diagnostic_parent=failed.name))
    atomic(diagnostic / 'state.json', dict(status='diagnostic_complete'))
    last = diagnostic / 'factor-4/substep-020/checkpoint.npz'
    last.parent.mkdir(parents=True)
    last.write_bytes((failed / 'factor-4/substep-020/checkpoint.npz').read_bytes())
    for name in ('plan.json', 'cell-path.json', 'cell-frames/index.json'):
        artifact(production.ROOT, diagnostic, diagnostic / name, 'Completed path audit')
    return diagnostic, failed


def test_saved_frame_gate_demonstrates_improved_proof_before_production(saved_diagnostic):
    diagnostic, _ = saved_diagnostic
    result = production.check_saved_cell_frames(diagnostic)
    assert result['passed'] and result['improved_matched_trials'] == 1 and result['margin'] == 1e-9
    assert result['checks'][-1]['proof']['lower_bound'] > 1e-9


def test_unimproved_evidence_prevents_submission_without_creating_run(saved_diagnostic):
    diagnostic, _ = saved_diagnostic
    report = production.load(diagnostic / 'cell-path.json')
    report['frames'] = report['frames'][:1]
    atomic(diagnostic / 'cell-path.json', report)
    artifact(production.ROOT, diagnostic, diagnostic / 'cell-path.json', 'No improved trial')
    run = diagnostic.parent / 'must-not-start'
    with pytest.raises(AssertionError, match='No previously blocked trial'):
        production.initialize_adaptive_retry(run, diagnostic)
    assert not run.exists()


def test_changed_saved_cell_is_rejected(saved_diagnostic):
    diagnostic, _ = saved_diagnostic
    (diagnostic / 'cell-frames/frame-001.npz').write_bytes(b'changed')
    with pytest.raises(AssertionError):
        production.check_saved_cell_frames(diagnostic)


def test_production_uses_interval_fallback_and_original_accepted_histories(saved_diagnostic):
    diagnostic, failed = saved_diagnostic
    run = diagnostic.parent / 'adaptive-production'
    production.initialize_adaptive_retry(run, diagnostic)
    plan = production.load(run / 'plan.json')
    assert plan['adaptive_interval_proof']['margin'] == 1e-9 and plan['certified_depth_retry']['depth'] == 32
    assert not plan.get('cell_path_audit') and not plan.get('positivity_replay')
    assert plan['requested_substeps'] == 4200 and plan['walltime_seconds'] == 168*3600
    assert sha(run / 'factor-4/substep-020/checkpoint.npz') == sha(failed / 'factor-4/substep-020/checkpoint.npz')
    assert sha(run / 'pressure.json') == sha(failed / 'pressure.json')
    assert production.load(run / 'saved-cell-proof-check.json')['passed']
    assert not (run / 'factor-4/substep-021').exists()
