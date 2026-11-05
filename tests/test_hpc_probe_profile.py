"""Diagnostic protocol checks without a cluster or local FE computation."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hpc'))
from registry import atomic, artifact, event, sha
from submit_probe_profile import parent_evidence, OUTER_SECONDS, assert_idle


def make_parent(root):
    parent = root / 'experiments/hpc/parent'
    parent.mkdir(parents=True)
    for filename in ['start.npz', 'canonical.npz']:
        (root / filename).write_bytes(b'fixture native history')
    atomic(root / 'snapshot.json', dict(accepted_steps=476))
    atomic(parent / 'plan.json', dict(snapshot='snapshot.json',
           snapshot_sha256=sha(root / 'snapshot.json'), known_start_checkpoint='start.npz',
           canonical_checkpoint='canonical.npz', cumulative_pipeline_budget_hours=192))
    atomic(parent / 'state.json', dict(status='blocked', terminal_rejection=True,
           phase='known_probe', generation_seconds=0, cumulative_seconds=34433.3,
           completed_phases=['platform', 'audit_resume_control'],
           attempts=dict(known_probe=1), blockers=['known_probe exited 1'],
           platform_output='experiments/hpc/parent/platform.json'))
    atomic(parent / 'platform.json', dict(passed=True, complete=True, audit_complete=True,
           checked_steps=476, total_steps=476))
    failed = parent / 'known_probe/attempt-001'
    failed.mkdir(parents=True)
    (failed / 'command.log').write_text('RuntimeError: Regularized Newton wall budget exhausted')
    atomic(failed / 'result.json', dict(passed=False, complete=False, records=[],
           active_trace=dict(custom_newton_steps=[dict(iteration=0)])))
    for path in [parent / 'plan.json', parent / 'platform.json',
                 root / 'start.npz', root / 'canonical.npz']:
        artifact(root, parent, path, 'Test fixture, not scientific evidence')
    return parent


def test_parent_pass_is_bound_and_budget_is_carried(tmp_path):
    parent = make_parent(tmp_path)
    _, state, hashes = parent_evidence(tmp_path, parent)
    assert state['cumulative_seconds'] == 34433.3
    assert state['cumulative_seconds'] + OUTER_SECONDS < 192 * 3600
    assert 'experiments/hpc/parent/known_probe/attempt-001/result.json' in hashes
    (tmp_path / 'start.npz').write_bytes(b'tampered')
    with pytest.raises(AssertionError, match='changed artifact'):
        parent_evidence(tmp_path, parent)


def test_partial_audit_and_exhausted_budget_cannot_launch(tmp_path):
    parent = make_parent(tmp_path)
    atomic(parent / 'platform.json', dict(passed=False, complete=False, audit_complete=False,
           checked_steps=128, total_steps=476))
    artifact(tmp_path, parent, parent / 'platform.json', 'Synthetic partial fixture')
    with pytest.raises(AssertionError):
        parent_evidence(tmp_path, parent)
    parent = make_parent(tmp_path / 'other')
    state = json.loads((parent / 'state.json').read_text())
    state['cumulative_seconds'] = 192 * 3600 - OUTER_SECONDS + 1
    atomic(parent / 'state.json', state)
    with pytest.raises(AssertionError):
        parent_evidence(tmp_path / 'other', parent)


def test_submission_refuses_other_jobs(monkeypatch, tmp_path):
    import submit_probe_profile as launcher
    monkeypatch.setenv('USER', 'fixture')
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(stdout='1489984.pbs fixture workq R\n'))
    with pytest.raises(AssertionError, match='existing PBS job'):
        assert_idle(tmp_path)


def test_one_job_submission_preserves_parent_and_rejects_duplicate(monkeypatch, tmp_path):
    import submit_probe_profile as launcher
    parent = make_parent(tmp_path)
    source = json.loads((parent / 'plan.json').read_text())
    source.update(scientific_sources={}, queue='workq', cpus=28, ranks=12,
                  memory_gb=240, software='PBSJobApp')
    atomic(parent / 'plan.json', source)
    artifact(tmp_path, parent, parent / 'plan.json', 'Fixture plan')
    for filename in ['profile_probe_native.py', 'profile_report.py', 'submit_probe_profile.py',
                     'manuscript.py', 'registry.py', 'policy.py']:
        path = tmp_path / 'hpc' / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text('# synthetic code fixture\n')
    before = {p: sha(p) for p in parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(launcher, 'P', tmp_path)
    monkeypatch.setattr(launcher, 'assert_idle', lambda root: '')
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: 'fixture-revision\n')
    commands = []
    def fake_submit(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='123.pbs\n', stderr='')
    monkeypatch.setattr(launcher.subprocess, 'run', fake_submit)
    args = SimpleNamespace(parent_run='parent', run_id='profile-01')
    launcher.submit(args)
    assert len(commands) == 1 and commands[0][0] == 'qsub'
    assert 'select=1:ncpus=28:mem=240gb' in commands[0]
    assert 'walltime=02:00:00' in commands[0] and 'place=excl' in commands[0]
    plan = json.loads((tmp_path / 'experiments/hpc/profile-01/plan.json').read_text())
    assert plan['cumulative_seconds_before'] == 34433.3 and plan['per_equation_seconds'] == 1800
    assert not plan['reference_continuation_allowed']
    assert before == {p: sha(p) for p in before}
    with pytest.raises(FileExistsError):
        launcher.submit(args)
    assert len(commands) == 1


@pytest.mark.parametrize('timeout', [True, False])
def test_observer_preserves_terminal_native_arrays_and_never_advances_history(monkeypatch, tmp_path, timeout):
    import profile_probe_native as worker
    monkeypatch.setattr(worker, 'P', tmp_path)
    run = tmp_path / 'diagnostic'
    run.mkdir()
    fields = tmp_path / 'data/processed/source'
    fields.mkdir(parents=True)
    archived = np.zeros((2, 3))
    np.save(fields / 'displacement_coefficients.npy', archived)
    atomic(tmp_path / 'snapshot.json', dict(reference='source', files={}, accepted_steps=2,
           prefix_fields_sha256=hashlib.sha256(archived.tobytes()).hexdigest()))
    np.savez(tmp_path / 'start.npz', completed_steps=1)
    atomic(tmp_path / 'plan.json', dict(snapshot='snapshot.json', known_start_checkpoint='start.npz'))
    monkeypatch.setattr(sys, 'argv', ['profile', '--spec', 'plan.json', '--directory', 'diagnostic'])
    comm = SimpleNamespace(rank=0, size=1, bcast=lambda value, root: value,
                           Ibarrier=lambda: SimpleNamespace(Test=lambda: True))
    def abort(code):
        raise RuntimeError(f'expected diagnostic abort {code}')
    comm.Abort = abort
    array = np.array([1., 2., 3.])
    function = SimpleNamespace(vector=lambda: SimpleNamespace(get_local=lambda: array.copy()))
    problem = SimpleNamespace(**{k: function for k in ['u', 'u_old', 'v_old', 'a_old']})
    problem.solver = SimpleNamespace(solve=lambda: None, _snes=SimpleNamespace(
        getConvergedReason=lambda: 2, getIterationNumber=lambda: 0))
    adapter = SimpleNamespace(mapping=np.arange(3), gather=lambda name: archived[1])
    advanced = []
    def original(*args):
        if timeout:
            raise RuntimeError('Regularized Newton wall budget exhausted')
    fake = SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=comm), solve=original,
                           residual=lambda p: SimpleNamespace(norm=lambda _: 5e-12))
    def main():
        fake.solve(problem, adapter, {}, None, dict(wall_budget_seconds=1800), {}, lambda: None)
        advanced.append('would advance original histories')
    fake.main = main
    monkeypatch.setitem(sys.modules, 'probe_native', fake)
    if timeout:
        with pytest.raises(RuntimeError, match='expected diagnostic abort'):
            worker.main()
    else:
        worker.main()
    assert not advanced
    report = json.loads((run / 'rank-000.json').read_text())
    assert report['first_known_equation_converged'] == (not timeout)
    assert not report['known_pair_qualified'] and not report['independent_audit_passed']
    assert not report['histories_advanced'] and not report['physical_new_state_generated']
    assert report['functions'] and report['exclusive_profiled_seconds'] >= 0
    assert report['terminal_owned_fields_sha256'] == sha(run / 'rank-000.npz')
    with np.load(run / 'rank-000.npz') as saved:
        for key in ['u', 'u_old', 'v_old', 'a_old']:
            np.testing.assert_array_equal(saved[key], array)
