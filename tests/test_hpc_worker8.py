"""Evidence-driven follow-up orchestration; no FE solves or model training."""
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'hpc'), str(ROOT / 'src')]
from registry import atomic, artifact, read, sha
import launch_worker8 as launcher
from worker8_gates import known_gate
from worker8_native import check_policy, install


def benchmark_fixture(root):
    original = root / 'experiments/hpc/original'
    profile = root / 'experiments/hpc/profile'
    benchmark = root / 'experiments/hpc/benchmark'
    for path in [original, profile, benchmark, root / 'hpc']:
        path.mkdir(parents=True, exist_ok=True)
    (root / 'hpc/run.py').write_text('# frozen original fixture\n')
    for name in ['launch_worker8.py', 'run_worker8.py', 'worker8_gates.py', 'worker8_native.py']:
        (root / 'hpc' / name).write_text('# new wrapper fixture\n')
    atomic(root / 'config.json', dict(name='old', dt_s=.000625))
    atomic(root / 'snapshot.json', dict(accepted_steps=476, source_config='config.json'))
    for name in ['canonical.npz', 'start.npz']:
        (original / name).write_bytes(b'exact native fixture')
    atomic(original / 'plan.json', dict(snapshot='snapshot.json', snapshot_sha256=sha(root / 'snapshot.json'),
           scientific_sources={'hpc/run.py': sha(root / 'hpc/run.py')},
           known_start_checkpoint='experiments/hpc/original/start.npz',
           canonical_checkpoint='experiments/hpc/original/canonical.npz',
           ranks=12, cpus=28, memory_gb=240, queue='workq', software='PBSJobApp',
           cumulative_pipeline_budget_hours=192, native_generation_budget_hours=168,
           per_step_seconds=1800, segment_steps=4, environment_sha256={}, native_prefix='.hpc/native-v2'))
    atomic(original / 'audit.json', dict(complete=True, passed=True, audit_complete=True,
           checked_steps=476, total_steps=476,
           records=[dict(native_residual_N=5e-12) for _ in range(476)]))
    atomic(original / 'state.json', dict(terminal_rejection=True, phase='known_probe',
           completed_phases=['audit_resume_control', 'platform'], generation_seconds=0,
           platform_output='experiments/hpc/original/audit.json'))
    artifact(root, original, original / 'plan.json', 'Synthetic fixture')
    for name in ['audit.json', 'canonical.npz', 'start.npz']:
        artifact(root, original, original / name, 'Synthetic fixture')
    atomic(profile / 'plan.json', dict(parent_run='original'))
    binding_paths = [original / 'plan.json', original / 'state.json', original / 'audit.json', profile / 'plan.json',
                     original / 'canonical.npz', original / 'start.npz', root / 'snapshot.json']
    atomic(benchmark / 'plan.json', dict(parent_run='profile', parent_files_sha256={
        str(p.relative_to(root)): sha(p) for p in binding_paths}, scientific_sources={
            'hpc/run.py': sha(root / 'hpc/run.py')}, diagnostic_sources={}, cumulative_seconds_before=37078.3))
    order = [4, 1, 2, 8, 16, 16, 8, 2, 1, 4]
    times = {1:428., 2:340., 4:254., 8:212., 16:211.}
    records = []
    for index, workers in enumerate(order):
        atomic(benchmark / f'proof-{index:02d}.json', dict(scope='Synthetic proof fixture'))
        records.append(dict(workers=workers, pair_seconds=times[workers], semantic_proof_agreement=True,
                            profiled=False, proof_sha256=sha(benchmark / f'proof-{index:02d}.json')))
    atomic(benchmark / 'result.json', dict(complete=True, passed=True, worker_order=order, records=records))
    atomic(benchmark / 'summary.json', dict(complete=True, proof_benchmark_passed=True,
           stop_reason=None, cumulative_seconds=40638.6))
    atomic(benchmark / 'state.json', dict(status='diagnostic_complete'))
    for name in ['plan.json', 'result.json', 'summary.json']:
        artifact(root, benchmark, benchmark / name, 'Synthetic benchmark evidence')
    return benchmark


def test_qualified_evidence_reused_and_time_carried_without_editing_parent(tmp_path, monkeypatch):
    parent = benchmark_fixture(tmp_path)
    before = {p: sha(p) for p in (tmp_path / 'experiments').rglob('*') if p.is_file()}
    monkeypatch.setattr(launcher, 'assert_idle', lambda root: '')
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: 'fixture-revision\n')
    run, plan = launcher.prepare(tmp_path, 'worker8', 'benchmark')
    state = json.loads((run / 'state.json').read_text())
    assert state['completed_phases'] == ['audit_resume_control', 'platform']
    assert state['cumulative_seconds'] == 40638.6 and state['generation_seconds'] == 0
    assert not state['attempts'] and state['phase'] == 'known_probe'
    assert plan['worker_policy']['workers'] == 8 and plan['ranks'] == 12
    assert plan['per_step_seconds'] == 1800 and plan['native_generation_budget_hours'] == 168
    assert plan['cumulative_pipeline_budget_hours'] == 192
    assert plan['reference_config_sha256'] == sha(run / 'reference-config.json')
    assert all(sha(p) == digest for p, digest in before.items())
    with pytest.raises(FileExistsError):
        launcher.prepare(tmp_path, 'worker8', 'benchmark')


def test_partial_or_tampered_or_unjustified_benchmark_cannot_release_solver(tmp_path):
    parent = benchmark_fixture(tmp_path)
    result = json.loads((parent / 'result.json').read_text())
    result['records'] = result['records'][:-1]
    atomic(parent / 'result.json', result)
    with pytest.raises(AssertionError, match='changed artifact'):
        launcher.parent_evidence(tmp_path, parent)
    artifact(tmp_path, parent, parent / 'result.json', 'Synthetic incomplete counterexample')
    with pytest.raises(AssertionError):
        launcher.parent_evidence(tmp_path, parent)


def test_cofactor_launch_requires_registered_comparison_and_carries_spent_time(tmp_path, monkeypatch):
    parent = benchmark_fixture(tmp_path)
    cofactor = parent.parent / 'cofactor'
    cofactor.mkdir()
    atomic(cofactor / 'plan.json', dict(snapshot_sha256=sha(tmp_path / 'snapshot.json'),
           native_prefix='.hpc/native-v2', environment_sha256={}, workers=8, fixed_steps=[474, 475],
           cumulative_seconds_before=43227.,
           frozen={str((parent / 'summary.json').relative_to(tmp_path)):sha(parent / 'summary.json')}))
    atomic(cofactor / 'summary.json', dict(complete=True, proof_benchmark_passed=True, stop_reason=None,
           reference_continuation_allowed=False, physical_new_state_generated=False, cumulative_seconds=43755.,
           records=[dict(step=i, proof_agreement=True, pair_seconds=dict(legacy=212., cofactor=3.2))
                    for i in [474, 475]]))
    for step in [474, 475]:
        for variant in ['legacy', 'cofactor']:
            path = cofactor / f'proof-{step}-{variant}.json'
            atomic(path, dict(scope='Synthetic proof fixture'))
            artifact(tmp_path, cofactor, path, 'Synthetic proof fixture')
    for name in ['plan.json', 'summary.json']:
        artifact(tmp_path, cofactor, cofactor / name, 'Synthetic comparison fixture')
    monkeypatch.setattr(launcher, 'assert_idle', lambda root: '')
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: 'fixture-revision\n')
    run, plan = launcher.prepare(tmp_path, 'candidate', 'benchmark', 'cofactor')
    assert plan['worker_policy']['implementation'] == 'cofactor'
    assert plan['worker_policy']['benchmark_summary_sha256'] == sha(cofactor / 'summary.json')
    state = json.loads((run / 'state.json').read_text())
    assert state['cumulative_seconds'] == 43755. and state['phase'] == 'known_probe'
    atomic(cofactor / 'proof-475-cofactor.json', dict(changed=True))
    with pytest.raises(AssertionError, match='changed artifact'):
        launcher.prepare(tmp_path, 'reject', 'benchmark', 'cofactor')


def test_dependency_chain_is_one_submission_and_never_submits_setup(tmp_path, monkeypatch):
    benchmark_fixture(tmp_path)
    monkeypatch.setattr(launcher, 'assert_idle', lambda root: '')
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: 'fixture-revision\n')
    calls = []
    def qsub(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=f'{100+len(calls)}.pbs\n', stderr='')
    monkeypatch.setattr(launcher.subprocess, 'run', qsub)
    run = launcher.launch(tmp_path, 'worker8', 'benchmark', jobs=3)
    assert len(calls) == 3
    assert all('select=1:ncpus=28:mem=240gb' in c and 'walltime=24:00:00' in c for c in calls)
    assert not any('setup.pbs' in word for c in calls for word in c)
    assert 'depend=afterok:101.pbs' in calls[1] and 'depend=afterok:102.pbs' in calls[2]
    assert 'hpc/run_worker8.py' in (run / 'job-001.pbs').read_text()


def known_fixture(root):
    fields, checkpoint = root / 'fields.npy', root / 'checkpoint.npz'
    fields.write_bytes(b'known fields fixture')
    checkpoint.write_bytes(b'actual histories fixture')
    proof = {k:dict(passed=True, minimum_Bernstein_bound=.001, evaluation_workers=8)
             for k in ['endpoint', 'stress']}
    probe = dict(complete=True, passed=True, start_step=474, fields_sha256=sha(fields),
                 checkpoint_sha256=sha(checkpoint), records=[dict(step=i, SNES_reason=2, native_updates=0,
                 native_force_N=5e-12, known_field_max_difference_m=1e-17,
                 custom_trace=dict(custom_force_N=5e-12), whole_element_proof=proof) for i in [474,475]])
    atomic(root / 'probe.json', probe)
    independent = dict(complete=True, passed=True, fields_sha256=sha(fields), checkpoint_sha256=sha(checkpoint),
                       probe_sha256=sha(root / 'probe.json'), history_relative_errors={k:0. for k in ['u','u_old','v_old','a_old']},
                       records=[dict(step=i, native_residual_N=6e-12, independent_norm_relative_error=1e-14,
                                     whole_element_proof=proof) for i in [474,475]])
    atomic(root / 'independent.json', independent)
    return probe, independent


@pytest.mark.parametrize('bad', ['force', 'native_updates', 'field_difference', 'history', 'checkpoint', 'workers'])
def test_known_gate_requires_actual_strict_native_evidence(tmp_path, bad):
    probe, independent = known_fixture(tmp_path)
    def check():
        return known_gate(tmp_path, 'probe.json', 'independent.json', 'fields.npy', 'checkpoint.npz')
    assert check()['known_pair_passed']
    if bad == 'force':probe['records'][0]['native_force_N'] = .006
    if bad == 'native_updates':probe['records'][0]['native_updates'] = 1
    if bad == 'field_difference':probe['records'][1]['known_field_max_difference_m'] = 1e-8
    if bad == 'workers':probe['records'][0]['whole_element_proof']['endpoint']['evaluation_workers'] = 4
    if bad == 'history':independent['history_relative_errors'].pop('a_old')
    if bad == 'checkpoint':(tmp_path / 'checkpoint.npz').write_bytes(b'tampered checkpoint')
    atomic(tmp_path / 'probe.json', probe)
    independent['probe_sha256'] = sha(tmp_path / 'probe.json')
    atomic(tmp_path / 'independent.json', independent)
    with pytest.raises(AssertionError):check()


def test_generation_policy_does_not_accept_benchmark_as_solver_qualification(tmp_path):
    atomic(tmp_path / 'benchmark.json', dict(complete=True, proof_benchmark_passed=True))
    spec = dict(worker_policy=dict(workers=8, per_step_seconds=1800, benchmark_summary='benchmark.json',
                benchmark_summary_sha256=sha(tmp_path / 'benchmark.json')))
    check_policy(tmp_path, spec)
    with pytest.raises(KeyError):check_policy(tmp_path, spec, generation=True)
    spec['worker_policy']['per_step_seconds'] = 3600
    with pytest.raises(AssertionError):check_policy(tmp_path, spec)


def test_cofactor_generation_requires_native_and_independent_proof_evidence(tmp_path):
    probe, independent = known_fixture(tmp_path)
    benchmark = dict(complete=True, proof_benchmark_passed=True, stop_reason=None,
                     reference_continuation_allowed=False, physical_new_state_generated=False,
                     records=[dict(step=i, proof_agreement=True, pair_seconds=dict(legacy=212., cofactor=3.2))
                              for i in [474, 475]])
    atomic(tmp_path / 'comparison.json', benchmark)
    policy = dict(workers=8, per_step_seconds=1800, implementation='cofactor', benchmark_steps=[474, 475],
                  benchmark_summary='comparison.json', benchmark_summary_sha256=sha(tmp_path / 'comparison.json'),
                  known_probe='probe.json', known_verification='independent.json',
                  known_fields='fields.npy', known_checkpoint='checkpoint.npz',
                  known_verification_sha256=sha(tmp_path / 'independent.json'))
    with pytest.raises(AssertionError, match='Missing cofactor probe proof'):
        check_policy(tmp_path, dict(worker_policy=policy), generation=True)
    for row in probe['records']:
        row['whole_element_proof']['proof_implementation'] = 'cofactor'
    atomic(tmp_path / 'probe.json', probe)
    independent['probe_sha256'] = sha(tmp_path / 'probe.json')
    atomic(tmp_path / 'independent.json', independent)
    policy['known_verification_sha256'] = sha(tmp_path / 'independent.json')
    check_policy(tmp_path, dict(worker_policy=policy), generation=True)


@pytest.mark.parametrize('implementation', ['legacy', 'cofactor'])
def test_fixed_worker_factory_preserves_original_geometry_and_criteria(implementation):
    from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit
    from test_hpc_proof_benchmark import curved_fixture
    from proof_benchmark import same_proof
    arrays, u = curved_fixture()
    module = SimpleNamespace(ThreadedBernsteinAudit=ThreadedBernsteinAudit)
    install(module, implementation)
    original = ThreadedBernsteinAudit(arrays)
    followup = module.ThreadedBernsteinAudit(arrays)
    try:
        assert len(original.shards) == 4 and len(followup.shards) == 8
        assert original.margin == followup.margin == 1e-9
        assert original.chunk == followup.chunk == 8192
        assert original.max_depth == followup.max_depth == 12
        actual = followup.pair(u,u)
        same_proof(original.pair(u,u), actual)
        if implementation == 'cofactor':
            assert actual['proof_implementation'] == 'cofactor'
        with pytest.raises(AssertionError):module.ThreadedBernsteinAudit(arrays, workers=16)
    finally:
        original.close();followup.close()


def test_actual_followup_driver_preserves_failed_probe_and_never_completes_it(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv('PBS_JOBID', 'fixture.pbs')
    tree = ast.parse((ROOT / 'hpc/run_worker8.py').read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in ['execute', 'next_output', 'Paused', 'Rejected']]
    run = tmp_path / 'experiments/hpc/followup'
    run.mkdir(parents=True)
    state = dict(attempts={}, completed_phases=['platform'], cumulative_seconds=40638.6)
    saved = []
    class FailedNative:
        def __init__(self, command, **kwargs):
            directory = Path(command[0]).parent
            (directory / 'terminal-rank-000.json').write_text('{"failed":true}')
            (directory / 'terminal-rank-000.npz').write_bytes(b'exact native fixture')
        def poll(self):return 1
        def wait(self):return 1
    ns = dict(P=tmp_path, run=run, state=state,
              plan=dict(cumulative_pipeline_budget_hours=192), interrupted=False,
              deadline=time.monotonic()+10000, time=time, env={}, os=__import__('os'),
              subprocess=SimpleNamespace(Popen=FailedNative, STDOUT=-2),
              event=lambda *a, **k: None, render=lambda *a, **k: None, persist=lambda: None,
              artifact=lambda root, run, path, role: saved.append(path), json=json)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'actual follow-up functions', 'exec'), ns)
    output = ns['next_output']('known_probe', 'result.json')
    with pytest.raises(ns['Rejected'], match='known_probe exited 1'):
        ns['execute']('known_probe', [str(tmp_path/output)], 5400)
    assert state['completed_phases'] == ['platform']
    assert state['attempts']['known_probe'] == 1
    assert {p.name for p in saved} == {'command.log', 'terminal-rank-000.json', 'terminal-rank-000.npz'}
    assert (tmp_path/output).with_name('terminal-rank-000.npz').read_bytes() == b'exact native fixture'


def test_actual_seed_branch_rejects_success_flag_with_bad_native_force(tmp_path):
    probe, independent = known_fixture(tmp_path)
    probe['records'][0]['native_force_N'] = .006
    atomic(tmp_path / 'probe.json', probe)
    independent['probe_sha256'] = sha(tmp_path / 'probe.json')
    atomic(tmp_path / 'independent.json', independent)
    tree = ast.parse((ROOT / 'hpc/run_worker8.py').read_text())
    branch = next(node for top in tree.body if isinstance(top, ast.Try)
                  for node in top.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "not completed('seed')")
    ns = dict(P=tmp_path, completed=lambda phase: False, known_gate=known_gate, plan={'worker_policy': {}},
              state=dict(probe_output='probe.json', probe_verification='independent.json',
                         probe_fields='fields.npy', probe_checkpoint='checkpoint.npz'))
    with pytest.raises(AssertionError):
        exec(compile(ast.Module(body=[branch], type_ignores=[]), 'actual seed branch', 'exec'), ns)
    assert not (tmp_path / 'stream-protocol.json').exists()
