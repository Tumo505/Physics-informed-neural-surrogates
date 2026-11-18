import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import launch_refined_production as production
import compare_reference_timesteps_555 as refined
from registry import artifact, atomic, sha


@pytest.fixture
def production_run(tmp_path, monkeypatch):
    root = tmp_path
    target = root / 'data/processed/original'
    target.mkdir(parents=True)
    gpu = root / 'gpu'
    (gpu / 'lib').mkdir(parents=True)
    (gpu / 'lib/libpetsc.so').write_bytes(b'CUDA library fingerprint')
    monkeypatch.setattr(production, 'ROOT', root)
    monkeypatch.setattr(production, 'TARGET', target)
    monkeypatch.setattr(production, 'GPU', gpu)
    config = dict(duration_s=1., dt_s=1/1600, amplitude=1., timing_scale=1.)
    atomic(target / 'manifest.json', dict(config=config, completed_steps=554))
    (target / 'restart.npz').write_bytes(b'unchanged original checkpoint')
    times = np.arange(1, 1601)*config['dt_s']
    np.savez(target / 'inputs.npz', time=times, pressure=10000+(times-.4*config['dt_s'])**2)
    parent = root / 'experiments/hpc/refined-complete'
    parent.mkdir(parents=True)
    dt, times, stress_times = production.production_schedule(config)
    pressure = 10000+stress_times**2
    old = dict(factors={'4': dict(dt_s=dt, endpoint_times=times[:20].tolist(),
                                pressure_times=stress_times[:20].tolist(), pressures=pressure[:20].tolist())},
               material_parameters={})
    atomic(parent / 'pressure.json', old)
    refined.save_npz(parent / 'seed-550.npz', u=np.zeros(3))
    refined.save_npz(parent / 'patch.npz', vertices=np.zeros((1,4,3)))
    rows = []
    digest = sha(parent / 'seed-550.npz')
    for index in range(1, 21):
        step = parent / f'factor-4/substep-{index:03d}'
        step.mkdir(parents=True)
        refined.save_npz(step / 'checkpoint.npz', u=np.full(3, index), u_old=np.full(3, index),
                         v_old=np.full(3, 2.), a_old=np.full(3, 3.), previous=np.full(3, index-1),
                         completed_substeps=index, factor=4, time_s=times[index-1])
        atomic(step / 'trace.json', dict(custom_force_N=2e-10))
        row = dict(substep=index, time_s=times[index-1], pressure_time_s=stress_times[index-1],
                   pressure_Pa=pressure[index-1], native_force_N=2e-10, custom_force_N=2e-10,
                   native_updates=0, checkpoint_sha256=sha(step / 'checkpoint.npz'),
                   trace_sha256=sha(step / 'trace.json'), input_checkpoint_sha256=digest,
                   patch={}, whole_element_proof={'passed': True})
        atomic(step / 'accepted.json', row)
        rows.append(row)
        digest = row['checkpoint_sha256']
    frozen = {str(target / 'manifest.json'): sha(target / 'manifest.json')}
    atomic(parent / 'plan.json', dict(frozen=frozen, protocol=str(root / 'protocol.json'),
                                     source_checkpoint_sha256=sha(target / 'restart.npz'),
                                     gpu_library_sha256=sha(gpu / 'lib/libpetsc.so')))
    atomic(parent / 'state.json', dict(status='comparison_complete'))
    atomic(parent / 'result.json', dict(complete=True, outcomes={'4': dict(completed=dict(
        factor=4, start_state=550, stop_state=555, complete=True, records=rows, completed_substeps=20))}))
    for name in ('plan.json', 'result.json', 'seed-550.npz', 'patch.npz', 'pressure.json'):
        artifact(root, parent, parent / name, 'Qualified refinement fixture')
    run = root / 'experiments/hpc/production'
    production.initialize(run, parent)
    fake = SimpleNamespace(default_parameters_benchmark1=lambda: dict(t_sys_pre=1., t_dias_pre=1., gamma=1.),
                           pressure_function=lambda span, parameters, t_eval: 10000+t_eval**2)
    monkeypatch.setitem(sys.modules, 'lv_surrogate.physics.reference_fem', fake)
    production.prepare_schedule(run)
    artifact(root, run, run / 'pressure.json', 'Production pressure schedule')
    return run, parent, target


def test_full_production_inherits_completed_quarter_step_without_changing_original(production_run):
    run, parent, target = production_run
    plan = production.load(run / 'plan.json')
    assert plan['refined_production'] and plan['dt_s'] == 1/6400
    assert plan['requested_substeps'] == 4200 and plan['stop_state'] == 1600
    assert plan['walltime_seconds'] == 168*3600
    assert (target / 'restart.npz').read_bytes() == b'unchanged original checkpoint'
    assert production.load(target / 'manifest.json')['completed_steps'] == 554
    schedule = production.load(run / 'pressure.json')['factors']['4']
    prefix = production.load(parent / 'pressure.json')['factors']['4']
    for name in ('endpoint_times', 'pressure_times', 'pressures'):
        assert schedule[name][:20] == prefix[name]
    assert schedule['endpoint_times'][20] == 555/1600+1/6400
    assert schedule['endpoint_times'][-1] == 1.
    records = production.production_records(run, verify_all_checkpoints=True)
    assert len(records) == 20 and 'whole_element_proof' not in records[-1]
    checkpoint = refined.restored_checkpoint(run / 'factor-4/substep-020/checkpoint.npz', records[-1], 4)
    np.testing.assert_array_equal(checkpoint['a_old'], np.full(3, 3.))
    np.testing.assert_array_equal(checkpoint['previous'], np.full(3, 19))


def test_resume_ignores_unaccepted_newton_attempt(production_run):
    run, _, _ = production_run
    pending = run / 'factor-4/substep-021'
    pending.mkdir()
    atomic(pending / 'trace-slice-001.json', dict(custom_force_N=.1))
    (pending / 'checkpoint.npz').write_bytes(b'uncommitted candidate')
    assert len(production.production_records(run)) == 20


@pytest.mark.parametrize('name', ['accepted.json', 'checkpoint.npz', 'trace.json'])
def test_production_rejects_changed_accepted_evidence(production_run, name):
    run, _, _ = production_run
    (run / 'factor-4/substep-020' / name).write_text('{}')
    with pytest.raises(AssertionError):
        production.production_records(run)


def test_pbs_successor_uses_full_walltime_four_gpus_and_afterok(production_run, monkeypatch):
    run, _, _ = production_run
    commands = []

    def qsub(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='12345.pbs\n', stderr='', check_returncode=lambda: None)

    monkeypatch.setattr(production.subprocess, 'run', qsub)
    assert production.submit_slice(run, 2, after='12344.pbs') == '12345.pbs'
    assert 'walltime=168:00:00' in commands[0]
    assert 'select=1:ncpus=28:mem=240gb:ngpus=4' in commands[0]
    assert 'depend=afterok:12344.pbs' in commands[0]


def test_final_verification_obeys_slice_deadline(production_run):
    run, _, _ = production_run

    def pause():
        raise refined.SliceDeadline('planned pause')

    with pytest.raises(refined.SliceDeadline):
        production.production_records(run, verify_all_checkpoints=True, check_deadline=pause)


@pytest.mark.parametrize('worker_exit', [0, 1])
def test_only_a_planned_pause_automatically_submits_a_successor(production_run, monkeypatch, worker_exit):
    run, _, target = production_run
    nodes = run / 'nodes'
    nodes.write_text('vis01\n')
    monkeypatch.setenv('PBS_JOBID', '12344.pbs')
    monkeypatch.setenv('PBS_NODEFILE', str(nodes))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-a,GPU-b,GPU-c,GPU-d')
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        if command[0] == 'qsub':
            return SimpleNamespace(returncode=0, stdout='12345.pbs\n', stderr='', check_returncode=lambda: None)
        atomic(run / 'factor-4/summary.json', dict(complete=False, completed_substeps=20,
                                                 paused_for_walltime=worker_exit == 0))
        return SimpleNamespace(returncode=worker_exit)

    monkeypatch.setattr(production.subprocess, 'run', execute)
    if worker_exit:
        with pytest.raises(AssertionError, match='Native worker exited 1'):
            production.execute(run)
        assert len(commands) == 1
        assert production.load(run / 'state.json')['status'] == 'blocked'
    else:
        production.execute(run)
        assert len(commands) == 2 and 'depend=afterok:12344.pbs' in commands[-1]
        state = production.load(run / 'state.json')
        assert state['status'] == 'awaiting_next_slice' and state['next_job'] == '12345.pbs'
    assert (target / 'restart.npz').read_bytes() == b'unchanged original checkpoint'
