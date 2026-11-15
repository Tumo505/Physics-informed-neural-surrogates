import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import qualify_reference_restart as restart
from registry import artifact, atomic, sha


@pytest.mark.parametrize('marker', [None, restart.TIMEOUT, 'different error'])
def test_registered_timeout_with_optional_marker(tmp_path, monkeypatch, marker):
    monkeypatch.setattr(restart, 'ROOT', tmp_path)
    phase = 'reference_000547'
    parent = tmp_path / 'parent'
    target = tmp_path / 'target'
    target.mkdir()
    log = parent / phase / 'attempt-001/command.log'
    log.parent.mkdir(parents=True)
    log.write_text(f'RuntimeError: {restart.TIMEOUT}\n')
    artifact(tmp_path, parent, log, 'Executed command log')
    state = dict(phase=phase, attempts={phase: 1})

    if marker is not None:
        atomic(target / 'failure.json', dict(error=marker))
    if marker == 'different error':
        with pytest.raises(AssertionError, match='different failure marker'):
            restart.failure_evidence(parent, target, state)
        return

    expected = {} if marker is None else {'failure.json': sha(target / 'failure.json')}
    assert restart.failure_evidence(parent, target, state) == (log, expected)
    log.write_text('Changed log')
    with pytest.raises(AssertionError, match='Unregistered or changed artifact'):
        restart.failure_evidence(parent, target, state)


def test_state_552_plan_does_not_inherit_state_551_solver_inputs():
    from recover_reference_552 import recovery_plan, SHARED_FIELDS

    pilot = dict.fromkeys(SHARED_FIELDS, 'shared input')
    prior_solver = dict(saved_iterate='previous-iterate.npz', saved_iterate_force_N=9.34e-6,
                        warm_shift=0.001, screened_force_N=0.06, screen_predictors=True,
                        compare_shifts=True, fixed_shift=0.01)
    pilot.update(prior_solver)
    plan = recovery_plan(pilot, step=551, screen_and_solve=True, full_solve=True)
    assert not (prior_solver.keys() & plan.keys())
    assert plan['stream_protocol'] == pilot['stream_protocol']
    assert plan['step'] == 551 and plan['screen_and_solve'] and plan['full_solve']


def test_state_552_warm_iterate_must_match_verified_summary(tmp_path):
    import numpy as np
    from recover_reference_552 import saved_iterate_parameters

    path = tmp_path / 'last-newton.npz'
    np.savez(path, step=551, accepted=False, force_N=.087, next_shift=.0625, iteration=35)
    summary = dict(last_newton_iterate_sha256=sha(path), last_accepted_custom_force_N=.087)
    options = dict(minimum_regularization=1e-9, maximum_regularization=100)
    assert saved_iterate_parameters(path, summary, options) == (.087, .0625, 35)
    summary['last_accepted_custom_force_N'] = .09
    with pytest.raises(AssertionError):
        saved_iterate_parameters(path, summary, options)


def test_state_552_followup_checks_final_trend():
    from recover_reference_552 import verified_final_trend

    forces = [.08, .06, .03, .002, .00005]
    rows = [dict(iteration=i, next_regularization=.0625, accepted_custom_update=True,
                 trials=[dict(native_force_N=force, whole_element_admissible=True, armijo_passed=True)])
            for i, force in enumerate(forces)]
    trace = dict(step=551, custom_newton_steps=rows)
    summary = dict(accepted_custom_iterations=5)
    assert verified_final_trend(trace, summary, forces[-1], .0625, 4) == forces
    rows[2]['trials'][0]['native_force_N'] = .07
    with pytest.raises(AssertionError, match='did not decrease'):
        verified_final_trend(trace, summary, forces[-1], .0625, 4)


def test_admit_audited_state_552_updates_checkpoint_once(tmp_path, monkeypatch):
    import json
    import numpy as np
    import admit_recovery_551 as admission

    monkeypatch.setattr(admission, 'ROOT', tmp_path)
    run = tmp_path / 'experiments/hpc/recovery'
    parent = run.parent / 'parent'
    target = tmp_path / 'data/processed/reference'
    run.mkdir(parents=True)
    parent.mkdir()
    (target / 'step_journal').mkdir(parents=True)
    old = np.zeros(12)
    u = np.arange(12, dtype=float)
    old_history = dict.fromkeys(('u_old', 'v_old', 'a_old'), old)
    restart_path = target / 'restart.npz'
    np.savez(restart_path, completed_steps=551, u=old, **old_history)
    np.savez(target / 'restart.previous.npz', completed_steps=550, u=old, **old_history)
    np.savez(target / 'inputs.npz', time=np.arange(552), cells=[[0, 1, 2, 3]],
             coords=np.zeros((4, 3)), canonical_cell_dofs=np.arange(12).reshape(1, 4, 3))
    np.save(target / 'displacement_coefficients.npy', np.zeros((552, 12)))
    np.save(target / 'displacements.npy', np.zeros((552, 4, 3)))
    atomic(target / 'positivity.json', [{}] * 551)
    atomic(target / 'manifest.json', dict(completed_steps=551, n_steps=1600, steps=[{}] * 551,
                                          prefix_independent_native_audit_passed=True,
                                          prefix_positivity_passed=True, config_sha256='config',
                                          source_inputs_sha256=sha(target / 'inputs.npz')))
    config = tmp_path / 'config.json'
    atomic(config, {})
    frozen = {str(path.relative_to(tmp_path)): sha(path) for path in
              (config, target / 'manifest.json', restart_path)}
    atomic(run / 'plan.json', dict(step=551, warm_shift=.01, warm_restart_552=True,
                                   warm_followup_parent='warm', target_name='reference',
                                   parent_run='parent', config='config.json', frozen=frozen))
    atomic(run / 'summary.json', dict(complete=True, passed=True, step=551, native_updates=0,
                                      SNES_reason=2, native_force_N=2e-10, proof_agreement=True,
                                      physical_new_state_generated=False))
    candidate = run / 'candidate.npz'
    np.savez(candidate, accepted=False, step=551, time_s=551, u=u, **old_history,
             next_u_old=u, next_v_old=u + 1, next_a_old=u + 2)
    proof = {stage: dict(passed=True, minimum_Bernstein_bound=.1) for stage in ('endpoint', 'stress')}
    atomic(run / 'independent-audit.json', dict(passed=True, step=551, history_passed=True,
                                                candidate_sha256=sha(candidate), native_force_N=2e-10,
                                                norm_relative_error=0., whole_element_proof=proof))
    for name in ('plan.json', 'summary.json', 'independent-audit.json', 'candidate.npz'):
        artifact(tmp_path, run, run / name, name)

    admission.admit(run, step=551)
    digest = sha(restart_path)
    admission.admit(run, step=551)
    assert sha(restart_path) == digest
    assert sha(target / 'restart.previous.npz') == frozen['data/processed/reference/restart.npz']
    assert json.loads((target / 'manifest.json').read_text())['completed_steps'] == 552
    assert (target / 'step_journal/000551.json').exists()
    with np.load(restart_path) as checkpoint:
        assert int(checkpoint['completed_steps']) == 552
        np.testing.assert_array_equal(checkpoint['a_old'], u + 2)


def test_fresh_solver_keeps_force_and_time_gates(monkeypatch):
    from types import ModuleType
    from fresh_reference_native import solve_fresh

    native = ModuleType('lv_surrogate.physics.admissible_newton_gentle_shrink')
    seen = {}

    def solve(problem, adapter, history, audit, options, trace, persist, **policy):
        seen.update(policy)
        trace['custom_force_N'] = 2e-10

    native.solve = solve
    monkeypatch.setitem(sys.modules, native.__name__, native)
    trace = {}
    options = dict(custom_absolute_tolerance_N=1e-9, wall_budget_seconds=1800)
    solve_fresh(None, None, None, None, options, trace, lambda: None)
    assert trace['solver_policy'] == 'fresh_jacobian_two_good_steps'
    assert seen == dict(policy='two_good_steps', reuse_jacobian=False, reuse_gamg=False,
                        late_forcing_floor=1e-5)
    with pytest.raises(AssertionError):
        solve_fresh(None, None, None, None, dict(options, wall_budget_seconds=1801),
                    {}, lambda: None)


def test_production_submits_a_single_dependency_chain(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import launch_fresh_production as production

    monkeypatch.setattr(production, 'ROOT', tmp_path)
    commands = []

    def qsub(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=f'{len(commands)}.pbs\n', stderr='')

    monkeypatch.setattr(production.subprocess, 'run', qsub)
    production.submit(tmp_path / 'run', dict(queue='nvidia', cpus=28, memory_gb=240,
                                            software='PBSJobApp'), dict(status='submitted'))
    assert len(commands) == production.SLICES
    assert '-W' not in commands[0]
    for index, command in enumerate(commands[1:], start=1):
        assert command[command.index('-W') + 1] == f'depend=afterok:{index}.pbs'


def test_production_throughput_retains_final_stage_reserve():
    from launch_fresh_production import throughput_limit

    plan = dict(native_generation_budget_hours=168, cumulative_pipeline_budget_hours=192)
    state = dict(generation_seconds=30 * 3600, cumulative_seconds=58 * 3600)
    assert throughput_limit(plan, state, 1000) == 360
