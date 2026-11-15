"""Commit an independently audited recovery using the reference journal."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from registry import sha
from submit_probe_profile import assert_idle, registered

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lv_surrogate.physics.durable_reference import atomic_json, recover

def read(path):
    return json.loads(path.read_text())


def admit(run, step=550):
    assert step in (550, 551)
    for name in ('plan.json', 'summary.json', 'independent-audit.json', 'candidate.npz'):
        registered(ROOT, run, run / name)
    plan = read(run / 'plan.json')
    result = read(run / 'summary.json')
    audit_path = run / 'independent-audit.json'
    audit = read(audit_path)
    candidate_path = run / 'candidate.npz'
    candidate_hash = sha(candidate_path)
    assert plan['step'] == step and plan['warm_shift'] is not None
    if step == 551:
        assert plan['warm_restart_552'] and plan['warm_followup_parent']
        assert result['proof_agreement'] and not result['physical_new_state_generated']
    assert result['complete'] and result['passed'] and result['step'] == audit['step'] == step
    assert result['native_updates'] == 0 and result['SNES_reason'] > 0
    assert 0 <= result['native_force_N'] < 1e-9
    assert audit['passed'] and audit['history_passed'] and audit['candidate_sha256'] == candidate_hash
    assert 0 <= audit['native_force_N'] < 1e-8 and 0 <= audit['norm_relative_error'] <= 1e-11
    proof = audit['whole_element_proof']
    for stage in ('endpoint', 'stress'):
        assert proof[stage]['passed'] and proof[stage]['minimum_Bernstein_bound'] > 1e-9

    target = ROOT / 'data/processed' / plan['target_name']
    manifest_path = target / 'manifest.json'
    restart = target / 'restart.npz'
    journal = target / 'step_journal' / f'{step:06d}.json'
    parent = ROOT / 'experiments/hpc' / plan['parent_run']
    provenance = dict(candidate_sha256=candidate_hash, independent_audit_sha256=sha(audit_path),
                      run=str(run.relative_to(ROOT)))
    with (parent / 'execution.lock').open('a') as execution, (target / 'generation.lock').open('a') as generation:
        for lock in (execution, generation):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(manifest_path)
        count = manifest['completed_steps']
        if count > step:
            assert manifest['steps'][step]['recovery_admission'] == provenance
            print(f'State {step + 1} is already admitted. Accepted states: {count}/{manifest["n_steps"]}')
            return
        assert count == step
        assert manifest['prefix_independent_native_audit_passed'] and manifest['prefix_positivity_passed']
        for path in (manifest_path, ROOT / plan['config']):
            assert sha(path) == plan['frozen'][str(path.relative_to(ROOT))]
        restart_hash = plan['frozen'][str(restart.relative_to(ROOT))]
        fields = np.load(target / 'displacement_coefficients.npy', mmap_mode='r+')
        if journal.exists():
            saved = read(journal)
            assert saved['input_checkpoint_sha256'] == restart_hash
            assert saved['record']['recovery_admission'] == provenance
            assert recover(target, manifest, fields)
            print(f'Recovered admission of state {step + 1}. Accepted states: {manifest["completed_steps"]}/{manifest["n_steps"]}')
            return

        assert sha(restart) == restart_hash
        inputs_path = target / 'inputs.npz'
        assert sha(inputs_path) == manifest['source_inputs_sha256']
        with np.load(candidate_path) as candidate, np.load(restart) as prior, np.load(inputs_path) as inputs:
            assert not bool(candidate['accepted'])
            assert int(prior['completed_steps']) == int(candidate['step']) == step
            names = ('u_old', 'v_old', 'a_old')
            for name in names:
                np.testing.assert_array_equal(candidate[name], prior[name])
            u = candidate['u']
            next_history = {name: candidate['next_' + name] for name in names}
            assert u.shape == fields.shape[1:]
            assert all(field.shape == u.shape and np.isfinite(field).all()
                       for field in (u, *next_history.values()))
            np.testing.assert_array_equal(u, next_history['u_old'])
            time_s = float(inputs['time'][step])
            assert time_s == float(candidate['time_s'])
            maps = inputs['canonical_cell_dofs']
            cells = inputs['cells']
            node_map = np.full((len(inputs['coords']), 3), -1, dtype=np.int64)
            for component in range(3):
                for vertex in range(4):
                    node_map[cells[:, vertex], component] = maps[:, vertex, component]
            assert (node_map >= 0).all()
            nodes = np.load(target / 'displacements.npy', mmap_mode='r+')
            assert nodes[step].shape == node_map.shape
            assert len(read(target / 'positivity.json')) == step

        # Retain the pre-admission manifest and preceding restart by hard link.
        for path in (manifest_path, target / 'restart.previous.npz'):
            archive = run / ('pre-admission-' + path.name)
            if not archive.exists():
                os.link(path, archive)
            else:
                assert sha(archive) == sha(path)

        checkpoint = target / 'restart.pending.npz'
        with checkpoint.open('wb') as handle:
            np.savez_compressed(handle, completed_steps=step + 1, u=u, **next_history)
            handle.flush()
            os.fsync(handle.fileno())
        fields[step] = u
        nodes[step] = u[node_map]
        for array, filename in ((fields, 'displacement_coefficients.npy'), (nodes, 'displacements.npy')):
            array.flush()
            with (target / filename).open('rb') as handle:
                os.fsync(handle.fileno())
        record = dict(step=step, time_s=time_s, SNES_reason=result['SNES_reason'],
                      nonlinear_iterations=0, residual_l2_N=audit['native_force_N'],
                      whole_element_positivity_passed=True, recovery_admission=provenance)
        atomic_json(journal, dict(config_sha256=manifest['config_sha256'], record=record,
                                  positivity=dict(step=step, time_s=time_s, **proof),
                                  coefficient_sha256=hashlib.sha256(u.tobytes()).hexdigest(),
                                  input_checkpoint_sha256=restart_hash,
                                  accepted_checkpoint_sha256=sha(checkpoint)))
        assert recover(target, manifest, fields)
        print(f'Admitted state {step + 1}. Accepted states: {manifest["completed_steps"]}/{manifest["n_steps"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=int, choices=(551, 552), default=551)
    parser.add_argument('--run')
    args = parser.parse_args()
    if args.run is None:
        args.run = 'cmbbe-recovery-551-resume-01' if args.state == 551 else 'cmbbe-warm-552-02'
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('Supply a run ID')
    assert_idle(ROOT)
    admit(ROOT / 'experiments/hpc' / args.run, step=args.state - 1)
