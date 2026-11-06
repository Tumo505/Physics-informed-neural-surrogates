"""One submission for evidence-gated eight-worker qualification and reference slices."""
import argparse
import fcntl
import json
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'hpc'))
from registry import atomic, artifact, event, read, sha
from submit_probe_profile import assert_idle, registered
from manuscript import render


def parent_evidence(root, parent):
    plan = json.loads((parent / 'plan.json').read_text())
    for name in ['plan.json', 'summary.json', 'result.json']:
        registered(root, parent, parent / name)
    summary = json.loads((parent / 'summary.json').read_text())
    result = json.loads((parent / 'result.json').read_text())
    assert summary['complete'] and summary['proof_benchmark_passed'] and summary['stop_reason'] is None
    assert result['complete'] and result['passed'] and len(result['records']) == 10
    assert result['worker_order'] == [4, 1, 2, 8, 16, 16, 8, 2, 1, 4]
    rows = {}
    for workers in [1, 2, 4, 8, 16]:
        measurements = [r for r in result['records'] if r['workers'] == workers]
        assert len(measurements) == 2
        assert all(r['semantic_proof_agreement'] and not r['profiled'] and r['pair_seconds'] > 0
                   for r in measurements)
        rows[workers] = statistics.median(r['pair_seconds'] for r in measurements)
    assert rows[8] < .95 * rows[4], 'Eight-worker gain insufficient for this fixed follow-up'
    assert rows[16] > .95 * rows[8], 'Sixteen-worker benefit requires separate review'
    bindings = dict(plan['parent_files_sha256'])
    # The inherited profile/snapshot/code evidence must remain unchanged.
    for mapping in ['parent_files_sha256', 'scientific_sources', 'diagnostic_sources']:
        for path, digest in plan[mapping].items():
            assert sha(root / path) == digest, 'Parent evidence/source changed: ' + path
    for index, row in enumerate(result['records']):
        proof = parent / f'proof-{index:02d}.json'
        assert sha(proof) == row['proof_sha256']
        bindings[str(proof.relative_to(root))] = sha(proof)
    for name in ['plan.json', 'summary.json', 'result.json', 'state.json', 'events.jsonl']:
        bindings[str((parent / name).relative_to(root))] = sha(parent / name)
    profile = root / 'experiments/hpc' / plan['parent_run']
    profile_plan = json.loads((profile / 'plan.json').read_text())
    original = root / 'experiments/hpc' / profile_plan['parent_run']
    original_plan = json.loads((original / 'plan.json').read_text())
    original_state = json.loads((original / 'state.json').read_text())
    registered(root, original, original / 'plan.json')
    assert original_state['terminal_rejection'] and original_state['phase'] == 'known_probe'
    assert {'platform', 'audit_resume_control'} <= set(original_state['completed_phases'])
    assert original_state['generation_seconds'] == 0
    snapshot = json.loads((root / original_plan['snapshot']).read_text())
    assert sha(root / original_plan['snapshot']) == original_plan['snapshot_sha256']
    audit_path = root / original_state['platform_output']
    registered(root, original, audit_path)
    audit = json.loads(audit_path.read_text())
    assert audit['complete'] and audit['passed'] and audit['audit_complete']
    assert audit['checked_steps'] == audit['total_steps'] == snapshot['accepted_steps']
    assert len(audit['records']) == snapshot['accepted_steps']
    assert all(0 <= r['native_residual_N'] < 1e-8 for r in audit['records'])
    for key in ['canonical_checkpoint', 'known_start_checkpoint']:
        registered(root, original, root / original_plan[key])
    elapsed = summary['cumulative_seconds']
    assert elapsed >= plan['cumulative_seconds_before']
    assert elapsed + 5400 < original_plan['cumulative_pipeline_budget_hours'] * 3600
    return original_plan, original_state, elapsed, bindings, rows


def prepare(root, run_id, parent_id):
    parent = root / 'experiments/hpc' / parent_id
    source, inherited, elapsed, bindings, timings = parent_evidence(root, parent)
    assert_idle(root)
    run = root / 'experiments/hpc' / run_id
    assert not (root / 'data/processed' / ('cardiac_hpc_' + run_id)).exists(), 'Existing target retained'
    run.mkdir()
    snapshot = json.loads((root / source['snapshot']).read_text())
    config = json.loads((root / snapshot['source_config']).read_text())
    config['name'] = 'cardiac_hpc_' + run_id
    config['scope'] = 'Separate eight-proof-worker MPI reference follow-up; unchanged physics and final1e-8N. '
    config_path = run / 'reference-config.json'
    atomic(config_path, config)
    sources = dict(source['scientific_sources'])
    for file in sorted((root / 'hpc').glob('*.py')):
        sources[str(file.relative_to(root))] = sha(file)
    plan = dict(source, run_id=run_id, target_name=config['name'],
                config=str(config_path.relative_to(root)),
                reference_config_sha256=sha(config_path),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                inherited_evidence_sha256=bindings, benchmark_pair_medians=timings,
                prior_resource_evidence=dict(parent_run=parent_id, cumulative_seconds=elapsed,
                                             generation_seconds=0),
                scientific_sources=sources,
                stage_seconds=dict(known_probe=5400, known_probe_independent=1800),
                stage_budget_rationale='The known pair contains two unchanged1800s custom solves plus measured '
                                      'input hashing/mesh read and initial/final proof checks. The5400s outer '
                                      'phase bound does not extend either solve or the192h cumulative budget.',
                worker_policy=dict(workers=8, per_step_seconds=1800,
                                   benchmark_summary=str((parent / 'summary.json').relative_to(root)),
                                   benchmark_summary_sha256=sha(parent / 'summary.json')),
                scope='Distinct eight-worker known-pair qualification, independent serial native audit, '
                      'then seed/seed audit and bounded reference continuation only on success. '
                      'The full476-state platform audit is reused by exact provenance/input/source hashes. '
                      'No failed known probe is reused or retried; a new measured worker policy is qualified. '
                      'Original1800s per-equation,168h generation and192h cumulative caps unchanged. '
                      'Missing full-P2 training/calibration/final executors and failed1% controls remain gates.')
    atomic(run / 'plan.json', plan)
    state = dict(status='prepared', phase='known_probe', completed_phases=['audit_resume_control', 'platform'],
                 cumulative_seconds=elapsed, generation_seconds=0, attempts={}, blockers=[],
                 audit_progress=dict(phase='platform', checked_steps=snapshot['accepted_steps'],
                                     total_steps=snapshot['accepted_steps'], audit_complete=True, inherited=True),
                 platform_output=inherited['platform_output'])
    atomic(run / 'state.json', state)
    event(run, 'run_prepared', worker_policy=plan['worker_policy'], carried_cumulative_seconds=elapsed)
    artifact(root, run, run / 'plan.json', 'Immutable eight-worker follow-up plan')
    artifact(root, run, config_path, 'Same physical reference configuration; separate output name')
    artifact(root, run, root / inherited['platform_output'], 'Inherited full476-state passing native platform audit')
    artifact(root, run, parent / 'summary.json', 'Passing fixed-field worker benchmark')
    render(root, run)
    return run, plan


def launch(root, run_id, parent_id, jobs=8):
    assert 1 <= jobs <= 8
    run, plan = prepare(root, run_id, parent_id)
    dependency = None
    for index in range(1, jobs + 1):
        script = run / f'job-{index:03d}.pbs'
        script.write_text('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(root)) +
                          '\nexport PATH="$PWD/.hpc/tools/bin:$PATH"\nexec .venv/bin/python '
                          'hpc/run_worker8.py --run ' + shlex.quote(str(run.relative_to(root))) +
                          ' --slice-seconds 86400\n')
        command = ['qsub', '-N', 'lv_worker8', '-q', plan['queue'], '-l',
                   f"select=1:ncpus={plan['cpus']}:mem={plan['memory_gb']}gb",
                   '-l', 'walltime=24:00:00', '-l', 'place=excl', '-l', 'software=' + plan['software'],
                   '-j', 'oe', '-o', str(run / f'pbs-{index:03d}.log')]
        if plan.get('account'):
            command += ['-A', plan['account']]
        if dependency:
            command += ['-W', 'depend=afterok:' + dependency]
        result = subprocess.run(command + [str(script)], cwd=root, text=True, capture_output=True)
        event(run, 'job_submission', argv=command + [str(script)], return_code=result.returncode,
              stdout=result.stdout.strip(), stderr=result.stderr.strip())
        if result.returncode:
            raise SystemExit('Submission stopped; preserve existing jobs and inspect ledger: ' + result.stderr)
        dependency = result.stdout.strip().split()[0]
        assert re.fullmatch(r'[A-Za-z0-9_.-]+', dependency)
        print(f'slice_{index}: {dependency}', flush=True)
    print('Known-pair qualification precedes any new physical state. Failures stop the chain; logout is safe.')
    print('Status: .venv/bin/python hpc/status.py --run ' + str(run.relative_to(root)))
    return run


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--parent-run', default='cmbbe-proof-01')
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()
    for value in [args.run_id, args.parent_run]:
        assert re.fullmatch('[A-Za-z0-9_-]+', value), 'Invalid run ID'
    with (P / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch(P, args.run_id, args.parent_run, args.jobs)
