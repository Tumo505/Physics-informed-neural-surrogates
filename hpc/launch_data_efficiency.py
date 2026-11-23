"""Four-GPU, full-walltime scheduler for the frozen 24-model comparison.

No FE files/checkpoints are modified. One process owns each allocated GPU.
Only walltime pauses queue a successor; scientific/runtime failures stop the run.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from data_efficiency import digest, write

WALL_SECONDS = 168 * 3600
# Leave ample room for the current epoch, checkpoint and PBS submission.
CHECKPOINT_RESERVE = 1800


def load(path):
    return json.loads(path.read_text())


def task_list(cfg):
    # Interleave paired architectures and training sizes across GPUs.
    return [(size, kind, seed) for seed in cfg['seeds'] for size in cfg['sizes'] for kind in ('gnn', 'point')]


def initialize(run, python):
    if not python.is_file() and python != ROOT / '.hpc/surrogate-ml/bin/python':
        raise FileNotFoundError(f'{python}: run bash hpc/setup_surrogate_torch.sh first, or pass --python')
    required = [ROOT / 'data/raw/lvData/rawData' / split / (name + '.npy')
                for split in ('train', 'validation', 'test') for name in
                ('real-node-coords', 'real-node-features', 'global-features', 'real-node-displacement')]
    required += [ROOT / 'external/passive-lv-gnn-emul/data/lvDataFixedGeom/topologyData' / name for name in
                 ('kmeans-labels-list.npy', 'node-layer-labels.npy', 'sparse-topology.npy')]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Stage the supplied benchmark files before submitting: ' + ', '.join(missing))
    run.mkdir(parents=True, exist_ok=False)
    cfg = load(ROOT / 'configs/gnn/data_efficiency.json')
    write(run / 'protocol.json', cfg)
    sources = [ROOT / 'scripts/data_efficiency.py', Path(__file__),
               ROOT / 'src/lv_surrogate/models/multiscale.py', ROOT / 'configs/gnn/data_efficiency.json',
               ROOT / 'hpc/setup_surrogate_torch.sh']
    write(run / 'plan.json', dict(python=str(python), source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
          protocol_sha256=digest(run / 'protocol.json'), tasks=task_list(cfg), gpus=4, walltime='168:00:00',
          successor_policy='Automatic after allocation pause only; no total experiment time cap',
          validation_selection='All 150 validation geometries; no test-dependent tuning',
          epochs='500 maximum per model, validation every 5, stop after 25 checks without improvement',
          shape_coefficients='Excluded; their fitted provenance is not established',
          capacity_control='Pointwise decoder width chosen to match graph parameter count within 2%'))
    (run / 'job.pbs').write_text('#!/bin/bash\nset -eu\ncd ' + shlex.quote(str(ROOT)) + '\nexec ' +
          shlex.join([str(ROOT / '.venv/bin/python'), str(Path(__file__).resolve()), 'execute', '--run-id', run.name]) + '\n')
    write(run / 'state.json', dict(status='prepared', slice=0, completed_models=0, requested_models=24))


def submit(run):
    state = load(run / 'state.json')
    index = state['slice'] + 1
    reply = subprocess.run(['qsub', '-N', 'lv_learning', '-q', 'iworkq', '-l',
            'select=1:ncpus=28:mem=240gb:ngpus=4', '-l', 'place=excl', '-l', 'software=PBSJobApp',
            '-l', 'walltime=168:00:00', '-j', 'oe', '-o', str(run / f'pbs-{index:03d}.log'),
            str(run / 'job.pbs')], text=True, capture_output=True, check=True)
    job = reply.stdout.strip()
    if not re.fullmatch(r'\d+(?:\.[\w.-]+)?', job):
        raise RuntimeError(f'Unexpected PBS submission reply: {job}')
    state.update(status='submitted', slice=index, job=job)
    write(run / 'state.json', state)
    print('Data-efficiency experiment job:', job, flush=True)


def verify_sources(run):
    plan = load(run / 'plan.json')
    for name, value in plan['source_sha256'].items():
        if digest(ROOT / name) != value:
            raise RuntimeError(f'Frozen source changed: {name}; restore that version before resuming')
    if digest(run / 'protocol.json') != plan['protocol_sha256']:
        raise RuntimeError('Frozen protocol changed')


def worker(run, slot, deadline):
    import copy
    import torch
    from data_efficiency import model_for, train
    cfg = load(run / 'protocol.json')
    # A real forward/backward detects unsupported M60 kernels, not just GPU discovery.
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch is unavailable; inspect the separate ML environment')
    torch.set_num_threads(cfg['threads_per_worker'])
    reference = model_for('gnn', cfg)
    model = copy.deepcopy(reference).cuda()
    coarse = torch.randn(2, 3, 13, device='cuda')
    globals_ = torch.randn(2, 4, device='cuda')
    points = torch.randn(2, 4, 13, device='cuda')
    edges = torch.randn(2, 4, 4, device='cuda')
    topo = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], device='cuda')
    parent = torch.tensor([0, 1, 1, 2], device='cuda')
    prediction = model(points, coarse, globals_, edges, topo, parent)
    prediction.square().mean().backward()
    cpu_prediction = reference(*[v.cpu() for v in (points, coarse, globals_, edges, topo, parent)])
    cpu_prediction.square().mean().backward()
    torch.testing.assert_close(prediction.cpu(), cpu_prediction, atol=1e-4, rtol=1e-4)
    for left, right in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad.cpu(), right.grad, atol=1e-4, rtol=1e-4)
    torch.cuda.synchronize()
    write(run / f'gpu-{slot}.json', dict(torch=torch.__version__, cuda_runtime=torch.version.cuda,
          gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
          compiled_architectures=torch.cuda.get_arch_list(), real_graph_backward_passed=True,
          cpu_cuda_prediction_gradient_agreement=True))
    del model, reference
    # File claims prevent duplicate execution and distribute expensive models dynamically.
    for size, kind, seed in task_list(cfg):
        if time.time() >= deadline:
            return 75
        task = run / 'tasks' / f'{size}-{kind}-{seed}'
        task.mkdir(parents=True, exist_ok=True)
        with (task / 'owner.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            progress = task / 'progress.json'
            if progress.exists() and load(progress)['complete']:
                continue
            if not train(run, size, kind, seed, 'cuda:0', deadline):
                return 75
    return 0


def execute(run):
    # Separate execution lock tolerates a fast-starting queued successor.
    with (run / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_sources(run)
        plan = load(run / 'plan.json')
        state = load(run / 'state.json')
        index = state['slice']
        state.update(status='running')
        write(run / 'state.json', state)
        deadline = time.time() + WALL_SECONDS - CHECKPOINT_RESERVE
        children = []
        try:
            # Preserve PBS GPU isolation, including UUID-based allocations.
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if visible:
                devices = visible.split(',')
            else:
                devices = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()
            if len(devices) != 4 or len(set(devices)) != 4:
                raise RuntimeError(f'Expected exactly four allocated GPUs, found {devices}')
            (run / f'hardware-{index:03d}.log').write_text(subprocess.check_output(['nvidia-smi'], text=True))
            default_python = ROOT / '.hpc/surrogate-ml/bin/python'
            if Path(plan['python']) == default_python and not (default_python.parent.parent / '.installed').exists():
                with (run / f'setup-{index:03d}.log').open('a') as log:
                    subprocess.run(['bash', str(ROOT / 'hpc/setup_surrogate_torch.sh')],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            environment = subprocess.check_output([plan['python'], '-m', 'pip', 'freeze'], text=True)
            environment_file = run / 'python-environment.txt'
            if environment_file.exists() and environment_file.read_text() != environment:
                raise RuntimeError('Python environment changed since the previous allocation')
            environment_file.write_text(environment)
            with (run / f'prepare-{index:03d}.log').open('a') as log:
                subprocess.run([plan['python'], str(ROOT / 'scripts/data_efficiency.py'), 'prepare', '--run', str(run)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            for name, value in load(run / 'data/ready.json')['prepared_sha256'].items():
                if digest(run / 'data' / name) != value:
                    raise RuntimeError(f'Prepared data changed: {name}')
            for slot, gpu in enumerate(devices):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='6', MKL_NUM_THREADS='6',
                           OPENBLAS_NUM_THREADS='6', PYTHONUNBUFFERED='1')
                with (run / f'worker-{index:03d}-{slot}.log').open('a') as log:
                    child = subprocess.Popen([plan['python'], str(Path(__file__).resolve()), 'worker',
                        '--run-id', run.name, '--slot', str(slot), '--deadline', str(deadline)],
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                children.append(child)
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0, 75) for child in children):
                    raise RuntimeError('A GPU worker failed; inspect worker logs. Completed checkpoints retained.')
                completed = sum(path.exists() and load(path)['complete'] for path in
                    (run / 'tasks' / f'{size}-{kind}-{seed}' / 'progress.json' for size, kind, seed in task_list(load(run / 'protocol.json'))))
                state.update(completed_models=completed)
                write(run / 'state.json', state)
                time.sleep(10)
            if any(child.returncode not in (0, 75) for child in children):
                raise RuntimeError('GPU worker failed')
            complete = all((run / 'tasks' / f'{size}-{kind}-{seed}' / 'progress.json').exists() and
                           load(run / 'tasks' / f'{size}-{kind}-{seed}' / 'progress.json')['complete']
                           for size, kind, seed in task_list(load(run / 'protocol.json')))
            if not complete or time.time() >= deadline:
                state.update(status='paused')
                write(run / 'state.json', state)
                submit(run)
                return
            state.update(status='evaluating', completed_models=24)
            write(run / 'state.json', state)
            with (run / f'evaluate-{index:03d}.log').open('a') as log:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[0])
                reply = subprocess.run([plan['python'], str(ROOT / 'scripts/data_efficiency.py'), 'evaluate',
                               '--run', str(run), '--deadline', str(deadline)],
                               env=env, stdout=log, stderr=subprocess.STDOUT)
                if reply.returncode == 75:
                    state.update(status='paused')
                    write(run / 'state.json', state)
                    submit(run)
                    return
                reply.check_returncode()
            state.update(status='complete', result='evaluation/result.json')
            write(run / 'state.json', state)
        except BaseException as exc:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            state.update(status='failed', failure=f'{type(exc).__name__}: {exc}')
            write(run / 'state.json', state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'resume', 'execute', 'worker'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--python', type=Path, default=ROOT / '.hpc/surrogate-ml/bin/python')
    parser.add_argument('--slot', type=int)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('Invalid run ID')
    run = ROOT / 'experiments/hpc' / args.run_id
    if args.action in ('submit', 'resume'):
        with (ROOT / 'experiments/hpc/.learning-curve-submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'submit':
                initialize(run, args.python.resolve())
            else:
                verify_sources(run)
                state = load(run / 'state.json')
                if state['status'] in ('submitted', 'running', 'evaluating'):
                    # A killed allocation may leave stale state. Check this job only.
                    reply = subprocess.run(['qstat', '-f', state['job']], capture_output=True, text=True)
                    if reply.returncode == 0 and not re.search(r'job_state\s*=\s*F', reply.stdout):
                        raise RuntimeError('This experiment already has an active PBS job')
                    if reply.returncode != 0 and not re.search(r'unknown\s+job', reply.stderr + reply.stdout, re.I):
                        raise RuntimeError('Could not establish whether the existing PBS job has ended')
                if state['status'] == 'complete':
                    raise RuntimeError('Experiment is already complete')
            submit(run)
    elif args.action == 'worker':
        sys.exit(worker(run, args.slot, args.deadline))
    else:
        execute(run)
