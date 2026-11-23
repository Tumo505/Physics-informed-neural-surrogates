"""Frozen, paired learning curves. Training never opens test labels.

Units follow the supplied benchmark (cm); reported distances are mm.
Shape coefficients and released model weights/statistics are deliberately unused.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
NAMES = ('points', 'coarse', 'globals', 'edges', 'targets')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def subsets(count, sizes, seed):
    if not sizes or len(set(sizes)) != len(sizes) or min(sizes) < 1 or max(sizes) > count:
        raise ValueError('Training sizes must be unique, positive and within the training cohort')
    order = np.random.default_rng(seed).permutation(count)
    return {str(size): order[:size].tolist() for size in sizes}


def topology(source):
    from scipy.sparse import csr_matrix
    labels = np.load(source / 'kmeans-labels-list.npy', allow_pickle=True)
    first, second = (np.asarray(v, dtype=int) for v in labels[:2])
    layers = np.load(source / 'node-layer-labels.npy')
    edges = np.load(source / 'sparse-topology.npy').astype('int64')
    n1, n2 = int((layers == 1).sum()), int((layers == 2).sum())
    p1 = csr_matrix((1 / np.bincount(first)[first], (first, np.arange(len(first)))),
                    shape=(n1, len(first)))
    p2 = csr_matrix((1 / np.bincount(second)[second], (second, np.arange(len(second)))),
                    shape=(n2, n1))
    chosen = (layers[edges[:, 0]] == 2) & (layers[edges[:, 1]] == 2)
    edges = edges[chosen] - int((layers < 2).sum())
    return (p2 @ p1).astype('float32'), second[first], np.concatenate((edges, edges[:, ::-1]))


def raw_case(arrays, index):
    return {name: np.array(value[index if len(value) > 1 else 0], dtype='float32')
            for name, value in arrays.items()}


def fit_stats(arrays, ids):
    """Streaming float64 moments, exclusively on selected training geometries."""
    stats = {}
    for name, array in arrays.items():
        total = np.zeros(array.shape[-1], dtype='float64')
        square = total.copy()
        count = 0
        for index in ids:
            value = np.asarray(array[index], dtype='float64').reshape(-1, array.shape[-1])
            total += value.sum(0)
            square += (value * value).sum(0)
            count += len(value)
        mean = total / count
        stats[name + '_mean'] = mean.astype('float32')
        stats[name + '_std'] = np.maximum(np.sqrt(np.maximum(square / count - mean**2, 0)), 1e-6).astype('float32')
    return stats


def prepare(run, raw=None, graph_source=None):
    """Read all raw splits once; learn only four training-subset normalizers."""
    cfg = json.loads((run / 'protocol.json').read_text())
    cache = run / 'data'
    if (cache / 'ready.json').exists():
        if json.loads((cache / 'ready.json').read_text()).get('protocol_sha256') != digest(run / 'protocol.json'):
            raise RuntimeError('Prepared data belong to a different protocol')
        return
    raw = raw or ROOT / 'data/raw/lvData/rawData'
    graph_source = graph_source or ROOT / 'external/passive-lv-gnn-emul/data/lvDataFixedGeom/topologyData'
    cache.mkdir(exist_ok=True)
    pool, parents, edges = topology(graph_source)
    np.save(cache / 'parents.npy', parents)
    np.save(cache / 'topology.npy', edges)
    sources = {}
    counts = {}
    fields = ('real-node-coords', 'real-node-features', 'global-features', 'real-node-displacement')
    for split in ('train', 'validation', 'test'):
        arrays = {name: np.load(raw / split / (name + '.npy'), mmap_mode='r') for name in fields}
        sources.update({str((raw / split / (name + '.npy')).relative_to(ROOT)): digest(raw / split / (name + '.npy'))
                        for name in fields})
        count, nodes = arrays['real-node-displacement'].shape[:2]
        counts[split] = count
        if nodes != len(parents):
            raise ValueError('Supplied clustering does not match raw node correspondence')
        shapes = dict(points=(nodes, 13), coarse=(pool.shape[0], 13), globals=(4,),
                      edges=(len(edges), 4), targets=(nodes, 3))
        outputs = {name: np.lib.format.open_memmap(cache / f'{split}_{name}.npy', mode='w+',
                   dtype='float32', shape=(count, *shape)) for name, shape in shapes.items()}
        for index in range(count):
            case = raw_case(arrays, index)
            points = np.concatenate((case['real-node-coords'], case['real-node-features']), -1)
            coarse = pool @ points
            delta = coarse[edges[:, 1], :3] - coarse[edges[:, 0], :3]
            values = dict(points=points, coarse=coarse, globals=case['global-features'],
                          edges=np.c_[delta, np.linalg.norm(delta, axis=-1)], targets=case['real-node-displacement'])
            for name, value in values.items():
                if not np.isfinite(value).all():
                    raise ValueError(f'Nonfinite {split} {index} {name}')
                outputs[name][index] = value
        for value in outputs.values():
            value.flush()
        print('PREPARED', split, count, flush=True)
    selected = subsets(counts['train'], cfg['sizes'], cfg['subset_seed'])
    if counts != dict(train=2250, validation=150, test=567):
        raise ValueError(f'Unexpected benchmark split sizes: {counts}')
    train = load_data(cache, 'train')
    for size, ids in selected.items():
        np.savez(cache / f'normalizers-{size}.npz', **fit_stats(train, ids))
    sources.update({str(p.relative_to(ROOT)): digest(p) for p in graph_source.glob('*.npy')})
    write(cache / 'ready.json', dict(counts=counts, subsets=selected, source_sha256=sources,
          protocol_sha256=digest(run / 'protocol.json'),
          normalization='Each training subset only; shared by paired models and seeds',
          shape_coefficients_used=False, topology='Fixed supplied clustering, not fitted here',
          prepared_sha256={p.name: digest(p) for p in cache.glob('*.np*')}))


def load_data(cache, split):
    return {name: np.load(cache / f'{split}_{name}.npy', mmap_mode='r') for name in NAMES}


def model_for(kind, cfg):
    import torch
    from lv_surrogate.models.multiscale import MultiscaleSurrogate
    graph = MultiscaleSurrogate(global_dim=4, width=cfg['width'], steps=cfg['steps'])
    if kind == 'gnn':
        return graph
    target = sum(p.numel() for p in graph.parameters())
    def parameters(h):
        return (17 + 1) * h + (h + 1) * h + (h + 1) * (h // 2) + (h // 2 + 1) * 3
    h = min(range(16, 2049), key=lambda width: abs(parameters(width) - target))
    model = MultiscaleSurrogate(global_dim=4, graph=False)
    model.decoder = torch.nn.Sequential(torch.nn.Linear(17, h), torch.nn.SiLU(),
          torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, h // 2),
          torch.nn.SiLU(), torch.nn.Linear(h // 2, 3))
    assert abs(sum(p.numel() for p in model.parameters()) / target - 1) < .02
    return model


def sync(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def save_checkpoint(path, value):
    import torch
    temp = path.with_suffix('.pending.pt')
    torch.save(value, temp)
    temp.replace(path)


class Batches:
    def __init__(self, cache, size, device):
        import torch
        self.device = device
        self.parents = np.load(cache / 'parents.npy')
        self.topology = torch.as_tensor(np.load(cache / 'topology.npy'), device=device)
        with np.load(cache / f'normalizers-{size}.npz') as f:
            self.stats = {key: value.copy() for key, value in f.items()}
        self.umean = torch.as_tensor(self.stats['targets_mean'], device=device)
        self.ustd = torch.as_tensor(self.stats['targets_std'], device=device)

    def __call__(self, data, ids, queries=None):
        import torch
        ids = np.asarray(ids)
        batch = {}
        for name, array in data.items():
            value = array[ids[:, None], queries] if queries is not None and name in ('points', 'targets') else array[ids]
            if name != 'targets':
                value = (value - self.stats[name + '_mean']) / self.stats[name + '_std']
            batch[name] = torch.as_tensor(np.array(value, copy=True), device=self.device)
        batch['parents'] = torch.as_tensor(self.parents if queries is None else self.parents[queries], device=self.device)
        return batch

    def predict(self, model, batch):
        return model(batch['points'], batch['coarse'], batch['globals'], batch['edges'],
                     self.topology, batch['parents']) * self.ustd + self.umean


def train(run, size, kind, seed, device_name, deadline=float('inf')):
    import torch
    cfg = json.loads((run / 'protocol.json').read_text())
    torch.set_num_threads(cfg['threads_per_worker'])
    device = torch.device(device_name)
    out = run / 'tasks' / f'{size}-{kind}-{seed}'
    out.mkdir(parents=True, exist_ok=True)
    cache = run / 'data'
    data = load_data(cache, 'train')
    valid = load_data(cache, 'validation')
    ids = np.array(json.loads((cache / 'ready.json').read_text())['subsets'][str(size)])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model_for(kind, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'], eta_min=cfg['minimum_lr'])
    batcher = Batches(cache, size, device)
    nodes = data['points'].shape[1]
    node_order = rng.permutation(nodes)
    offsets = rng.integers(nodes, size=len(data['points']))
    start, best, stale, seconds, history = 0, float('inf'), 0, 0., []
    latest = out / 'latest.pt'
    if latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        if ck['protocol_sha256'] != digest(run / 'protocol.json'):
            raise ValueError('Cannot change the protocol beneath checkpoints')
        if ck['complete']:
            # Recover a kill between checkpoint commit and progress JSON commit.
            write(out / 'progress.json', dict(complete=True, epochs=ck['epoch'] + 1,
                  best_validation_mm=ck['best'], training_seconds=ck['seconds'],
                  parameters=sum(p.numel() for p in model.parameters()), history=ck['history']))
            return True
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        schedule.load_state_dict(ck['scheduler'])
        start, best, stale, seconds, history = ck['epoch'] + 1, ck['best'], ck['stale'], ck['seconds'], ck['history']
        rng.bit_generator.state = ck['rng']
        torch.set_rng_state(ck['torch_rng'])
        node_order, offsets = ck['node_order'], ck['offsets']
        if device.type == 'cuda':
            torch.cuda.set_rng_state(ck['cuda_rng'], device)
    manifest = dict(size=size, kind=kind, seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                    normalizers_sha256=digest(cache / f'normalizers-{size}.npz'),
                    protocol_sha256=digest(run / 'protocol.json'), torch=torch.__version__,
                    device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    initialization='Random, from scratch', selected_training_ids=ids.tolist(),
                    determinism='Paired RNG seeds; GPU indexed reductions may be nondeterministic')
    write(out / 'manifest.json', manifest)
    def validate():
        model.eval()
        errors = []
        with torch.inference_mode():
            for i in range(len(valid['points'])):
                b = batcher(valid, [i])
                errors.append(float((batcher.predict(model, b) - b['targets']).norm(dim=-1).mean().cpu()) * 10)
        return float(np.mean(errors))
    started = time.perf_counter()
    for epoch in range(start, cfg['epochs']):
        if time.time() >= deadline:
            return False
        model.train()
        order = rng.permutation(ids)
        losses = []
        for begin in range(0, len(order), cfg['batch_size']):
            selected = order[begin:begin + cfg['batch_size']]
            q = node_order[(offsets[selected, None] + epoch * cfg['sample_nodes'] + np.arange(cfg['sample_nodes'])) % nodes]
            b = batcher(data, selected, q)
            opt.zero_grad(set_to_none=True)
            loss = ((batcher.predict(model, b) - b['targets']) / batcher.ustd).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        schedule.step()
        row = dict(epoch=epoch, normalized_mse=float(np.mean(losses)))
        if epoch % cfg['validation_every'] == 0 or epoch == cfg['epochs'] - 1:
            value = validate()
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite validation error')
            row['validation_mean_nodal_error_mm'] = value
            if value < best:
                best, stale = value, 0
                save_checkpoint(out / 'best.pt', dict(model=model.state_dict(), epoch=epoch, validation_mm=value))
            else:
                stale += 1
        history.append(row)
        sync(device)
        complete = epoch == cfg['epochs'] - 1 or stale >= cfg['patience_checks']
        elapsed = seconds + time.perf_counter() - started
        ck = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=schedule.state_dict(),
                  epoch=epoch, best=best, stale=stale, seconds=elapsed, history=history,
                  rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), node_order=node_order,
                  offsets=offsets, complete=complete, protocol_sha256=manifest['protocol_sha256'])
        if device.type == 'cuda':
            ck['cuda_rng'] = torch.cuda.get_rng_state(device)
        save_checkpoint(latest, ck)
        write(out / 'progress.json', dict(complete=complete, epochs=epoch + 1, best_validation_mm=best,
              training_seconds=elapsed, parameters=manifest['parameters'], history=history))
        print('TRAIN', size, kind, seed, epoch, best, flush=True)
        if complete:
            return True
    return True


def paired_interval(values, replicates, seed=2027):
    """Resample geometry units; values have already been averaged across seeds."""
    rng = np.random.default_rng(seed)
    samples = [np.mean(values[rng.integers(len(values), size=len(values))]) for _ in range(replicates)]
    return np.quantile(samples, [.025, .975]).tolist()


def evaluate(run, device_name, deadline=float('inf')):
    import torch
    cfg = json.loads((run / 'protocol.json').read_text())
    # Gate access to test labels on completion of every predeclared training task.
    tasks = [(size, kind, seed) for size in cfg['sizes'] for kind in ('gnn', 'point') for seed in cfg['seeds']]
    for size, kind, seed in tasks:
        path = run / 'tasks' / f'{size}-{kind}-{seed}'
        if not json.loads((path / 'progress.json').read_text())['complete']:
            raise RuntimeError('Finish all validation-selected models before test evaluation')
    device = torch.device(device_name)
    torch.set_num_threads(cfg['threads_per_worker'])
    data = load_data(run / 'data', 'test')
    records, results = [], []
    out = run / 'evaluation'
    out.mkdir(exist_ok=True)
    for size in cfg['sizes']:
        by_kind = {}
        for kind in ('gnn', 'point'):
            predictions, seed_errors = [], []
            training_seconds = 0.
            batcher = Batches(run / 'data', size, device)
            for seed in cfg['seeds']:
                if time.time() >= deadline:
                    return False
                task = run / 'tasks' / f'{size}-{kind}-{seed}'
                model = model_for(kind, cfg).to(device)
                ck = torch.load(task / 'best.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(ck['model'])
                model.eval()
                pred = np.empty(data['targets'].shape, dtype='float32')
                with torch.inference_mode():
                    for i in range(len(pred)):
                        if i % 100 == 0 and time.time() >= deadline:
                            return False
                        pred[i] = batcher.predict(model, batcher(data, [i]))[0].cpu().numpy()
                    b = batcher(data, [0])
                    for _ in range(cfg['inference_warmup']):
                        batcher.predict(model, b)
                    sync(device)
                    clock = time.perf_counter()
                    for _ in range(cfg['inference_repetitions']):
                        batcher.predict(model, b)
                    sync(device)
                timing = (time.perf_counter() - clock) / cfg['inference_repetitions']
                distance = np.linalg.norm(pred - data['targets'], axis=-1) * 10
                error = distance.mean(1)
                seed_errors.append(error)
                for i in range(len(error)):
                    records.append(dict(size=size, model=kind, seed=str(seed), geometry=i,
                         mean_nodal_mm=float(error[i]), p95_nodal_mm=float(np.quantile(distance[i], .95)),
                         max_nodal_mm=float(distance[i].max())))
                progress = json.loads((task / 'progress.json').read_text())
                training_seconds += progress['training_seconds']
                write(out / f'{size}-{kind}-{seed}.json', dict(mean_nodal_mm=float(error.mean()),
                      geometry_mean_error_p95_mm=float(np.quantile(error, .95)),
                      worst_geometry_mean_error_mm=float(error.max()),
                      mean_geometry_nodal_p95_mm=float(np.quantile(distance, .95, axis=1).mean()),
                      worst_nodal_error_mm=float(distance.max()),
                      inference_seconds=timing, timing_scope='Device-resident batch=1 full-field forward; excludes transfer',
                      parameters=progress['parameters'], training_seconds=progress['training_seconds'],
                      selected_epoch=ck['epoch'], checkpoint_sha256=digest(task / 'best.pt')))
                predictions.append(pred)
            ensemble = np.mean(predictions, axis=0)
            distance = np.linalg.norm(ensemble - data['targets'], axis=-1) * 10
            for i in range(len(distance)):
                records.append(dict(size=size, model=kind, seed='ensemble', geometry=i,
                     mean_nodal_mm=float(distance[i].mean()), p95_nodal_mm=float(np.quantile(distance[i], .95)),
                     max_nodal_mm=float(distance[i].max())))
            models = []
            for seed in cfg['seeds']:
                member = model_for(kind, cfg).to(device)
                saved = torch.load(run / 'tasks' / f'{size}-{kind}-{seed}' / 'best.pt',
                                   map_location='cpu', weights_only=False)
                member.load_state_dict(saved['model'])
                models.append(member.eval())
            with torch.inference_mode():
                b = batcher(data, [0])
                def ensemble_forward():
                    return torch.stack([batcher.predict(member, b) for member in models]).mean(0)
                for _ in range(cfg['inference_warmup']):
                    ensemble_forward()
                sync(device)
                clock = time.perf_counter()
                for _ in range(cfg['inference_repetitions']):
                    ensemble_forward()
                sync(device)
                ensemble_seconds = (time.perf_counter() - clock) / cfg['inference_repetitions']
            write(out / f'{size}-{kind}-ensemble.json', dict(members=len(models),
                  mean_nodal_mm=float(distance.mean()),
                  geometry_mean_error_p95_mm=float(np.quantile(distance.mean(1), .95)),
                  worst_geometry_mean_error_mm=float(distance.mean(1).max()),
                  mean_geometry_nodal_p95_mm=float(np.quantile(distance, .95, axis=1).mean()),
                  worst_nodal_error_mm=float(distance.max()),
                  inference_seconds=ensemble_seconds, training_seconds=training_seconds,
                  timing_scope='Sequential ensemble members on one GPU, device-resident batch=1; transfers excluded'))
            del models
            by_kind[kind] = dict(single=np.array(seed_errors), ensemble=distance.mean(1))
        for mode in ('single', 'ensemble'):
            g, p = by_kind['gnn'][mode], by_kind['point'][mode]
            difference = (g - p).mean(0) if mode == 'single' else g - p
            results.append(dict(size=size, comparison=mode, gnn_mean_mm=float(g.mean()),
                point_mean_mm=float(p.mean()), gnn_minus_point_mm=float(difference.mean()),
                paired_geometry_bootstrap_95_ci_mm=paired_interval(difference, cfg['bootstrap_replicates']),
                per_seed_mean_mm={kind: by_kind[kind]['single'].mean(1).tolist() for kind in by_kind}))
    with (out / 'per-geometry.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write(out / 'result.json', dict(complete=True, comparisons=results, seeds=cfg['seeds'],
          test_geometries=len(data['targets']), evaluation_role=cfg['evaluation_role'], scope=cfg['scope'],
          bootstrap_scope='Paired geometries, conditional on fixed subsets and seeds; seed variation reported separately',
          ensemble_members=len(cfg['seeds']), protocol_sha256=digest(run / 'protocol.json')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind in ('gnn', 'point'):
        for mode in ('single', 'ensemble'):
            rows = [r for r in results if r['comparison'] == mode]
            ax.plot([r['size'] for r in rows], [r[kind + '_mean_mm'] for r in rows],
                    marker='o', linestyle='-' if mode == 'single' else '--', label=f'{kind}: {mode}')
    ax.set(xscale='log', xlabel='Training geometries', ylabel='Mean nodal error (mm)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'learning-curves.png', dpi=200)
    plt.close(fig)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train', 'evaluate'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--size', type=int)
    parser.add_argument('--kind', choices=('gnn', 'point'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--deadline', type=float, default=float('inf'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.run)
    elif args.action == 'evaluate':
        sys.exit(0 if evaluate(args.run, args.device, args.deadline) else 75)
    else:
        sys.exit(0 if train(args.run, args.size, args.kind, args.seed, args.device, args.deadline) else 75)
