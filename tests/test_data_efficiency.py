"""Scientific split control, real training recovery and paired inference tests."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import data_efficiency as study


def fixture_run(path, epochs=4):
    path.mkdir()
    cfg = json.loads((ROOT / 'configs/gnn/data_efficiency.json').read_text())
    cfg.update(sizes=[4], seeds=[11], epochs=epochs, validation_every=1,
               width=8, steps=1, sample_nodes=3, batch_size=2, threads_per_worker=1,
               inference_repetitions=2, inference_warmup=1, bootstrap_replicates=100)
    study.write(path / 'protocol.json', cfg)
    cache = path / 'data'
    cache.mkdir()
    rng = np.random.default_rng(20)
    for split, count in [('train', 6), ('validation', 2), ('test', 3)]:
        for name, shape in dict(points=(6, 13), coarse=(3, 13), globals=(4,), edges=(4, 4), targets=(6, 3)).items():
            np.save(cache / f'{split}_{name}.npy', (rng.normal(size=(count, *shape)) * .01).astype('float32'))
    np.save(cache / 'topology.npy', np.array([[0, 1], [1, 0], [1, 2], [2, 1]]))
    np.save(cache / 'parents.npy', np.array([0, 1, 1, 2, 2, 0]))
    ids = [0, 2, 3, 5]
    study.write(cache / 'ready.json', dict(subsets={'4': ids}))
    np.savez(cache / 'normalizers-4.npz', **study.fit_stats(study.load_data(cache, 'train'), ids))
    return cfg


def test_nested_paired_subsets():
    selected = study.subsets(2250, [64, 256, 1024, 2250], 2027)
    assert selected['64'] == selected['256'][:64]
    assert selected['256'] == selected['1024'][:256]
    assert selected['1024'] == selected['2250'][:1024]
    assert sorted(selected['2250']) == list(range(2250))
    assert selected == study.subsets(2250, [64, 256, 1024, 2250], 2027)
    with pytest.raises(ValueError):
        study.subsets(20, [21], 1)


def test_normalization_cannot_see_unselected_labels():
    array = np.arange(24, dtype='float32').reshape(4, 2, 3)
    before = study.fit_stats({'targets': array}, [0, 1])
    array[2:] = 1e20
    after = study.fit_stats({'targets': array}, [0, 1])
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    np.testing.assert_allclose(before['targets_mean'], np.arange(12).reshape(4, 3).mean(0))


def test_capacity_control_and_no_shape_coefficients():
    cfg = json.loads((ROOT / 'configs/gnn/data_efficiency.json').read_text())
    graph, point = (study.model_for(kind, cfg) for kind in ('gnn', 'point'))
    g, p = (sum(v.numel() for v in model.parameters()) for model in (graph, point))
    assert abs(g - p) / g < .02
    assert graph.node_encoder[0].in_features == 17
    assert point.decoder[0].in_features == 17


def test_exact_epoch_resume_and_progress_recovery(tmp_path, monkeypatch):
    one, resumed = tmp_path / 'one', tmp_path / 'resumed'
    fixture_run(one)
    fixture_run(resumed)
    # Training remains possible with all test labels physically absent.
    for path in (one / 'data').glob('test_*.npy'):
        path.unlink()
    assert study.train(one, 4, 'gnn', 11, 'cpu')
    with monkeypatch.context() as context:
        ticks = iter([0., 10.])
        context.setattr(study.time, 'time', lambda: next(ticks))
        assert not study.train(resumed, 4, 'gnn', 11, 'cpu', deadline=5.)
    assert study.train(resumed, 4, 'gnn', 11, 'cpu')
    a = torch.load(one / 'tasks/4-gnn-11/latest.pt', weights_only=False)
    b = torch.load(resumed / 'tasks/4-gnn-11/latest.pt', weights_only=False)
    assert a['history'] == b['history']
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
    (resumed / 'tasks/4-gnn-11/progress.json').unlink()
    assert study.train(resumed, 4, 'gnn', 11, 'cpu')
    assert json.loads((resumed / 'tasks/4-gnn-11/progress.json').read_text())['complete']


def test_complete_evaluation_and_gate(tmp_path):
    run = tmp_path / 'study'
    fixture_run(run, epochs=1)
    with pytest.raises(FileNotFoundError):
        study.evaluate(run, 'cpu')
    for kind in ('gnn', 'point'):
        assert study.train(run, 4, kind, 11, 'cpu')
    study.evaluate(run, 'cpu')
    result = json.loads((run / 'evaluation/result.json').read_text())
    assert result['test_geometries'] == 3
    assert result['ensemble_members'] == 1
    assert len(result['comparisons']) == 2
    assert 'repeated benchmark' in result['evaluation_role']
    import csv
    with (run / 'evaluation/per-geometry.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert all(float(r['max_nodal_mm']) >= float(r['mean_nodal_mm']) for r in rows)


def test_paired_geometry_resampling():
    assert study.paired_interval(np.full(8, -2.), 100) == [-2., -2.]
    interval = study.paired_interval(np.array([-10., 0., 10.]), 1000)
    assert interval[0] < 0 < interval[1]


def test_pbs_full_resources_and_immutable_protocol(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('learning_launcher', ROOT / 'hpc/launch_data_efficiency.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    run = tmp_path / 'production'
    launcher.initialize(run, Path(sys.executable))
    calls = []
    def qsub(args, **kwargs):
        calls.append(args)
        return type('Reply', (), {'stdout': '123.pbs\n'})()
    monkeypatch.setattr(launcher.subprocess, 'run', qsub)
    launcher.submit(run)
    assert 'walltime=168:00:00' in calls[0]
    assert 'select=1:ncpus=28:mem=240gb:ngpus=4' in calls[0]
    plan = json.loads((run / 'plan.json').read_text())
    assert len(plan['tasks']) == 24
    assert len(set(tuple(t) for t in plan['tasks'])) == 24
    launcher.verify_sources(run)
    (run / 'protocol.json').write_text('{}')
    with pytest.raises(RuntimeError, match='protocol changed'):
        launcher.verify_sources(run)


@pytest.mark.parametrize('failure', [False, True])
def test_only_walltime_pause_queues_successor(tmp_path, monkeypatch, failure):
    spec = importlib.util.spec_from_file_location('scheduler_test', ROOT / 'hpc/launch_data_efficiency.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    run = tmp_path / 'scheduler'
    launcher.initialize(run, Path(sys.executable))
    state = json.loads((run / 'state.json').read_text())
    state.update(slice=1, job='122.pbs')
    study.write(run / 'state.json', state)
    study.write(run / 'data/ready.json', {'prepared_sha256': {}})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *args, **kwargs: 'frozen-environment\n')
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        if failure:
            raise RuntimeError('Preparation failed')
        return type('Reply', (), {'stdout': '123.pbs\n'})()
    monkeypatch.setattr(launcher.subprocess, 'run', command)
    class PausedWorker:
        returncode = 75
        def poll(self):
            return 75
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *args, **kwargs: PausedWorker())
    if failure:
        with pytest.raises(RuntimeError, match='Preparation failed'):
            launcher.execute(run)
        assert json.loads((run / 'state.json').read_text())['status'] == 'failed'
        assert not any(args[0] == 'qsub' for args in calls)
    else:
        launcher.execute(run)
        assert json.loads((run / 'state.json').read_text())['status'] == 'submitted'
        assert json.loads((run / 'state.json').read_text())['slice'] == 2
        assert sum(args[0] == 'qsub' for args in calls) == 1
