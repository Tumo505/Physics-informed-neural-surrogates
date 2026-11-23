"""Consolidate completed learning curves without training or model inference.

Full mode reads saved per-geometry errors and measured timings on the HPC.
Summary mode makes accuracy-only figures from a supplied result.json; it never
substitutes estimates for unavailable timing or tail measurements.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from data_efficiency import digest, write

COLORS = {'gnn': '#0072B2', 'point': '#D55E00'}
LABELS = {'gnn': 'Graph', 'point': 'Pointwise'}
LIMITS = (
    'Repeated benchmark evaluation: this test cohort was inspected in previous work. '
    'Results concern approximation of supplied discrete FE displacement labels on '
    'the supplied geometry distribution; no clinical, continuum-convergence or '
    'physics-informed claim. Geometry-bootstrap intervals are conditional on the '
    'fixed training subsets and three model seeds. They do not capture training-subset '
    'uncertainty or represent population-level anatomical validation. Intervals are '
    'pointwise 95% intervals without multiple-comparison adjustment.'
)


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def accuracy_rows(result):
    if result.get('complete') is not True:
        raise ValueError('The evaluation is incomplete')
    seeds = result['seeds']
    if len(seeds) != 3 or len(set(seeds)) != len(seeds):
        raise ValueError('This report requires the three distinct predeclared paired seeds')
    rows, seen = [], set()
    for value in result['comparisons']:
        key = value['size'], value['comparison']
        if key in seen or key[1] not in ('single', 'ensemble'):
            raise ValueError(f'Duplicate or invalid comparison: {key}')
        seen.add(key)
        g, p = float(value['gnn_mean_mm']), float(value['point_mean_mm'])
        lo, hi = value['paired_geometry_bootstrap_95_ci_mm']
        if not np.isfinite([g, p, lo, hi]).all() or min(g, p) < 0 or lo > hi or p == 0:
            raise ValueError('Invalid accuracy or interval')
        if not np.isclose(g - p, value['gnn_minus_point_mm'], atol=2e-6, rtol=1e-5):
            raise ValueError('Paired difference disagrees with the reported means')
        row = dict(training_geometries=key[0], comparison=key[1],
                   graph_mean_mm=g, pointwise_mean_mm=p,
                   graph_seed_sd_mm='', pointwise_seed_sd_mm='',
                   graph_minus_pointwise_mm=float(value['gnn_minus_point_mm']),
                   paired_geometry_ci95_low_mm=float(lo), paired_geometry_ci95_high_mm=float(hi),
                   graph_relative_error_reduction_pct=100 * (p - g) / p)
        for kind, name in [('gnn', 'graph'), ('point', 'pointwise')]:
            values = np.asarray(value['per_seed_mean_mm'][kind])
            if len(values) != len(seeds) or not np.isfinite(values).all():
                raise ValueError('Missing or invalid seed results')
            if key[1] == 'single':
                if not np.isclose(values.mean(), g if kind == 'gnn' else p, atol=2e-6):
                    raise ValueError('Seed means disagree with the aggregate')
                row[name + '_seed_sd_mm'] = float(values.std(ddof=1))
        rows.append(row)
    sizes = sorted({r['training_geometries'] for r in rows})
    if seen != {(size, mode) for size in sizes for mode in ('single', 'ensemble')}:
        raise ValueError('Missing single-model or ensemble comparison')
    if result['ensemble_members'] != len(seeds):
        raise ValueError('Unequal ensemble and seed counts')
    return sorted(rows, key=lambda r: (r['training_geometries'], r['comparison'] == 'ensemble'))


def detail_rows(run, result, accuracy):
    """Validate rectangular geometry/seed coverage before aggregating tails."""
    sources = [run / 'evaluation/per-geometry.csv', run / 'protocol.json', run / 'state.json']
    protocol = read(run / 'protocol.json')
    if read(run / 'state.json')['status'] != 'complete':
        raise ValueError('The run has not completed evaluation')
    if digest(run / 'protocol.json') != result['protocol_sha256']:
        raise ValueError('Protocol hash does not match the evaluated protocol')
    sizes = sorted({r['training_geometries'] for r in accuracy})
    if sizes != sorted(protocol['sizes']) or result['seeds'] != protocol['seeds']:
        raise ValueError('Evaluation does not cover the frozen protocol')
    groups = {}
    expected_ids = set(range(result['test_geometries']))
    with (run / 'evaluation/per-geometry.csv').open() as stream:
        for row in csv.DictReader(stream):
            key = int(row['size']), row['model'], row['seed']
            geometry = int(row['geometry'])
            group = groups.setdefault(key, {})
            if geometry in group:
                raise ValueError(f'Duplicate geometry row: {key}, {geometry}')
            values = [float(row[name]) for name in ('mean_nodal_mm', 'p95_nodal_mm', 'max_nodal_mm')]
            if not np.isfinite(values).all() or min(values) < 0 or values[2] + 1e-6 < max(values[:2]):
                raise ValueError('Invalid per-geometry errors')
            group[geometry] = values
    expected = {(size, kind, seed) for size in sizes for kind in ('gnn', 'point')
                for seed in [*map(str, result['seeds']), 'ensemble']}
    if set(groups) != expected or any(set(group) != expected_ids for group in groups.values()):
        raise ValueError('Missing or unexpected geometry, seed or architecture rows')
    arrays = {key: np.asarray([group[i] for i in sorted(expected_ids)]) for key, group in groups.items()}
    rows = []
    for size in sizes:
        for kind in ('gnn', 'point'):
            records = []
            for seed in result['seeds']:
                path = run / 'evaluation' / f'{size}-{kind}-{seed}.json'
                sources.append(path)
                records.append(read(path))
            if len({r['parameters'] for r in records}) != 1:
                raise ValueError('Architecture parameter count changes across seeds')
            for mode in ('single', 'ensemble'):
                if mode == 'single':
                    values = np.stack([arrays[size, kind, str(seed)] for seed in result['seeds']])
                    metrics = records
                    train_seconds = float(np.mean([r['training_seconds'] for r in metrics]))
                    parameters = records[0]['parameters']
                else:
                    values = arrays[size, kind, 'ensemble'][None]
                    path = run / 'evaluation' / f'{size}-{kind}-ensemble.json'
                    sources.append(path)
                    metrics = [read(path)]
                    if metrics[0]['members'] != len(result['seeds']):
                        raise ValueError('Ensemble member mismatch')
                    train_seconds = metrics[0]['training_seconds']
                    if not np.isclose(train_seconds, sum(r['training_seconds'] for r in records), atol=1e-3):
                        raise ValueError('Ensemble training cost differs from member totals')
                    parameters = sum(r['parameters'] for r in records)
                for index, metric in enumerate(metrics):
                    if not np.isclose(metric['mean_nodal_mm'], values[index, :, 0].mean(), atol=2e-6):
                        raise ValueError('Timing file accuracy differs from per-geometry data')
                    if not np.isfinite([metric['inference_seconds'], metric['training_seconds']]).all() or min(
                            metric['inference_seconds'], metric['training_seconds']) <= 0:
                        raise ValueError('Invalid recorded timing')
                    expected_tails = dict(
                        geometry_mean_error_p95_mm=np.quantile(values[index, :, 0], .95),
                        worst_geometry_mean_error_mm=values[index, :, 0].max(),
                        mean_geometry_nodal_p95_mm=values[index, :, 1].mean(),
                        worst_nodal_error_mm=values[index, :, 2].max())
                    for name, expected_value in expected_tails.items():
                        if name in metric and not np.isclose(metric[name], expected_value, atol=2e-6):
                            raise ValueError(f'Saved tail statistic disagrees with geometry data: {name}')
                reference = next(r for r in accuracy if r['training_geometries'] == size and r['comparison'] == mode)
                if not np.isclose(values[:, :, 0].mean(), reference[('graph' if kind == 'gnn' else 'pointwise') + '_mean_mm'], atol=2e-6):
                    raise ValueError('Per-geometry results disagree with the summary')
                rows.append(dict(training_geometries=size, model=kind, comparison=mode,
                    mean_nodal_mm=float(values[:, :, 0].mean()),
                    mean_seed_geometry_p95_error_mm=float(np.quantile(values[:, :, 0], .95, axis=1).mean()),
                    mean_seed_worst_geometry_error_mm=float(values[:, :, 0].max(1).mean()),
                    mean_geometry_nodal_p95_mm=float(values[:, :, 1].mean()),
                    worst_nodal_error_all_members_mm=float(values[:, :, 2].max()),
                    training_worker_minutes=train_seconds / 60,
                    inference_ms=float(np.mean([r['inference_seconds'] for r in metrics])) * 1000,
                    parameters=parameters,
                    member_count=1 if mode == 'single' else len(result['seeds'])))
    # Keep raw measurements, including the original timing scope, in the report.
    measurements = {str(p.relative_to(run)): read(p) for p in sources if p.parent == run / 'evaluation' and p.suffix == '.json'}
    return rows, sources, measurements


def accuracy_tex(rows):
    lines = [r'\begin{tabular}{rlrrr}', r'\hline',
             r'$N$ & Evaluation & Graph (mm) & Pointwise (mm) & $\Delta$ [95\% CI] (mm) \\', r'\hline']
    for row in rows:
        cells = []
        for kind in ('graph', 'pointwise'):
            value = f"{row[kind + '_mean_mm']:.3f}"
            if row['comparison'] == 'single':
                value += r' $\pm$ ' + f"{row[kind + '_seed_sd_mm']:.3f}"
            cells.append(value)
        lines.append(f"{row['training_geometries']} & {row['comparison']} & " + ' & '.join(cells) +
                     f" & {row['graph_minus_pointwise_mm']:.3f} [{row['paired_geometry_ci95_low_mm']:.3f}, {row['paired_geometry_ci95_high_mm']:.3f}] " + r'\\')
    return '\n'.join(lines + [r'\hline', r'\end{tabular}',
        '% Single: mean +/- sample SD across seeds; ensemble: three-member prediction average.',
        '% Delta = Graph - Pointwise; negative values favor Graph.', '% ' + LIMITS]) + '\n'


def save_figure(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=300, facecolor='white')
    svg = output / (name + '.svg')
    fig.savefig(svg, facecolor='white', metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


def figures(output, result, accuracy, details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'cardiac-data-efficiency', 'axes.labelsize': 10})
    sizes = sorted({r['training_geometries'] for r in accuracy})
    def axis_format(ax):
        ax.set_xscale('log')
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel('Training geometries')
        ax.grid(axis='y', alpha=.2)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True, sharey=True)
    for ax, mode, title in zip(axes, ('single', 'ensemble'), ('Individual models', 'Three-model ensembles')):
        data = [r for r in accuracy if r['comparison'] == mode]
        for kind, name in [('gnn', 'graph'), ('point', 'pointwise')]:
            values = [r[name + '_mean_mm'] for r in data]
            errors = [r[name + '_seed_sd_mm'] for r in data] if mode == 'single' else None
            ax.errorbar(sizes, values, yerr=errors, marker='o', capsize=3, color=COLORS[kind], label=LABELS[kind])
        ax.set_title(title)
        axis_format(ax)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Mean nodal displacement error (mm)')
    fig.suptitle(f"Repeated benchmark evaluation: {result['test_geometries']} geometries", fontsize=11)
    save_figure(fig, output, '01-learning-curves')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), constrained_layout=True, sharex=True, sharey=True)
    for ax, mode, title in zip(axes, ('single', 'ensemble'), ('Individual models', 'Three-model ensembles')):
        data = [r for r in accuracy if r['comparison'] == mode]
        for i, row in enumerate(data):
            mid = row['graph_minus_pointwise_mm']
            ax.errorbar(mid, i, xerr=[[mid - row['paired_geometry_ci95_low_mm']],
                                     [row['paired_geometry_ci95_high_mm'] - mid]],
                        fmt='o', color='#333333', capsize=4)
        ax.axvline(0, color='#888888', linestyle='--', linewidth=1)
        ax.set_yticks(range(len(sizes)), labels=list(map(str, sizes)))
        ax.set_title(title)
        ax.set_xlabel('Graph − Pointwise error (mm)')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_ylabel('Training geometries')
    axes[0].invert_yaxis()
    fig.suptitle('Paired geometry-bootstrap 95% intervals (conditional)', fontsize=11)
    save_figure(fig, output, '02-paired-differences')
    plt.close(fig)
    if not details:
        return
    for filename, metrics in [
        ('03-tail-errors', [('mean_seed_geometry_p95_error_mm', '95th-percentile geometry error (mm)'),
                            ('worst_nodal_error_all_members_mm', 'Worst nodal error (mm)')]),
        ('04-computational-cost', [('training_worker_minutes', 'Training worker-minutes'),
                                   ('inference_ms', 'Full-field inference (ms)')])]:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
        for col, mode in enumerate(('single', 'ensemble')):
            for row_index, (metric, label) in enumerate(metrics):
                ax = axes[row_index, col]
                for kind in ('gnn', 'point'):
                    data = [r for r in details if r['comparison'] == mode and r['model'] == kind]
                    ax.plot([r['training_geometries'] for r in data], [r[metric] for r in data],
                            '-o', color=COLORS[kind], label=LABELS[kind])
                axis_format(ax)
                ax.set_ylabel(label)
                if row_index == 0:
                    ax.set_title('Individual models' if mode == 'single' else 'Three-model ensembles')
                    ax.legend(frameon=False)
        save_figure(fig, output, filename)
        plt.close(fig)


def build(output, run=None, summary=None):
    result_path = run / 'evaluation/result.json' if run else summary
    result = read(result_path)
    accuracy = accuracy_rows(result)
    details, sources, measurements = [], [result_path], {}
    if run:
        details, more, measurements = detail_rows(run, result, accuracy)
        sources += more
    metadata = []
    if run:
        metadata = sorted(run.glob('gpu-*.json')) + sorted(run.glob('hardware-*.log'))
        if (run / 'python-environment.txt').exists():
            metadata.append(run / 'python-environment.txt')
        sources += metadata
    # Validate everything before creating a potentially shareable report.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(result_path, output / 'evaluation-summary.json')
    csv_write(output / 'accuracy-table.csv', accuracy)
    (output / 'accuracy-table.tex').write_text(accuracy_tex(accuracy))
    if details:
        shutil.copyfile(run / 'evaluation/per-geometry.csv', output / 'per-geometry.csv')
        shutil.copyfile(run / 'protocol.json', output / 'protocol.json')
        (output / 'runtime-provenance').mkdir()
        for path in metadata:
            shutil.copyfile(path, output / 'runtime-provenance' / path.name)
        csv_write(output / 'cost-and-tail-table.csv', details)
        write(output / 'timing-measurements.json', measurements)
        lines = [r'\begin{tabular}{rllrrrrr}', r'\hline',
          r'$N$ & Model & Evaluation & Geom. P95 & Worst node & Train (min) & Infer (ms) & Params \\', r'\hline']
        for r in details:
            lines.append(f"{r['training_geometries']} & {LABELS[r['model']]} & {r['comparison']} & "
                f"{r['mean_seed_geometry_p95_error_mm']:.3f} & {r['worst_nodal_error_all_members_mm']:.3f} & "
                f"{r['training_worker_minutes']:.2f} & {r['inference_ms']:.3f} & {r['parameters']} " + r'\\')
        (output / 'cost-and-tail-table.tex').write_text('\n'.join(lines + [r'\hline', r'\end{tabular}']) + '\n')
    manifest = dict(status='complete' if run else 'accuracy_only',
          accuracy=accuracy, cost_and_tail=details,
          source_sha256={str(p): digest(p) for p in sources},
          reporting_source_sha256=digest(Path(__file__)),
          result_protocol_sha256=result['protocol_sha256'], test_geometries=result['test_geometries'],
          runtime_metadata_files=[p.name for p in metadata],
          model_seeds=result['seeds'], limitations=LIMITS,
          missing_measurements=[] if run else ['Measured training and inference times', 'Per-geometry tail errors'],
          total_training_worker_hours=sum(r['training_worker_minutes'] for r in details if r['comparison'] == 'ensemble') / 60 if run else None)
    captions = [
        '01-learning-curves: Mean nodal Euclidean displacement error on all test geometries. '
        'Left: mean across three individual model seeds; error bars are sample SD across '
        'the three seed-level means, not confidence intervals. Right: arithmetic mean '
        'of three full-field predictions per architecture. All distances are in mm.',
        '02-paired-differences: Graph minus pointwise mean nodal error; negative favors '
        'Graph. Intervals are the saved paired geometry-bootstrap 95% intervals. '
        'For individual models, seed errors are averaged within each geometry before '
        'resampling. These intervals are conditional and not simultaneous.',
        '03-tail-errors (full report): Top: per-seed 95th percentile across geometry-level '
        'mean nodal errors, then averaged across seeds. Bottom: maximum nodal error '
        'across every geometry and (for individual models) every seed. Ensemble errors '
        'are computed from the averaged predictions, not averaged member errors.',
        '04-computational-cost (full report): Single-model training and inference costs '
        'are averaged across three seeds. Ensemble training cost is the sum across '
        'members; inference is the measured sequential three-member forward on one '
        'GPU. Training is elapsed worker time including validation and checkpoint I/O, '
        'excluding setup/preprocessing; it is not utilization-weighted GPU time or '
        'total allocation walltime. Inference is warmed, synchronized, device-resident, '
        'batch-one full-field forward, excluding input transfer and preprocessing. '
        'Timings are descriptive saved means, not latency confidence intervals.',
        'Accuracy table: Single entries mean +/- sample SD across seed means; no seed '
        'SD applies to the one equal-size ensemble. Cost/tail table: single-model '
        'Geom. P95 and worst-geometry mean are averaged over per-seed statistics; '
        'worst-node is the maximum over members. Mean geometry nodal P95, recorded '
        'separately in CSV, describes within-geometry node errors and is a different '
        'statistic from Geom. P95. Parameters are per model or summed for the ensemble.',
        LIMITS]
    (output / 'captions-and-limitations.txt').write_text('\n\n'.join(captions) + '\n')
    notes = []
    for row in accuracy:
        notes.append(f"N={row['training_geometries']}, {row['comparison']}: Graph "
             f"{row['graph_mean_mm']:.3f} mm, Pointwise {row['pointwise_mean_mm']:.3f} mm; "
             f"relative error reduction {row['graph_relative_error_reduction_pct']:.1f}%; "
             f"paired difference {row['graph_minus_pointwise_mm']:.3f} mm "
             f"[95% conditional interval {row['paired_geometry_ci95_low_mm']:.3f}, "
             f"{row['paired_geometry_ci95_high_mm']:.3f}].")
    (output / 'results-text.txt').write_text('\n'.join(notes) + '\n\n' + LIMITS + '\n')
    figures(output, result, accuracy, details)
    write(output / 'report.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=Path, help='Completed HPC run directory (full report)')
    source.add_argument('--summary', type=Path, help='Reported result.json (accuracy only)')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing reports are preserved')
    args = parser.parse_args()
    report = build(args.output.resolve(), run=args.run.resolve() if args.run else None,
                   summary=args.summary.resolve() if args.summary else None)
    print(json.dumps(dict(status=report['status'], output=str(args.output.resolve()),
                         missing_measurements=report['missing_measurements']), indent=2))
