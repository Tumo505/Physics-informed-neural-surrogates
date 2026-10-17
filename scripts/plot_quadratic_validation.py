"""Plot completed one-factor refinement evidence, never provisional curves."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parents[1]
audit_path = P/'experiments/metrics/cardiac_quadratic_reference_dt00125_v1_bernstein.json'
if audit_path.exists():
    audit = json.loads(audit_path.read_text())
    native = json.loads((P/'data/processed'/audit['reference_name']/'manifest.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, state in zip(axes, ['endpoint', 'stress']):
        times = [r['time_s'] for r in audit['records']]
        ax.plot(times, [r[state+'_minimum_Bernstein_lower_bound'] for r in audit['records']],
                label='Whole-element Bernstein lower bound')
        ax.plot(times, [r['minimum_'+state+'_sampled_detF'] for r in native['steps']],
                label='Minimum at source sample points', linestyle='--')
        ax.axhline(0, color='black', linewidth=.8)
        ax.set(xlabel='Time (s)', ylabel='Minimum determinant / lower bound', title=state.capitalize()+' state')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('P2 / 1.25 ms: 800 steps, 17,625 tetrahedra\nSufficient positivity bound in float64; not an interval-arithmetic proof')
    fig.tight_layout()
    fig.savefig(P/'reports/figures/dynamic_quadratic_positivity.png', dpi=180)
    plt.close(fig)
evidence = []
geometry_cases = []
for case in ['g004_l0', 'g005_l1']:
    path = P/f'experiments/metrics/dynamic_quadratic_validation_{case}.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['complete']:
            geometry_cases.append((case, result))
if geometry_cases:
    fig, axes = plt.subplots(1, len(geometry_cases), figsize=(6.4*len(geometry_cases), 4.5), squeeze=False)
    for ax, (case, result) in zip(axes[0], geometry_cases):
        ax.plot([r['time_s'] for r in result['records']],
            [1e3*np.sqrt(r['error_squared_integral_m5']/(3*result['physical_volume_m3'])) for r in result['records']])
        ax.set(xlabel='Time (s)', ylabel='Volume component RMSE (mm)',
            title=f'{case}: {result["relative_space_time_L2"]:.2%} full-cycle L2 difference')
        ax.grid(alpha=.2)
    fig.suptitle('P2 mesh sensitivity on fixed validation geometries\nMatched per-case loads, timestep and solver; two levels only')
    fig.tight_layout()
    fig.savefig(P/'reports/figures/dynamic_quadratic_geometry_controls.png', dpi=180)
    plt.close(fig)
order_path = P/'experiments/metrics/dynamic_quadratic_order_control.json'
if order_path.exists():
    order = json.loads(order_path.read_text())
    if order['complete']:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot([r['time_s'] for r in order['records']],
            [1e3*np.sqrt(r['difference_squared_integral_m5']/(3*order['physical_volume_m3'])) for r in order['records']])
        ax.set(xlabel='Time (s)', ylabel='Volume component RMSE (mm)',
            title=f'Same mesh and 1 ms timestep: P1 versus P2 displacement\nFull-cycle volume L2 difference {order["relative_space_time_volume_L2_difference"]:.2%}; component RMSE {order["component_volume_time_RMSE_m"]*1000:.3f} mm')
        ax.grid(alpha=.2)
        fig.text(.5, .015, 'One known passive baseline case; P2 remains a discrete reference, not continuum truth.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(P/'reports/figures/dynamic_quadratic_order_control.png', dpi=180)
        plt.close(fig)
solver_path = P/'experiments/metrics/dynamic_quadratic_solver_equivalence.json'
if solver_path.exists():
    solver = json.loads(solver_path.read_text())
    if solver['complete']:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.semilogy([r['time_s'] for r in solver['records']],
            [max(r['maximum_absolute_coefficient_difference_m']*1e9, 1e-10) for r in solver['records']],
            label='Maximum absolute displacement-DOF difference')
        ax.axhline(1, color='black', linestyle='--', label='Declared 1 nm tolerance')
        ax.set(xlabel='Time (s)', ylabel='Maximum coefficient difference (nm)',
            title=f'800-step recursive solver control: GMRES/GAMG versus MUMPS\nSame discretization; agreement passed: {solver["passed"]}')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(P/'reports/figures/dynamic_quadratic_solver_equivalence.png', dpi=180)
        plt.close(fig)
for factor in ['time', 'mesh', 'mesh_gamg', 'quadrature']:
    f = P/f'experiments/metrics/dynamic_quadratic_controlled_{factor}.json'
    if f.exists():
        d = json.loads(f.read_text())
        if d['complete']:
            evidence.append((factor, d))
if not evidence:
    raise SystemExit(0 if audit_path.exists() else 'No completed validation evidence to plot')
fig, axes = plt.subplots(1, len(evidence), figsize=(6.4*len(evidence), 4.4), squeeze=False)
for ax, (factor, d) in zip(axes[0], evidence):
    label = dict(time='timestep', mesh='mesh (MUMPS)', mesh_gamg='mesh (GMRES/GAMG)', quadrature='integration degree')[factor]
    records = d['records']
    ax.plot([r['time_s'] for r in records],
            [1e3*np.sqrt(r['error_squared_integral_m5']/(3*d['physical_volume_m3'])) for r in records])
    ax.set(xlabel='Time (s)', ylabel='Volume component RMSE (mm)',
           title=f'Controlled {label} sensitivity\nSpace-time relative L2: {d["relative_space_time_L2"]:.2%}')
    ax.grid(alpha=.25)
fig.suptitle('Passive P2 reference: controlled two-level sensitivity\nFiner level is not continuum truth')
fig.tight_layout()
fig.savefig(P/'reports/figures/dynamic_quadratic_controlled.png', dpi=180)
