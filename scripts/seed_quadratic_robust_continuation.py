"""Preserve and hash an accepted FE restart prefix in a separate candidate run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

P = Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--config',default='configs/physics/dynamic_quadratic_reference_halfstep_robust_refined.json')
ap.add_argument('--source-reference',default='cardiac_quadratic_reference_halfstep_refined_gamg_v1')
ap.add_argument('--probes',nargs='+',default=['dynamic_quadratic_halfstep_recovery_mumps','dynamic_quadratic_halfstep_direct_repair'])
args=ap.parse_args()
config_path = P/args.config
cfg = json.loads(config_path.read_text())
source = P/'data/processed'/args.source_reference
old = json.loads((source/'manifest.json').read_text())
assert not old['complete'] and (source/'failure.json').exists()
prefix_steps=old['completed_steps']
for k in ['dt_s','duration_s','displacement_degree','refinement_level','geometry_scales','amplitude','timing_scale','quadrature_degree']:
    assert cfg[k] == old['config'][k]
target = P/'data/processed'/cfg['name']
assert not target.exists(), 'Never overwrite a continuation run'
names = args.probes
probes = [json.loads((P/f'experiments/metrics/{n}.json').read_text()) for n in names]
audits = [json.loads((P/f'experiments/metrics/{n}_verification.json').read_text()) for n in names]
assert all(a['passed'] for a in audits)
assert probes[-1]['all_requested_steps_accepted']
for p,a in zip(probes,audits):
    assert hashlib.sha256((P/p['protocol']['fields']).read_bytes()).hexdigest()==a['fields_sha256']
    assert hashlib.sha256((source/'restart.npz').read_bytes()).hexdigest()==p['checkpoint_sha256']
inputs = np.load(source/'inputs.npz')
target.mkdir(parents=True)
shutil.copy2(source/'inputs.npz',target/'inputs.npz')
old_u = np.load(source/'displacement_coefficients.npy',mmap_mode='r')
old_nodes = np.load(source/'displacement_vertices.npy',mmap_mode='r')
u = np.lib.format.open_memmap(target/'displacement_coefficients.npy',mode='w+',dtype=np.float64,shape=old_u.shape)
nodes = np.lib.format.open_memmap(target/'displacement_vertices.npy',mode='w+',dtype=np.float64,shape=old_nodes.shape)
prefix_hash = hashlib.sha256()
for i in range(prefix_steps):
    u[i] = old_u[i]; nodes[i] = old_nodes[i]; prefix_hash.update(np.ascontiguousarray(old_u[i]).tobytes())
vertex_dofs = np.full((len(inputs['coords']),3),-1,dtype=np.int64)
for c in range(3):
    vertex_dofs[inputs['cells'],c] = inputs['canonical_cell_dofs'][:,:4,c]
assert (vertex_dofs>=0).all()
np.testing.assert_allclose(old_u[0][vertex_dofs],old_nodes[0],atol=1e-14,rtol=0)
records = list(old['steps'])
index=prefix_steps
appended_seconds=0.
for p,a in zip(probes,audits):
    fields=np.load(P/p['protocol']['fields'],mmap_mode='r')
    assert p['reference_config_sha256']==old['config_sha256']
    assert p['dt_s']==cfg['dt_s'] and not p['changed_time_discretization']
    for offset,audited in enumerate(a['records']):
        record=p['records'][offset];assert record['accepted']
        field=fields[offset]
        np.testing.assert_allclose(inputs['time'][index],record['time_s'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(audited['time_s'],record['time_s'],atol=1e-12,rtol=0)
        u[index]=field;nodes[index]=field[vertex_dofs]
        records.append(dict(step=index,time_s=record['time_s'],
            iterations=len(record['iteration_history'])-1,iterations_kind='Newton updates from monitor',
            SNES_reason=record['SNES_reason'],residual_l2_N=record['residual_N'],
            minimum_endpoint_sampled_detF=record['minimum_endpoint_sampled_detF'],
            minimum_stress_sampled_detF=record['minimum_stress_sampled_detF'],
            nonlinear_solve_seconds=record['nonlinear_seconds'],inherited_verified_probe_state=True))
        appended_seconds+=record['nonlinear_seconds'];index+=1
u.flush(); nodes.flush()
restart_path = P/probes[-1]['protocol']['checkpoint']
assert hashlib.sha256(restart_path.read_bytes()).hexdigest()==probes[-1]['probe_checkpoint_sha256']
restart = np.load(restart_path)
np.testing.assert_allclose(restart['u'],u[index-1],atol=0,rtol=0)
np.savez_compressed(target/'restart.npz',completed_steps=index,
    **{n:restart[n] for n in ['u','u_old','v_old','a_old']})
result = dict(old)
result.pop('output_sha256',None)
result.update(config=cfg,config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    complete=False,completed_steps=index,steps=records,
    measured_wall_seconds=old['measured_wall_seconds']+appended_seconds,
    inherited_prefix=dict(original_reference=old['config']['name'],original_config_sha256=old['config_sha256'],
        original_accepted_steps=prefix_steps,original_prefix_coefficient_sha256=prefix_hash.hexdigest(),
        original_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        earlier_inherited_prefix=old.get('inherited_prefix'),
        appended_audits=[str(P/f'experiments/metrics/{n}_verification.json') for n in names],
        seeding_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Inherited accepted states, not re-executed under the new fallback policy. Failed original remains intact. Candidate continuation is unqualified until full-cycle generation and audits finish.'),
    timing_scope='Inherited accepted-prefix recorded wall plus appended direct solve times and future continuation wall; excludes discarded attempts and is not an isolated runtime measurement.')
(target/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print('SEEDED SEPARATE CANDIDATE',cfg['name'],index,'accepted states; original untouched',flush=True)
