"""Freeze a stopped laptop trajectory for a separately qualified cluster handoff."""
import argparse,fcntl,json,sys,mmap,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'hpc'))
from registry import atomic,sha
ap=argparse.ArgumentParser();ap.add_argument('--output',default='experiments/manifests/hpc_migration_snapshot.json');a=ap.parse_args();out=P/a.output;assert not out.exists(),'Preserve existing migration snapshot; use a new output identifier'
pp=P/'configs/physics/dynamic_regularized_stream_third_20261101_v3.json';p=json.loads(pp.read_text());cfg=json.loads((P/p['config']).read_text());root=P/'data/processed'/cfg['name']
with (root/'generation.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=json.loads((root/'manifest.json').read_text());cp=np.load(root/'restart.npz');assert not(root/'failure.json').exists() and int(cp['completed_steps'])==m['completed_steps'];count=m['completed_steps']
 for f,h in p['frozen_source_sha256'].items():assert sha(P/f)==h
 fields=np.load(root/'displacement_coefficients.npy',mmap_mode='r');digest=hashlib.sha256()
 for i in range(count):
  digest.update(np.asarray(fields[i]).tobytes())
  if (i+1)%8==0 and hasattr(fields._mmap,'madvise'):fields._mmap.madvise(mmap.MADV_DONTNEED)
 pins={f:sha(P/f) for f in ['manifest.json','restart.npz','positivity.json','inputs.npz'] for f in [str((root/f).relative_to(P))]}
 snapshot=dict(complete=True,reference=cfg['name'],source_reference=p['source_reference'],source_stream_protocol=str(pp.relative_to(P)),source_config=p['config'],accepted_steps=count,n_steps=m['n_steps'],last_time_s=m['steps'][-1]['time_s'],prefix_fields_sha256=digest.hexdigest(),files=pins,numerical_sources=p['frozen_source_sha256'],native_platform='original macOS; Linux qualification still required',no_active_generation_lock=True,scope='Stopped accepted chain only. No new scientific qualification implied; strict cluster force/norm/history/maps/provenance and known-field reproduction must pass before continuation. Preserve original histories and failure records.')
 atomic(out,snapshot);print(out.relative_to(P),count,'accepted states')
