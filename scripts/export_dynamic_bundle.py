"""Export exact frozen models/preprocessing, without publishing or retraining."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile

P = Path(__file__).resolve().parents[1]
freeze = json.loads((P/'experiments/manifests/dynamic_final_freeze.json').read_text())
training = json.loads((P/'configs/physics/dynamic_training.json').read_text())
paths = set(freeze['checkpoint_sha256'])
for relative, digest in freeze['checkpoint_sha256'].items():
    assert hashlib.sha256((P/relative).read_bytes()).hexdigest() == digest
for regime in training['regimes']:
    name = training['name']+'_'+regime
    manifest = f'experiments/manifests/{name}.json'
    metadata = json.loads((P/manifest).read_text())
    normalizers = f'data/processed/{training["data_tag"]}_{regime}/normalizers.npz'
    assert hashlib.sha256((P/normalizers).read_bytes()).hexdigest() == metadata['normalizer_sha256']
    paths.update([manifest, normalizers])
paths.update(str(f.relative_to(P)) for f in (P/'src/lv_surrogate/models').glob('*.py'))
paths.update(['data/processed/cardiac_geometry_ml/cluster_labels.npy',
              'LICENSES_AND_ATTRIBUTION.md',
              'data/processed/cardiac_geometry_ml/topology.npy',
              'configs/physics/dynamic_training.json', 'configs/physics/dynamic_study.json',
              'configs/physics/dynamic_confirmation.json',
              'experiments/manifests/dynamic_final_freeze.json',
              'experiments/manifests/dynamic_confirmation_freeze.json',
              'experiments/metrics/dynamic_final_evaluation.json',
              'experiments/metrics/dynamic_confirmation_evaluation.json'])
content_hashes = {}
for relative in sorted(paths):
    file = P/relative
    assert file.resolve().is_relative_to(P.resolve()) and file.is_file()
    content_hashes[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
inner = json.dumps(dict(sha256=content_hashes,
    scope='Exact selected20 checkpoint files, preprocessing, source, configs and frozen evaluation. '
          'No FE corpus or Python/native environment. Use the repository and locked setup for replay.'),
    sort_keys=True, indent=2).encode()
out = P/'experiments/checkpoints/dynamic_frozen_bundle.tar.gz'
temporary = out.with_suffix('.tmp')
with temporary.open('wb') as raw:
    with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for relative in sorted(paths | {'bundle_manifest.json'}):
                value = inner if relative=='bundle_manifest.json' else (P/relative).read_bytes()
                header = tarfile.TarInfo(relative)
                header.size = len(value); header.mtime = 0; header.mode = 0o644
                archive.addfile(header, io.BytesIO(value))
temporary.replace(out)
with tarfile.open(out, 'r:gz') as archive:
    for relative, digest in content_hashes.items():
        assert hashlib.sha256(archive.extractfile(relative).read()).hexdigest() == digest
result = dict(complete=True, archive=str(out.relative_to(P)), archive_bytes=out.stat().st_size,
    archive_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), members=content_hashes,
    selected_checkpoint_count=len(freeze['checkpoint_sha256']),
    roundtrip_archive_member_checksums_passed=True,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Local exact frozen-artifact bundle; not uploaded or committed as binary weights. '
          'Tar/gzip metadata fixed for deterministic export. FE data/environment are separate.')
(P/'experiments/manifests/dynamic_replay_bundle.json').write_text(json.dumps(result, indent=2))
print('FROZEN BUNDLE VERIFIED', result['selected_checkpoint_count'], result['archive_bytes'],
      result['archive_sha256'], flush=True)
