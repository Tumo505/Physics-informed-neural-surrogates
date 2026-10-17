"""Complete postprocessing after the independently verified quadratic comparison."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

P = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/physics/dynamic_quadratic_reference.json')
args = ap.parse_args()
cfg = json.loads((P/args.config).read_text())
root = P/'data/processed'/cfg['name']
metric = P/'experiments/metrics/dynamic_quadratic_comparison.json'
while True:
    try:
        data = json.loads(metric.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    if (data.get('complete') and data.get('reference_config') == cfg
            and data.get('native_norm_verification', {}).get('passed')):
        break
    if (root/'failure.json').exists():
        subprocess.run([str(P/'.venv/bin/python'), str(P/'scripts/write_reports.py')],
                       cwd=P, check=True)
        raise RuntimeError('Reference attempt failed; no figures or frozen-model sensitivity claimed')
    time.sleep(30)
env = os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MPLCONFIGDIR=str(P/'.cache/matplotlib'))
for script in ['evaluate_reference_sensitivity.py', 'plot_quadratic_reference.py', 'write_reports.py']:
    subprocess.run([str(P/'.venv/bin/python'), str(P/'scripts'/script)], cwd=P, env=env, check=True)
print('QUADRATIC FOLLOW-UP COMPLETE', flush=True)
