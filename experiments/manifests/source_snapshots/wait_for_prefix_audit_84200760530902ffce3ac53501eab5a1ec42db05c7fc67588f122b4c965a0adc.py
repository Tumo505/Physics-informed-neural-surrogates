"""Wait only for the already running bounded audit; never launch another copy."""
import json,time
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'experiments/metrics/dynamic_quadratic_multistep_window_v1_prefix_audit_workflow.json'
while True:
 x=json.loads(p.read_text())
 if x['complete']:
  assert x.get('accepted_prefix_independently_verified') and not x.get('failure');break
 time.sleep(2)
