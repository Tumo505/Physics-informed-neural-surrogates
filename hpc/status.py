"""Lightweight gateway-safe status, ledger verification and manuscript refresh."""
import argparse,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'hpc'))
from registry import read
from manuscript import render
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=P/a.run;events=read(run);s=json.loads((run/'state.json').read_text());print(json.dumps(s,indent=2));print('Verified ledger events:',len(events));print('Manuscript:',render(P,run))
