"""Isolate native solver failure without modifying the pinned reference source."""
import os,sys,runpy
from pathlib import Path
P=Path(__file__).resolve().parents[1]
os.environ['DIJITSO_CACHE_DIR']=str(P/'.fenics-cache-activated')
def trace(frame,event,arg):
 f=frame.f_code.co_filename
 if event=='line' and ('microstructure.py' in f or 'solve_cardiac_probe.py' in f or '/cardiac_benchmark/problem.py' in f or '/dolfin/function/' in f):
  print('TRACE',f.rsplit('/',1)[-1],frame.f_lineno,flush=True)
 return trace
sys.settrace(trace)
runpy.run_path(str(P/'scripts/solve_cardiac_probe.py'),run_name='__main__')
