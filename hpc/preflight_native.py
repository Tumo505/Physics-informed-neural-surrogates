"""Check actual native import/CLI paths without generating fields or advancing histories."""
import hashlib,importlib,importlib.util,json,platform,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'src'),str(P/'hpc')]
import dolfin as df
from petsc4py import PETSc
from mpi4py import MPI
import meshio
p=PETSc.PC().create();assert hasattr(p,'setReusePreconditioner');p.destroy()
modules=['lv_surrogate.physics.reference_fem','lv_surrogate.physics.parallel_fem','lv_surrogate.physics.admissible_newton_regularized_forcing','lv_surrogate.physics.threaded_bernstein','platform_native','probe_native','platform_native_chunked','audit_checkpoint']
for name in modules:importlib.import_module(name)
entrypoints=['hpc/platform_native.py','hpc/platform_native_chunked.py','hpc/qualify_audit_resume.py','hpc/probe_native.py','hpc/verify_probe_native.py','scripts/generate_regularized_reference.py','scripts/compare_quadratic_references.py']
checks=[]
for name in entrypoints:
 result=subprocess.run([sys.executable,str(P/name),'--help'],cwd=P,text=True,capture_output=True)
 if result.returncode:raise RuntimeError(name+' import/CLI smoke check failed:\n'+result.stderr+'\n'+result.stdout)
 checks.append(dict(path=name,return_code=0,stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest()))
print(json.dumps(dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),dolfin=df.__version__,meshio=meshio.__version__,petsc=PETSc.Sys.getVersion(),mpi=MPI.Get_library_version(),single_rank=MPI.COMM_WORLD.size==1,import_checks=modules,entrypoint_checks=checks,mesh_generation_requested=False,gmsh_available=importlib.util.find_spec('gmsh') is not None,scope='Imports and command help only. No mesh generation, FE solve, new physical state, numerical equivalence or scientific validation implied.'),indent=2))
