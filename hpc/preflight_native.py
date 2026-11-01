"""Read native capabilities before any FE trajectory is attempted."""
import json,platform,sys
import dolfin as df
from petsc4py import PETSc
from mpi4py import MPI
from pathlib import Path
p=PETSc.PC().create();assert hasattr(p,'setReusePreconditioner');p.destroy()
print(json.dumps(dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),dolfin=df.__version__,petsc=PETSc.Sys.getVersion(),mpi=MPI.Get_library_version(),single_rank=MPI.COMM_WORLD.size==1,scope='Capability check only; not mechanics,gradient or cross-platform equivalence validation.'),indent=2))
