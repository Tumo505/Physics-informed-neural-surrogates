from pathlib import Path
import os
P=Path(__file__).resolve().parents[1];os.environ['DIJITSO_CACHE_DIR']=str(P/'.fenics-cache-activated')
import dolfin as d
import numpy as np
print('dolfin',d.__version__,flush=True)
m=d.UnitCubeMesh(2,2,2);V=d.VectorFunctionSpace(m,'CG',1);u=d.Function(V);w=d.TestFunction(V)
print('space',V.dim(),flush=True)
print('functionspace',u.function_space().dim(),flush=True)
print('nested_mesh',u.function_space().mesh().num_cells(),flush=True)
print('nested_comm',u.function_space().mesh().mpi_comm().size,flush=True)
a=d.Constant(.4)*u+d.Constant(.6)*d.Function(V)
print('arithmetic',a.ufl_shape,flush=True)
F=d.variable(d.grad(a)+d.Identity(3));W=d.inner(F,F);P=d.diff(W,F)
print('form created',flush=True)
r=d.assemble(d.inner(P,d.grad(w))*d.dx)
print('residual',r.norm('l2'),flush=True)
