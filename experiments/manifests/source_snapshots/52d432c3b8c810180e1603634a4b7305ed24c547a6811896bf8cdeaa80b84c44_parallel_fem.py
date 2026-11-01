"""Exact cached geometry and owned/global canonical maps for MPI P2 streaming."""
from pathlib import Path
import hashlib,json
import numpy as np
from .reference_fem import df,LVProblem
from mpi4py import MPI
from scipy.spatial import cKDTree
from cardiac_benchmark.geometry import LVGeometry
from cardiac_benchmark.material import HolzapfelOgden
from .quadratic_solver import make_quadratic_problem

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

class ParallelP2:
    def __init__(self,project,reference,cfg):
        self.comm=MPI.COMM_WORLD;self.rank=self.comm.rank
        cache=Path(project)/'data/processed/parallel_mesh_cache'/reference
        meta=json.loads((cache/'manifest.json').read_text())
        if self.rank==0:
            for n,h in meta['output_sha256'].items():assert sha(cache/n)==h
        df.set_log_level(30);df.parameters['form_compiler']['quadrature_degree']=cfg['quadrature_degree']
        df.parameters['form_compiler']['cpp_optimize']=True;mesh=df.Mesh()
        with df.HDF5File(mesh.mpi_comm(),str(cache/'mesh.h5'),'r') as h:
            h.read(mesh,'mesh',False);ffun=df.MeshFunction('size_t',mesh,2);h.read(ffun,'ffun')
            space=df.VectorFunctionSpace(mesh,'CG',1);fibers=[]
            for key in ['f0','s0','n0']:
                f=df.Function(space);h.read(f,key);fibers.append(f)
        self.geo=LVGeometry(mesh=mesh,ffun=ffun,markers=meta['markers'],f0=fibers[0],s0=fibers[1],n0=fibers[2])
        material=HolzapfelOgden(fibers[0],fibers[1],tau=df.Constant(0.),parameters={k:df.Constant(v) for k,v in meta['material_parameters'].items()})
        params=LVProblem.default_parameters();params.update(function_space='P_2',dt=cfg['dt_s'])
        self.problem=make_quadratic_problem(self.geo,material,params,'gamg',
            linear_maximum_iterations=cfg.get('linear_maximum_iterations',3000),gmres_restart=cfg.get('gmres_restart',200))
        self.problem.solver._solver.parameters.update(dict(absolute_tolerance=1e-8,relative_tolerance=0.,maximum_iterations=100,line_search='bt'))
        self.problem.solver._snes.setUseEW(True,version=2,rtol_0=.01,rtol_max=.1)
        V=self.problem.u_space;coords=V.tabulate_dof_coordinates();n=self.problem.u.vector().local_size();lo,hi=V.dofmap().ownership_range()
        assert n==len(coords)==hi-lo;component=np.full(n,-1,dtype=np.int8)
        for c in range(3):
            ids=np.asarray(V.sub(c).dofmap().dofs());ids=ids[(ids>=lo)&(ids<hi)]-lo;component[ids]=c
        assert (component>=0).all();original=np.load(cache/'serial_dof_map.npz')
        self.mapping=np.empty(n,dtype=np.int64);maximum=0.
        for c in range(3):
            source=np.flatnonzero(original['components']==c);target=np.flatnonzero(component==c)
            distance,near=cKDTree(original['coordinates'][source]).query(coords[target],workers=1)
            maximum=max(maximum,float(distance.max(initial=0.)));self.mapping[target]=source[near]
        assert maximum<1e-12;self.map_error=self.comm.allreduce(maximum,op=MPI.MAX)
        self.parts=self.comm.gather(self.mapping,root=0);self.ndof=len(original['components'])
        if self.rank==0:np.testing.assert_array_equal(np.bincount(np.concatenate(self.parts),minlength=self.ndof),1)
    def assign(self,name,canonical):
        f=getattr(self.problem,name);f.vector().set_local(canonical[self.mapping]);f.vector().apply('insert')
    def gather(self,name):
        values=self.comm.gather(getattr(self.problem,name).vector().get_local(),root=0)
        if self.rank:return None
        result=np.empty(self.ndof)
        for indices,data in zip(self.parts,values):result[indices]=data
        return result
