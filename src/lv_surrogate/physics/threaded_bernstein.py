"""Disjoint-cell threaded evaluation of the unchanged whole-element proof."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .bernstein import BernsteinAudit

class ThreadedBernsteinAudit(BernsteinAudit):
    def __init__(self,arrays,chunk=8192,margin=1e-9,max_depth=12,workers=4):
        super().__init__(arrays,chunk,margin,max_depth)
        self.executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='whole_element')
        edges=np.linspace(0,len(self.maps),workers+1,dtype=int);self.shards=[]
        assert edges[0]==0 and edges[-1]==len(self.maps) and (np.diff(edges)>0).all()
        for start,stop in zip(edges[:-1],edges[1:]):
            child=object.__new__(BernsteinAudit);child.__dict__={k:v for k,v in self.__dict__.items() if k not in ['executor','shards']}
            child.maps=self.maps[start:stop];child.gl=self.gl[start:stop];child.X=self.X[start:stop]
            self.shards.append((int(start),child))
    def field(self,u):
        def evaluate(shard):
            start,child=shard;r=child.field(u)
            for item in r['refined_elements']:item['element']+=start
            return r
        records=list(self.executor.map(evaluate,self.shards))
        return dict(minimum_Bernstein_bound=min(r['minimum_Bernstein_bound'] for r in records),
            minimum_parent_Bernstein_bound=min(r['minimum_parent_Bernstein_bound'] for r in records),
            original_unresolved_elements=sum(r['original_unresolved_elements'] for r in records),
            refined_elements=[e for r in records for e in r['refined_elements']],
            polynomial_reconstruction_max_abs_error=max(r['polynomial_reconstruction_max_abs_error'] for r in records),
            passed=all(r['passed'] for r in records),evaluation_workers=len(self.shards),
            disjoint_cells_total=sum(len(child.maps) for _,child in self.shards))
    def close(self):self.executor.shutdown(wait=True)
