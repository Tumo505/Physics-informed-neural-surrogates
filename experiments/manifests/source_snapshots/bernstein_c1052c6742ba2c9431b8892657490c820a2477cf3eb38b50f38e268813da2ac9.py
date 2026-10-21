"""Memory-bounded whole-tetrahedron P2 positivity, retaining inconclusive parents."""
import itertools
import math
import numpy as np

class BernsteinAudit:
    def __init__(self, arrays, chunk=8192, margin=1e-9, max_depth=12):
        self.X=np.asarray(arrays['coords'])[arrays['cells']]
        inverse=np.linalg.inv((self.X[:,1:]-self.X[:,:1]).transpose(0,2,1))
        self.gl=np.concatenate((-inverse.sum(1,keepdims=True),inverse),axis=1)
        self.maps=np.asarray(arrays['canonical_cell_dofs']);self.chunk=chunk
        self.margin=margin;self.max_depth=max_depth
        self.edges=list(itertools.combinations(range(4),2))
        multi=np.array([a for a in itertools.product(range(4),repeat=4) if sum(a)==3])
        self.multi=multi;self.lattice=multi/3
        self.mult=np.array([math.factorial(3)/math.prod(math.factorial(int(i)) for i in a) for a in multi])
        self.transform=np.linalg.inv(self.basis(self.lattice))
        self.check=np.random.default_rng(2027).dirichlet(np.ones(4),size=7)
    def basis(self,b):
        return self.mult[None,:]*np.prod(np.asarray(b)[:,None,:]**self.multi[None,:,:],axis=-1)
    def resolve(self,F,X):
        stack=[(np.eye(4),0)];low=float('inf');leaves=0;depth_max=0
        while stack:
            vertices,depth=stack.pop();depth_max=max(depth_max,depth)
            points=self.lattice@vertices;values=np.linalg.det(np.einsum('qv,vij->qij',points,F))
            assert np.isfinite(values).all(), 'Nonfinite determinant'
            if values.min()<=0:raise ValueError('Direct nonpositive determinant witness')
            coefficients=values@self.transform.T
            if coefficients.min()>self.margin:
                low=min(low,float(coefficients.min()));leaves+=1;continue
            if depth==self.max_depth:raise ValueError('Whole-element positivity remains unresolved at maximum depth')
            physical=vertices@X
            a,b=max(self.edges,key=lambda e:float(np.sum((physical[e[0]]-physical[e[1]])**2)))
            midpoint=(vertices[a]+vertices[b])/2;one=vertices.copy();two=vertices.copy()
            one[a]=midpoint;two[b]=midpoint;stack.extend([(one,depth+1),(two,depth+1)])
        return dict(lower_bound=low,leaf_tetrahedra=leaves,maximum_depth=depth_max)
    def field(self,u):
        minimum=float('inf');parent=float('inf');refined=[];error=0.
        for start in range(0,len(self.maps),self.chunk):
            stop=min(start+self.chunk,len(self.maps));gl=self.gl[start:stop]
            grad=np.stack([np.concatenate(((4*b[None,:,None]-1)*gl,
                np.stack([4*(b[i]*gl[:,j]+b[j]*gl[:,i]) for i,j in self.edges],axis=1)),axis=1) for b in np.eye(4)],axis=1)
            F=np.eye(3)+np.einsum('eki,evkj->evij',u[self.maps[start:stop]],grad)
            values=np.linalg.det(np.einsum('qv,evij->eqij',self.lattice,F));coeff=values@self.transform.T
            assert np.isfinite(coeff).all();lower=coeff.min(axis=1);parent=min(parent,float(lower.min()))
            positive=lower[lower>self.margin]
            if len(positive):minimum=min(minimum,float(positive.min()))
            for i in np.flatnonzero(lower<=self.margin):
                proof=self.resolve(F[i],self.X[start+i]);minimum=min(minimum,proof['lower_bound'])
                refined.append(dict(element=int(start+i),classification='positive_bound',**proof))
            # An independent random-point identity on each chunk checks polynomial ordering.
            direct=np.linalg.det(np.einsum('qv,vij->qij',self.check,F[0]))
            reconstructed=self.basis(self.check)@coeff[0]
            np.testing.assert_allclose(direct,reconstructed,rtol=1e-11,atol=1e-11)
            error=max(error,float(np.max(abs(direct-reconstructed))))
        assert minimum>self.margin
        return dict(minimum_Bernstein_bound=minimum,minimum_parent_Bernstein_bound=parent,
            original_unresolved_elements=len(refined),refined_elements=refined,
            polynomial_reconstruction_max_abs_error=error,passed=True)
    def pair(self,u,old,alpha_f=.4):
        return {name:self.field(field) for name,field in [('endpoint',u),('stress',alpha_f*old+(1-alpha_f)*u)]}
