"""Independent differentiable P1 assembly of the pinned first-step LV equations.

Units m/Pa/N/s. Includes benchmark anisotropy, viscosity, generalized-alpha
inertia, follower pressure, epicardial normal and base vector Robin terms.
Only zero initial displacement/velocity/acceleration is supported.
"""
import torch
from .kinematics import tetra_geometry

class CardiacFirstStep:
    def __init__(self, arrays, dtype=torch.float64):
        self.dtype=dtype
        self.X=torch.as_tensor(arrays['coords'],dtype=dtype)
        self.cells=torch.as_tensor(arrays['cells'],dtype=torch.long)
        self.inv,self.vol,self.grad=tetra_geometry(self.X,self.cells)
        q=torch.as_tensor(arrays['qpoints'],dtype=dtype)
        self.bary=torch.cat((1-q.sum(-1,keepdim=True),q),-1)
        weights=torch.as_tensor(arrays['qweights'],dtype=dtype)
        self.weights=weights/weights.sum()
        self.f=torch.einsum('qa,eai->eqi',self.bary,torch.as_tensor(arrays['fiber'],dtype=dtype)[self.cells])
        self.s=torch.einsum('qa,eai->eqi',self.bary,torch.as_tensor(arrays['sheet'],dtype=dtype)[self.cells])
        self.facets=torch.as_tensor(arrays['facets'],dtype=torch.long)
        self.normals=torch.as_tensor(arrays['facet_normals'],dtype=dtype)
        self.markers=torch.as_tensor(arrays['facet_markers'],dtype=torch.long)
        self.owners=torch.as_tensor(arrays['facet_cells'],dtype=torch.long)
        xf=self.X[self.facets]
        self.area=torch.linalg.vector_norm(torch.linalg.cross(xf[:,1]-xf[:,0],xf[:,2]-xf[:,0]),dim=-1)/2
        self.material={str(k):float(v) for k,v in zip(arrays['material_keys'],arrays['material_values'])}
        self.I=torch.eye(3,dtype=dtype)
        self.alpha_f=.4; self.alpha_m=.2; self.dt=.001
        self.gamma=.5+self.alpha_f-self.alpha_m
        self.beta=(self.gamma+.5)**2/4

    def energy_density(self,F):
        p=self.material;C=F.transpose(-1,-2)@F;J=torch.linalg.det(F)
        if torch.any(J<=0):raise ValueError('nonpositive deformation determinant')
        I1=J.pow(-2/3)*C.diagonal(dim1=-2,dim2=-1).sum(-1)
        I4f=torch.einsum('eqi,eqij,eqj->eq',self.f,C,self.f)
        I4s=torch.einsum('eqi,eqij,eqj->eq',self.s,C,self.s)
        I8=torch.einsum('eqi,eqij,eqj->eq',self.f,C,self.s)
        W=p['a']/(2*p['b'])*torch.expm1(p['b']*(I1-3))
        for I4,d in [(I4f,'f'),(I4s,'s')]:
            W=W+p['a_'+d]/(2*p['b_'+d])*torch.sigmoid(p['k']*(I4-1))*torch.expm1(p['b_'+d]*(I4-1).square())
        W=W+p['a_fs']/(2*p['b_fs'])*torch.expm1(p['b_fs']*I8.square())
        return W+.25*p['kappa']*(J.square()-1-2*J.log())

    def __call__(self,u,pressure,create_graph=False):
        u=u.to(self.dtype);ue=u[self.cells]
        gradu=torch.einsum('eai,eaj->eij',ue,self.grad)
        F=self.I+(1-self.alpha_f)*gradu
        Fq=F[:,None].expand(-1,len(self.weights),-1,-1)
        if not Fq.requires_grad:Fq=Fq.detach().requires_grad_(True)
        W=self.energy_density(Fq)
        Pq=torch.autograd.grad(W.sum(),Fq,create_graph=create_graph)[0]
        P=torch.einsum('eqij,q->eij',Pq,self.weights)
        Fdot=(1-self.alpha_f)*self.gamma/(self.beta*self.dt)*gradu
        Edot=.5*(F.transpose(-1,-2)@Fdot+Fdot.transpose(-1,-2)@F)
        P=P+F@(self.material['eta']*Edot)
        element_force=self.vol[:,None,None]*torch.einsum('eij,eaj->eai',P,self.grad)
        inertia=(1000*(1-self.alpha_m)/(self.beta*self.dt**2))*(self.vol[:,None,None]/20)*(ue.sum(1,keepdim=True)+ue)
        R=torch.zeros_like(u).index_add(0,self.cells.reshape(-1),(element_force+inertia).reshape(-1,3))
        # Boundary P1 consistent mass on each triangular facet.
        uf=u[self.facets];mass_u=self.area[:,None,None]/12*(uf.sum(1,keepdim=True)+uf)
        for marker,alpha,beta in [(2,1e8,5e3),(3,1e5,5e3)]:
            mask=self.markers==marker
            if marker==2:
                normal=self.normals[mask,None,:];f=(mass_u[mask]*normal).sum(-1,keepdim=True)*normal
            else:f=mass_u[mask]
            coefficient=(1-self.alpha_f)*(alpha+beta*self.gamma/(self.beta*self.dt))
            R=R.index_add(0,self.facets[mask].reshape(-1),(coefficient*f).reshape(-1,3))
        endo=self.markers==1;Fe=F[self.owners[endo]];N=self.normals[endo]
        traction=float(pressure)*torch.linalg.det(Fe)[:,None]*torch.einsum('eij,ej->ei',torch.linalg.inv(Fe).transpose(-1,-2),N)
        force=self.area[endo,None,None]/3*traction[:,None,:].expand(-1,3,-1)
        return R.index_add(0,self.facets[endo].reshape(-1),force.reshape(-1,3))

    def detF(self,u):
        gradu=torch.einsum('eai,eaj->eij',u[self.cells].to(self.dtype),self.grad)
        return torch.linalg.det(self.I+gradu)
