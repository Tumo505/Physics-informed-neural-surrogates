"""Independent differentiable P1 assembly of the pinned first-step LV equations.

Units m/Pa/N/s. Includes benchmark anisotropy, viscosity, generalized-alpha
inertia, follower pressure, epicardial normal and base vector Robin terms.
Optional displacement/velocity/acceleration histories use the same source scheme.
"""
import torch
from .kinematics import tetra_geometry

class CardiacFirstStep:
    def __init__(self, arrays, dtype=torch.float64,dt=.001,analytic_piola=False):
        self.dtype=dtype
        self.analytic_piola=analytic_piola
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
        self.alpha_f=.4; self.alpha_m=.2; self.dt=float(dt)
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

    def to(self,device):
        for name,value in vars(self).copy().items():
            if isinstance(value,torch.Tensor):setattr(self,name,value.to(device))
        return self

    @staticmethod
    def cofactor(F):
        return torch.stack((torch.linalg.cross(F[...,1,:],F[...,2,:]),
                            torch.linalg.cross(F[...,2,:],F[...,0,:]),
                            torch.linalg.cross(F[...,0,:],F[...,1,:])),-2)

    def piola(self,F):
        """Exact dW/dF, retaining derivative of the smoothed fiber switch.

        Isotropic terms are elementwise; only fiber invariants require quadrature.
        Verified against energy autodiff and the independent trusted tangent.
        """
        p=self.material;cof=self.cofactor(F);J=(F[:,0]*cof[:,0]).sum(-1)
        if torch.any(J<=0):raise ValueError('nonpositive deformation determinant')
        invT=cof/J[:,None,None];I1=F.square().sum((-1,-2));Jm=J.pow(-2/3)
        P=(p['a']*torch.exp(p['b']*(Jm*I1-3))*Jm)[:,None,None]*(F-I1[:,None,None]/3*invT)
        P=P+(.5*p['kappa']*(J.square()-1))[:,None,None]*invT
        vf=torch.einsum('eij,eqj->eqi',F,self.f);vs=torch.einsum('eij,eqj->eqi',F,self.s)
        for vector,direction,key in [(vf,self.f,'f'),(vs,self.s,'s')]:
            x=vector.square().sum(-1)-1;switch=torch.sigmoid(p['k']*x);exponent=p['b_'+key]*x.square()
            coefficient=p['a_'+key]/p['b_'+key]*(p['k']*switch*(1-switch)*torch.expm1(exponent)+2*p['b_'+key]*x*switch*torch.exp(exponent))
            P=P+torch.einsum('eq,eqi,eqj,q->eij',coefficient,vector,direction,self.weights)
        I8=(vf*vs).sum(-1);coefficient=p['a_fs']*I8*torch.exp(p['b_fs']*I8.square())
        return P+torch.einsum('eq,eqi,eqj,q->eij',coefficient,vs,self.f,self.weights)+torch.einsum('eq,eqi,eqj,q->eij',coefficient,vf,self.s,self.weights)

    def __call__(self,u,pressure,create_graph=False,history=None):
        u=u.to(self.dtype)
        if history is None:history={}
        old=[torch.as_tensor(history.get(k,torch.zeros_like(u)),dtype=self.dtype,device=u.device) for k in ['u_old','v_old','a_old']]
        uold,vold,aold=old
        anew=(u-uold-self.dt*vold-(.5-self.beta)*self.dt**2*aold)/(self.beta*self.dt**2)
        vnew=vold+self.dt*((1-self.gamma)*aold+self.gamma*anew)
        ua=self.alpha_f*uold+(1-self.alpha_f)*u
        va=self.alpha_f*vold+(1-self.alpha_f)*vnew
        aa=self.alpha_m*aold+(1-self.alpha_m)*anew
        gradu=torch.einsum('eai,eaj->eij',ua[self.cells],self.grad)
        F=self.I+gradu
        if self.analytic_piola:P=self.piola(F)
        else:
            Fq=F[:,None].expand(-1,len(self.weights),-1,-1)
            if not Fq.requires_grad:Fq=Fq.detach().requires_grad_(True)
            W=self.energy_density(Fq)
            Pq=torch.autograd.grad(W.sum(),Fq,create_graph=create_graph)[0]
            P=torch.einsum('eqij,q->eij',Pq,self.weights)
        Fdot=torch.einsum('eai,eaj->eij',va[self.cells],self.grad)
        Edot=.5*(F.transpose(-1,-2)@Fdot+Fdot.transpose(-1,-2)@F)
        P=P+F@(self.material['eta']*Edot)
        element_force=self.vol[:,None,None]*torch.einsum('eij,eaj->eai',P,self.grad)
        ae=aa[self.cells]
        inertia=1000*(self.vol[:,None,None]/20)*(ae.sum(1,keepdim=True)+ae)
        R=torch.zeros_like(u).index_add(0,self.cells.reshape(-1),(element_force+inertia).reshape(-1,3))
        # Boundary P1 consistent mass on each triangular facet.
        uf=ua[self.facets];vf=va[self.facets]
        mass_u=self.area[:,None,None]/12*(uf.sum(1,keepdim=True)+uf)
        mass_v=self.area[:,None,None]/12*(vf.sum(1,keepdim=True)+vf)
        for marker,alpha,beta in [(2,1e8,5e3),(3,1e5,5e3)]:
            mask=self.markers==marker
            if marker==2:
                normal=self.normals[mask,None,:];f=((alpha*mass_u[mask]+beta*mass_v[mask])*normal).sum(-1,keepdim=True)*normal
            else:f=alpha*mass_u[mask]+beta*mass_v[mask]
            R=R.index_add(0,self.facets[mask].reshape(-1),f.reshape(-1,3))
        endo=self.markers==1;Fe=F[self.owners[endo]];N=self.normals[endo]
        traction=float(pressure)*torch.einsum('eij,ej->ei',self.cofactor(Fe),N)
        force=self.area[endo,None,None]/3*traction[:,None,:].expand(-1,3,-1)
        return R.index_add(0,self.facets[endo].reshape(-1),force.reshape(-1,3))

    def detF(self,u):
        gradu=torch.einsum('eai,eaj->eij',u[self.cells].to(self.dtype),self.grad)
        return torch.linalg.det(self.I+gradu)
