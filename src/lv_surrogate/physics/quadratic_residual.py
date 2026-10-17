"""Differentiable passive P2 weak form in native displacement-DOF ordering.

Same source HGO law, viscosity, generalized-alpha histories and natural loads.
Geometry and fibers remain P1; displacement/test functions are quadratic.
"""
import itertools
import torch
from .cardiac_residual import CardiacFirstStep
from .kinematics import tetra_geometry


class CardiacQuadratic(CardiacFirstStep):
    edges = list(itertools.combinations(range(4), 2))

    @classmethod
    def basis(cls, bary):
        return torch.cat((bary*(2*bary-1), torch.stack(
            [4*bary[..., i]*bary[..., j] for i, j in cls.edges], dim=-1)), dim=-1)

    @classmethod
    def gradients(cls, bary, grad_lambda):
        if bary.ndim == 2:
            bary = bary[None]
        g = grad_lambda[:, None]
        vertex = (4*bary[..., :, None]-1)*g
        edge = torch.stack([4*(bary[..., i, None]*g[..., j, :]+
                               bary[..., j, None]*g[..., i, :])
                            for i, j in cls.edges], dim=-2)
        return torch.cat((vertex, edge), dim=-2)

    def __init__(self, arrays, dtype=torch.float64, dt=.00125):
        self.dtype = dtype
        if float(arrays.get('activation', 0)) != 0:
            raise ValueError('This verified operator supports passive activation only')
        self.X = torch.as_tensor(arrays['coords'], dtype=dtype)
        self.cells = torch.as_tensor(arrays['cells'], dtype=torch.long)
        self.maps = torch.as_tensor(arrays['canonical_cell_dofs'], dtype=torch.long)
        self.inv, self.vol, self.grad_lambda = tetra_geometry(self.X, self.cells)
        q = torch.as_tensor(arrays['qpoints'], dtype=dtype)
        self.bary = torch.cat((1-q.sum(-1, keepdim=True), q), dim=-1)
        weights = torch.as_tensor(arrays['qweights'], dtype=dtype)
        self.weights = weights/weights.sum()
        self.N = self.basis(self.bary)
        self.grad = self.gradients(self.bary, self.grad_lambda)
        self.mass = torch.einsum('q,qa,qb->ab', self.weights, self.N, self.N)
        self.f = torch.einsum('qa,eai->eqi', self.bary,
            torch.as_tensor(arrays['fiber'], dtype=dtype)[self.cells])
        self.s = torch.einsum('qa,eai->eqi', self.bary,
            torch.as_tensor(arrays['sheet'], dtype=dtype)[self.cells])
        self.material = {str(k): float(v) for k, v in zip(arrays['material_keys'], arrays['material_values'])}
        self.facets = torch.as_tensor(arrays['facets'], dtype=torch.long)
        self.normals = torch.as_tensor(arrays['facet_normals'], dtype=dtype)
        self.markers = torch.as_tensor(arrays['facet_markers'], dtype=torch.long)
        self.owners = torch.as_tensor(arrays['facet_cells'], dtype=torch.long)
        self.face_maps = self.maps[self.owners]
        xf = self.X[self.facets]
        self.area = torch.linalg.vector_norm(torch.linalg.cross(xf[:, 1]-xf[:, 0], xf[:, 2]-xf[:, 0]), dim=-1)/2
        triangle = torch.as_tensor(arrays['face_qpoints'], dtype=dtype)
        triangle_bary = torch.cat((1-triangle.sum(-1, keepdim=True), triangle), dim=-1)
        face_weights = torch.as_tensor(arrays['face_qweights'], dtype=dtype)
        self.face_weights = face_weights/face_weights.sum()
        relation = self.cells[self.owners, :, None] == self.facets[:, None, :]
        if not torch.all(relation.sum(dim=1) == 1):
            raise ValueError('Boundary vertex does not belong to its owner tetrahedron')
        positions = relation.long().argmax(dim=1)
        face_bary = torch.zeros((len(self.facets), len(triangle), 4), dtype=dtype)
        face_bary.scatter_(2, positions[:, None].expand(-1, len(triangle), -1),
                           triangle_bary[None].expand(len(self.facets), -1, -1))
        self.face_N = self.basis(face_bary)
        self.face_grad = self.gradients(face_bary, self.grad_lambda[self.owners])
        self.I = torch.eye(3, dtype=dtype)
        self.alpha_f = float(arrays.get('alpha_f', .4)); self.alpha_m = float(arrays.get('alpha_m', .2))
        self.dt = float(dt); self.gamma = .5+self.alpha_f-self.alpha_m
        self.beta = (self.gamma+.5)**2/4
        self.rho = float(arrays.get('rho', 1000))
        self.robin = [(2, float(arrays.get('alpha_epi', 1e8)), float(arrays.get('beta_epi', 5e3))),
                      (3, float(arrays.get('alpha_top', 1e5)), float(arrays.get('beta_top', 5e3)))]

    def piola(self, F):
        p = self.material; cof = self.cofactor(F)
        J = (F[..., 0, :]*cof[..., 0, :]).sum(-1)
        if torch.any(J <= 0):
            raise ValueError('Nonpositive deformation determinant')
        invT = cof/J[..., None, None]; I1 = F.square().sum((-1, -2)); Jm = J.pow(-2/3)
        P = (p['a']*torch.exp(p['b']*(Jm*I1-3))*Jm)[..., None, None]*(F-I1[..., None, None]/3*invT)
        P = P+(.5*p['kappa']*(J.square()-1))[..., None, None]*invT
        vf = torch.einsum('eqij,eqj->eqi', F, self.f)
        vs = torch.einsum('eqij,eqj->eqi', F, self.s)
        for vector, direction, key in [(vf, self.f, 'f'), (vs, self.s, 's')]:
            x = vector.square().sum(-1)-1; switch = torch.sigmoid(p['k']*x)
            exponent = p['b_'+key]*x.square()
            coefficient = p['a_'+key]/p['b_'+key]*(p['k']*switch*(1-switch)*torch.expm1(exponent)+
                2*p['b_'+key]*x*switch*torch.exp(exponent))
            P = P+torch.einsum('eq,eqi,eqj->eqij', coefficient, vector, direction)
        I8 = (vf*vs).sum(-1); coefficient = p['a_fs']*I8*torch.exp(p['b_fs']*I8.square())
        return P+torch.einsum('eq,eqi,eqj->eqij', coefficient, vs, self.f)+torch.einsum('eq,eqi,eqj->eqij', coefficient, vf, self.s)

    def __call__(self, u, pressure, create_graph=False, history=None):
        # Analytic Piola remains differentiable; create_graph is accepted for API parity.
        u = u.to(self.dtype); history = {} if history is None else history
        old = [torch.as_tensor(history.get(k, torch.zeros_like(u)), dtype=self.dtype, device=u.device)
               for k in ['u_old', 'v_old', 'a_old']]
        uold, vold, aold = old
        anew = (u-uold-self.dt*vold-(.5-self.beta)*self.dt**2*aold)/(self.beta*self.dt**2)
        vnew = vold+self.dt*((1-self.gamma)*aold+self.gamma*anew)
        ua = self.alpha_f*uold+(1-self.alpha_f)*u
        va = self.alpha_f*vold+(1-self.alpha_f)*vnew
        aa = self.alpha_m*aold+(1-self.alpha_m)*anew
        F = self.I+torch.einsum('eai,eqaj->eqij', ua[self.maps], self.grad)
        Fdot = torch.einsum('eai,eqaj->eqij', va[self.maps], self.grad)
        Edot = .5*(F.transpose(-1, -2)@Fdot+Fdot.transpose(-1, -2)@F)
        P = self.piola(F)+F@(self.material['eta']*Edot)
        internal = self.vol[:, None, None]*torch.einsum('q,eqij,eqaj->eai', self.weights, P, self.grad)
        inertia = self.rho*self.vol[:, None, None]*torch.einsum('ab,ebi->eai', self.mass, aa[self.maps])
        R = torch.zeros_like(u).index_add(0, self.maps.reshape(-1), (internal+inertia).reshape(-1))
        for marker, alpha, beta in self.robin:
            mask = self.markers == marker
            N = self.face_N[mask]
            uq = torch.einsum('fqa,fai->fqi', N, ua[self.face_maps[mask]])
            vq = torch.einsum('fqa,fai->fqi', N, va[self.face_maps[mask]])
            traction = alpha*uq+beta*vq
            if marker == 2:
                normal = self.normals[mask, None]
                traction = (traction*normal).sum(-1, keepdim=True)*normal
            force = self.area[mask, None, None]*torch.einsum('q,fqa,fqi->fai', self.face_weights, N, traction)
            R = R.index_add(0, self.face_maps[mask].reshape(-1), force.reshape(-1))
        endo = self.markers == 1
        Fface = self.I+torch.einsum('fai,fqaj->fqij', ua[self.face_maps[endo]], self.face_grad[endo])
        traction = float(pressure)*torch.einsum('fqij,fj->fqi', self.cofactor(Fface), self.normals[endo])
        force = self.area[endo, None, None]*torch.einsum('q,fqa,fqi->fai', self.face_weights, self.face_N[endo], traction)
        return R.index_add(0, self.face_maps[endo].reshape(-1), force.reshape(-1))

    def detF(self, u):
        F = self.I+torch.einsum('eai,eqaj->eqij', u.to(self.dtype)[self.maps], self.grad)
        return (F[..., 0, :]*self.cofactor(F)[..., 0, :]).sum(-1)
