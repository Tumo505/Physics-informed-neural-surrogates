import numpy as np
from lv_surrogate.models.dynamic_graph import newmark_operators

def test_newmark_maps_are_causal_and_match_sequential_history_update():
    rng=np.random.default_rng(11);q=rng.normal(size=(20,3));dt=.005;V,A=newmark_operators(len(q),dt);v=np.zeros(3);a=np.zeros(3);old=np.zeros(3);vs=[];acc=[]
    for u in q:
        new=(u-old-dt*v-(.5-.36)*dt**2*a)/(.36*dt**2)
        v=v+dt*(.3*a+.7*new);a=new;old=u;vs.append(v.copy());acc.append(a.copy())
    np.testing.assert_allclose(V@q,vs,rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(A@q,acc,rtol=1e-12,atol=1e-8)
    assert np.all(V[np.triu_indices(len(q),1)]==0)
    assert np.all(A[np.triu_indices(len(q),1)]==0)
