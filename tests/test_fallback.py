from lv_surrogate.eval.fallback import InputDomainGate

def test_support_gate_routes_unknown_shifted_and_invalid_inputs():
    gate=InputDomainGate({'radial':(.95,1.05)})
    assert gate.decide({'radial':.95})['action']=='surrogate'
    for inputs in [{},{'radial':1.050001},{'radial':float('nan')}]:assert gate.decide(inputs)['action']=='FE'
    assert gate.decide({'radial':1.},minimum_detF=-.1)['action']=='FE'
    calls=[]
    decision,result=gate.dispatch({'radial':1.1},lambda:calls.append('NN'),lambda:calls.append('FE'))
    assert decision['action']=='FE' and calls==['FE']


def test_in_support_invalid_prediction_falls_back_after_validation():
    gate=InputDomainGate({'radial':(.95,1.05)})
    calls=[]
    decision,result=gate.dispatch({'radial':1.},lambda:{'minimum_J':-.1},lambda:calls.append('FE'),validate_prediction=lambda p:p['minimum_J'])
    assert decision['action']=='FE' and decision['reasons']==['invalid_predicted_geometry'] and calls==['FE']
