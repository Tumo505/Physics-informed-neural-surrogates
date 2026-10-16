"""Input-support admission and explicit FE routing, without test-error thresholds."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class InputDomainGate:
    bounds: dict

    def decide(self, inputs, minimum_detF=None):
        reasons=[]
        for name,(lower,upper) in self.bounds.items():
            if name not in inputs:reasons.append(f'missing:{name}');continue
            value=float(inputs[name])
            if not math.isfinite(value):reasons.append(f'nonfinite:{name}')
            elif value<lower or value>upper:reasons.append(f'outside_support:{name}')
        if minimum_detF is not None and (not math.isfinite(minimum_detF) or minimum_detF<=0):reasons.append('invalid_predicted_geometry')
        return {'action':'FE' if reasons else 'surrogate','reasons':reasons}

    def dispatch(self, inputs, surrogate, finite_element, validate_prediction=None):
        decision=self.decide(inputs)
        if decision['action']=='FE':return decision,finite_element()
        prediction=surrogate()
        if validate_prediction is not None:
            decision=self.decide(inputs,minimum_detF=validate_prediction(prediction))
            if decision['action']=='FE':return decision,finite_element()
        return decision,prediction
