from .Propagator import Propagator
from .CrownIBP import IBPPropagator, CROWNPropagator
from .AutoLIRPA import CROWNAutoLIRPAPropagator, IBPAutoLIRPAPropagator, CROWNIBPAutoLIRPAPropagator, FastLinAutoLIRPAPropagator, ExhaustiveAutoLIRPAPropagator
from .SDP import SDPPropagator

propagator_dict = {
    "IBP": IBPPropagator,
    "CROWN": CROWNPropagator,
    "CROWN_LIRPA": CROWNAutoLIRPAPropagator,
    "IBP_LIRPA": IBPAutoLIRPAPropagator,
    "CROWN-IBP_LIRPA": CROWNIBPAutoLIRPAPropagator,
    "FastLin_LIRPA": FastLinAutoLIRPAPropagator,
    "Exhaustive_LIRPA": ExhaustiveAutoLIRPAPropagator,
    "SDP": SDPPropagator,
}