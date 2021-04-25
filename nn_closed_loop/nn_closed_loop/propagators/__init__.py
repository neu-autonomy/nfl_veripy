from .ClosedLoopPropagator import ClosedLoopPropagator
from .ClosedLoopSDPPropagator import ClosedLoopSDPPropagator
from .ClosedLoopCROWNIBPCodebasePropagator import (
    ClosedLoopCROWNIBPCodebasePropagator,
    ClosedLoopCROWNLPPropagator,
    ClosedLoopIBPPropagator,
    ClosedLoopCROWNPropagator,
    ClosedLoopFastLinPropagator,
)

propagator_dict = {
    "CROWN": ClosedLoopCROWNPropagator,
    "CROWNLP": ClosedLoopCROWNLPPropagator,
    "IBP": ClosedLoopIBPPropagator,
    "FastLin": ClosedLoopFastLinPropagator,
    "SDP": ClosedLoopSDPPropagator,
}
