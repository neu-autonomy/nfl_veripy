from .ClosedLoopPropagator import ClosedLoopPropagator
from .ClosedLoopSDPPropagator import ClosedLoopSDPPropagator
from .ClosedLoopCROWNIBPCodebasePropagator import (
    ClosedLoopCROWNIBPCodebasePropagator,
    ClosedLoopIBPPropagator,
    ClosedLoopCROWNPropagator,
    ClosedLoopFastLinPropagator,
)

propagator_dict = {
    "CROWN": ClosedLoopCROWNPropagator,
    "IBP": ClosedLoopIBPPropagator,
    "FastLin": ClosedLoopFastLinPropagator,
    "SDP": ClosedLoopSDPPropagator,
}
