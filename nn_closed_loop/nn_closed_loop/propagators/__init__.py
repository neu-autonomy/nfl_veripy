from .ClosedLoopPropagator import ClosedLoopPropagator
from .ClosedLoopSDPPropagator import ClosedLoopSDPPropagator
from .ClosedLoopOVERTPropagator import ClosedLoopOVERTPropagator
from .ClosedLoopCROWNIBPCodebasePropagator import (
    ClosedLoopCROWNIBPCodebasePropagator,
    ClosedLoopCROWNLPPropagator,
    ClosedLoopIBPPropagator,
    ClosedLoopCROWNPropagator,
    ClosedLoopCROWNNStepPropagator,
    ClosedLoopFastLinPropagator,
)
from .ClosedLoopSeparablePropagator import (
    ClosedLoopSeparableCROWNPropagator,
    ClosedLoopSeparableIBPPropagator,
    ClosedLoopSeparableSGIBPPropagator,
)
from .ClosedLoopAUTOLIRPAPropagator import ClosedLoopAUTOLIRPAPropagator

propagator_dict = {
    "CROWN": ClosedLoopCROWNPropagator,
    "CROWNNStep": ClosedLoopCROWNNStepPropagator,
    "CROWNLP": ClosedLoopCROWNLPPropagator,
    "IBP": ClosedLoopIBPPropagator,
    "FastLin": ClosedLoopFastLinPropagator,
    "SDP": ClosedLoopSDPPropagator,
    "SeparableCROWN": ClosedLoopSeparableCROWNPropagator,
    "SeparableIBP": ClosedLoopSeparableIBPPropagator,
    "SeparableSGIBP": ClosedLoopSeparableSGIBPPropagator,
    "OVERT": ClosedLoopOVERTPropagator,
    "AutoLiRPA": ClosedLoopAUTOLIRPAPropagator
}
