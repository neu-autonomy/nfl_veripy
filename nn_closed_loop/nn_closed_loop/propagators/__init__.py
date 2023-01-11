from .ClosedLoopPropagator import ClosedLoopPropagator
from .ClosedLoopSDPPropagator import ClosedLoopSDPPropagator
from .ClosedLoopOVERTPropagator import ClosedLoopOVERTPropagator
from .ClosedLoopCROWNIBPCodebasePropagator import (
    ClosedLoopCROWNIBPCodebasePropagator,
    ClosedLoopCROWNLPPropagator,
    ClosedLoopIBPPropagator,
    ClosedLoopCROWNPropagator,
    ClosedLoopCROWNNStepPropagator,
    ClosedLoopCROWNRefinedPropagator,
    ClosedLoopFastLinPropagator,
)
from .ClosedLoopSeparablePropagator import (
    ClosedLoopSeparableCROWNPropagator,
    ClosedLoopSeparableIBPPropagator,
    ClosedLoopSeparableSGIBPPropagator,
)
from .ClosedLoopJaxVerifyPropagator import (
    ClosedLoopJaxPropagator,
    ClosedLoopJaxIterativePropagator,
    ClosedLoopJaxUnrolledPropagator,
    ClosedLoopJaxUnrolledJittedPropagator,
    ClosedLoopJaxPolytopePropagator,
    ClosedLoopJaxRectanglePropagator,
    ClosedLoopJaxLPPropagator,
    ClosedLoopJaxPolytopeJittedPropagator,
    ClosedLoopJaxRectangleJittedPropagator,
    ClosedLoopJaxLPJittedPropagator,
)

propagator_dict = {
    "CROWN": ClosedLoopCROWNPropagator,
    "CROWNNStep": ClosedLoopCROWNNStepPropagator,
    "CROWNRefined": ClosedLoopCROWNRefinedPropagator,
    "CROWNLP": ClosedLoopCROWNLPPropagator,
    "IBP": ClosedLoopIBPPropagator,
    "FastLin": ClosedLoopFastLinPropagator,
    "SDP": ClosedLoopSDPPropagator,
    "SeparableCROWN": ClosedLoopSeparableCROWNPropagator,
    "SeparableIBP": ClosedLoopSeparableIBPPropagator,
    "SeparableSGIBP": ClosedLoopSeparableSGIBPPropagator,
    "OVERT": ClosedLoopOVERTPropagator,
    "TorchBackwardCROWNReachLP": ClosedLoopCROWNPropagator,
    "TorchBackwardCROWNNStep": ClosedLoopCROWNNStepPropagator,
    "TorchCROWNLP": ClosedLoopCROWNLPPropagator,
    "TorchIBP": ClosedLoopIBPPropagator,
    "TorchFastLin": ClosedLoopFastLinPropagator,
    "JaxCROWNIterative": ClosedLoopJaxIterativePropagator,
    "JaxCROWNUnrolled": ClosedLoopJaxUnrolledPropagator,
    "JaxUnrolledJitted": ClosedLoopJaxUnrolledJittedPropagator,
    "JaxPolytope": ClosedLoopJaxPolytopePropagator,
    "JaxRectangle": ClosedLoopJaxRectanglePropagator,
    "JaxLP": ClosedLoopJaxLPPropagator,
    "JaxPolytopeJitted": ClosedLoopJaxPolytopeJittedPropagator,
    "JaxRectangleJitted": ClosedLoopJaxRectangleJittedPropagator,
    "JaxLPJitted": ClosedLoopJaxLPJittedPropagator,
}
