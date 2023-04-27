from .AutoLIRPA import (
    CROWNAutoLIRPAPropagator,
    CROWNIBPAutoLIRPAPropagator,
    ExhaustiveAutoLIRPAPropagator,
    FastLinAutoLIRPAPropagator,
    IBPAutoLIRPAPropagator,
)
from .ClosedLoopAUTOLIRPAPropagator import ClosedLoopAUTOLIRPAPropagator
from .ClosedLoopCROWNIBPCodebasePropagator import (  # noqa
    ClosedLoopCROWNIBPCodebasePropagator,
    ClosedLoopCROWNLPPropagator,
    ClosedLoopCROWNNStepPropagator,
    ClosedLoopCROWNPropagator,
    ClosedLoopFastLinPropagator,
    ClosedLoopIBPPropagator,
)
from .ClosedLoopJaxVerifyPropagator import ClosedLoopJaxPropagator  # noqa
from .ClosedLoopJaxVerifyPropagator import (
    ClosedLoopJaxIterativePropagator,
    ClosedLoopJaxLPJittedPropagator,
    ClosedLoopJaxLPPropagator,
    ClosedLoopJaxPolytopeJittedPropagator,
    ClosedLoopJaxPolytopePropagator,
    ClosedLoopJaxRectangleJittedPropagator,
    ClosedLoopJaxRectanglePropagator,
    ClosedLoopJaxUnrolledJittedPropagator,
    ClosedLoopJaxUnrolledPropagator,
)
from .ClosedLoopOVERTPropagator import ClosedLoopOVERTPropagator
from .ClosedLoopPropagator import ClosedLoopPropagator  # noqa
from .ClosedLoopSDPPropagator import ClosedLoopSDPPropagator
from .ClosedLoopSeparablePropagator import (
    ClosedLoopSeparableCROWNPropagator,
    ClosedLoopSeparableIBPPropagator,
    ClosedLoopSeparableSGIBPPropagator,
)
from .CrownIBP import CROWNPropagator, IBPPropagator
from .Propagator import Propagator  # noqa
from .SDP import SDPPropagator

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
    "AutoLiRPA": ClosedLoopAUTOLIRPAPropagator,
}


open_loop_propagator_dict = {
    "IBP": IBPPropagator,
    "CROWN": CROWNPropagator,
    "CROWN_LIRPA": CROWNAutoLIRPAPropagator,
    "IBP_LIRPA": IBPAutoLIRPAPropagator,
    "CROWN-IBP_LIRPA": CROWNIBPAutoLIRPAPropagator,
    "FastLin_LIRPA": FastLinAutoLIRPAPropagator,
    "Exhaustive_LIRPA": ExhaustiveAutoLIRPAPropagator,
    "SDP": SDPPropagator,
}
