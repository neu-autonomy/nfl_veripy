from .ClosedLoopGreedySimGuidedPartitioner import (
    ClosedLoopGreedySimGuidedPartitioner,
)
from .ClosedLoopNickPartitioner import ClosedLoopNickPartitioner
from .ClosedLoopNoPartitioner import ClosedLoopNoPartitioner
from .ClosedLoopPartitioner import ClosedLoopPartitioner  # noqa
from .ClosedLoopSimGuidedPartitioner import ClosedLoopSimGuidedPartitioner
from .ClosedLoopUnGuidedPartitioner import ClosedLoopUnGuidedPartitioner
from .ClosedLoopUniformPartitioner import ClosedLoopUniformPartitioner

partitioner_dict = {
    "None": ClosedLoopNoPartitioner,
    "Uniform": ClosedLoopUniformPartitioner,
    "SimGuided": ClosedLoopSimGuidedPartitioner,
    "GreedySimGuided": ClosedLoopGreedySimGuidedPartitioner,
    "UnGuided": ClosedLoopUnGuidedPartitioner,
    "Nick": ClosedLoopNickPartitioner,
}
