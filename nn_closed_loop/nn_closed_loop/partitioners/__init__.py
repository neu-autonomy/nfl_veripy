from .ClosedLoopPartitioner import ClosedLoopPartitioner
from .ClosedLoopNoPartitioner import ClosedLoopNoPartitioner
from .ClosedLoopUniformPartitioner import ClosedLoopUniformPartitioner
from .ClosedLoopSimGuidedPartitioner import ClosedLoopSimGuidedPartitioner
from .ClosedLoopGreedySimGuidedPartitioner import ClosedLoopGreedySimGuidedPartitioner
from .ClosedLoopUnGuidedPartitioner import ClosedLoopUnGuidedPartitioner

partitioner_dict = {
    "None": ClosedLoopNoPartitioner,
    "Uniform": ClosedLoopUniformPartitioner,
    "SimGuided": ClosedLoopSimGuidedPartitioner,
    "GreedySimGuided": ClosedLoopGreedySimGuidedPartitioner,
    "UnGuided": ClosedLoopUnGuidedPartitioner,
}
