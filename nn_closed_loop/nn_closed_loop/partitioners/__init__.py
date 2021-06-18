from .ClosedLoopPartitioner import ClosedLoopPartitioner
from .ClosedLoopNoPartitioner import ClosedLoopNoPartitioner
from .ClosedLoopUniformPartitioner import ClosedLoopUniformPartitioner
from .ClosedLoopSimGuidedPartitioner import ClosedLoopSimGuidedPartitioner
from .ClosedLoopGreedySimGuidedPartitioner import ClosedLoopGreedySimGuidedPartitioner

partitioner_dict = {
    "None": ClosedLoopNoPartitioner,
    "Uniform": ClosedLoopUniformPartitioner,
    "SimGuided": ClosedLoopSimGuidedPartitioner,
    "GreedySimGuided": ClosedLoopGreedySimGuidedPartitioner,
}
