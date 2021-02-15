from .ClosedLoopPartitioner import ClosedLoopPartitioner
from .ClosedLoopNoPartitioner import ClosedLoopNoPartitioner
from .ClosedLoopUniformPartitioner import ClosedLoopUniformPartitioner

partitioner_dict = {
    "None": ClosedLoopNoPartitioner,
    "Uniform": ClosedLoopUniformPartitioner,
    # "ProbPartition": ClosedLoopProbabilisticPartitioner,
}
