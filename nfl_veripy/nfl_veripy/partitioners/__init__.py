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


from .AdaptiveGreedySimGuidedPartitioner import (
    AdaptiveGreedySimGuidedPartitioner,
)
from .GreedySimGuidedPartitioner import GreedySimGuidedPartitioner
from .NoPartitioner import NoPartitioner
from .Partitioner import Partitioner
from .SimGuidedPartitioner import SimGuidedPartitioner
from .UnGuidedPartitioner import UnGuidedPartitioner
from .UniformPartitioner import UniformPartitioner

open_loop_partitioner_dict = {
    "None": NoPartitioner,
    "Uniform": UniformPartitioner,
    "SimGuided": SimGuidedPartitioner,
    "GreedySimGuided": GreedySimGuidedPartitioner,
    "AdaptiveGreedySimGuided": AdaptiveGreedySimGuidedPartitioner,
    "UnGuided": UnGuidedPartitioner,
}
