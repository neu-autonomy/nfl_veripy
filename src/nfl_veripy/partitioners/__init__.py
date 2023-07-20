from typing import Type

from .AdaptiveGreedySimGuidedPartitioner import (
    AdaptiveGreedySimGuidedPartitioner,
)
from .ClosedLoopGreedySimGuidedPartitioner import (
    ClosedLoopGreedySimGuidedPartitioner,
)
from .ClosedLoopNoPartitioner import ClosedLoopNoPartitioner
from .ClosedLoopPartitioner import ClosedLoopPartitioner  # noqa
from .ClosedLoopSimGuidedPartitioner import ClosedLoopSimGuidedPartitioner
from .ClosedLoopUnGuidedPartitioner import ClosedLoopUnGuidedPartitioner
from .ClosedLoopUniformPartitioner import ClosedLoopUniformPartitioner
from .GreedySimGuidedPartitioner import GreedySimGuidedPartitioner
from .NoPartitioner import NoPartitioner
from .Partitioner import Partitioner  # noqa
from .SimGuidedPartitioner import SimGuidedPartitioner
from .UnGuidedPartitioner import UnGuidedPartitioner
from .UniformPartitioner import UniformPartitioner

partitioner_dict: dict[str, Type[ClosedLoopPartitioner]] = {
    "None": ClosedLoopNoPartitioner,
    "Uniform": ClosedLoopUniformPartitioner,
    "SimGuided": ClosedLoopSimGuidedPartitioner,
    "GreedySimGuided": ClosedLoopGreedySimGuidedPartitioner,
    "UnGuided": ClosedLoopUnGuidedPartitioner,
}


open_loop_partitioner_dict = {
    "None": NoPartitioner,
    "Uniform": UniformPartitioner,
    "SimGuided": SimGuidedPartitioner,
    "GreedySimGuided": GreedySimGuidedPartitioner,
    "AdaptiveGreedySimGuided": AdaptiveGreedySimGuidedPartitioner,
    "UnGuided": UnGuidedPartitioner,
}
