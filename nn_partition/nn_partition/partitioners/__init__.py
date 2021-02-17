from .Partitioner import Partitioner
from .NoPartitioner import NoPartitioner
from .UniformPartitioner import UniformPartitioner
from .SimGuidedPartitioner import SimGuidedPartitioner
from .GreedySimGuidedPartitioner import GreedySimGuidedPartitioner
from .AdaptiveGreedySimGuidedPartitioner import (
    AdaptiveGreedySimGuidedPartitioner,
)
from .UnGuidedPartitioner import UnGuidedPartitioner

partitioner_dict = {
    "None": NoPartitioner,
    "Uniform": UniformPartitioner,
    "SimGuided": SimGuidedPartitioner,
    "GreedySimGuided": GreedySimGuidedPartitioner,
    "AdaptiveGreedySimGuided": AdaptiveGreedySimGuidedPartitioner,
    "UnGuided": UnGuidedPartitioner,
}
