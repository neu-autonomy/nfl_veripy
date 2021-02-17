from .GreedySimGuidedPartitioner import GreedySimGuidedPartitioner


class AdaptiveGreedySimGuidedPartitioner(GreedySimGuidedPartitioner):
    def __init__(
        self,
        num_simulations=1000,
        interior_condition="linf",
        make_animation=False,
        show_animation=False,
        termination_condition_type="interior_cell_size",
        termination_condition_value=0.02,
        show_input=True,
        show_output=True,
    ):
        GreedySimGuidedPartitioner.__init__(
            self,
            num_simulations=num_simulations,
            interior_condition=interior_condition,
            make_animation=make_animation,
            show_animation=show_animation,
            termination_condition_type=termination_condition_type,
            termination_condition_value=termination_condition_value,
            show_input=show_input,
            show_output=show_output,
            adaptive_flag=True,
        )
