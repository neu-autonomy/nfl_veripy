import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Analyzer:
    def __init__(self, torch_model):
        self.torch_model = torch_model

        self.partitioner = None
        self.propagator = None

    @property
    def partitioner(self):
        return self._partitioner

    @partitioner.setter
    def partitioner(self, partitioner):
        self._partitioner = partitioner

    @property
    def propagator(self):
        return self._propagator

    @propagator.setter
    def propagator(self, propagator):
        self._propagator = propagator
        if propagator is not None:
            self._propagator.network = self.torch_model

    def get_output_range(self, input_range):
        output_range, info = self.partitioner.get_output_range(input_range, self.propagator)
        return output_range, info

    def visualize(self, input_range, output_range_estimate, **kwargs):
        # def visualize_partitions(sampled_outputs, estimated_output_range, input_range, output_range_sim=None, interior_M=None, M=None):
        fig, axes = plt.subplots(1,2)

        if "all_partitions" in kwargs:
        # if interior_M is not None and len(interior_M) > 0:
            for (input_range_, output_range_) in kwargs["all_partitions"]:
                rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                        fc='none', linewidth=1,edgecolor='m')
                axes[1].add_patch(rect)

                rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
                        fc='none', linewidth=1,edgecolor='m')
                axes[0].add_patch(rect)
        # if M is not None and len(M) > 0:
        #     for (input_range_, output_range_) in M:
        #         rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
        #                 fc='none', linewidth=1,edgecolor='b')
        #         axes[1].add_patch(rect)

        #         rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
        #                 fc='none', linewidth=1,edgecolor='b')
        #         axes[0].add_patch(rect)

        # Make a rectangle for the estimated boundaries
        rect = Rectangle(output_range_estimate[:,0], output_range_estimate[0,1]-output_range_estimate[0,0], output_range_estimate[1,1]-output_range_estimate[1,0],
                        fc='none', linewidth=2,edgecolor='g')
        axes[1].add_patch(rect)

        # Make a rectangle for the Exact boundaries
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range_exact = self.samples_to_range(sampled_outputs)
        axes[1].scatter(sampled_outputs[:,0], sampled_outputs[:,1], c='k', zorder=2)
        rect = Rectangle(output_range_exact[:,0], output_range_exact[0,1]-output_range_exact[0,0], output_range_exact[1,1]-output_range_exact[1,0],
                        fc='none', linewidth=1,edgecolor='k')
        axes[1].add_patch(rect)

        axes[0].set_xlim(input_range[0,0], input_range[0,1])
        axes[0].set_ylim(input_range[1,0], input_range[1,1])

        plt.show()

    def get_sampled_outputs(self, input_range, N=1000):
        num_inputs = input_range.shape[0]
        sampled_inputs = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
        sampled_outputs = self.propagator.forward_pass(sampled_inputs)
        return sampled_outputs

    def samples_to_range(self, sampled_outputs):
        num_outputs = sampled_outputs.shape[-1]
        output_range = np.empty((num_outputs, 2))
        output_range[:,1] = np.max(sampled_outputs, axis=0)
        output_range[:,0] = np.min(sampled_outputs, axis=0)
        return output_range

    def get_exact_output_range(self, input_range):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range = self.samples_to_range(sampled_outputs)
        return output_range

# class SDPAnalysis(Analysis):

#     def get_output_range(self, input_range, verbose=False):
#         output_range = robust_sdp(net=model, input_range=input_range, verbose=verbose, viz=False)
#         return output_range

# class JuliaAnalysis(Analysis):

#     def get_output_range(self, input_range, verbose=False):
#         output_range = julia_output_range(net=model, input_range=input_range)
#         return output_range

# def experiment():
#     from partition.xiang import model_xiang_2020_robot_arm
#     import numpy as np
#     from partition.Partition import *
#     from partition.Analysis import *
#     torch_model = model_xiang_2020_robot_arm()
#     input_range = np.array([ # (num_inputs, 2)
#                       [np.pi/3, 2*np.pi/3], # x0min, x0max
#                       [np.pi/3, 2*np.pi/3] # x1min, x1max
#     ])

#     partitioner = "Uniform"
#     propagator = "IBP"
#     partitioner_hyperparams = {}
#     propagator_hyperparams = {}

#     df = pd.DataFrame()
#     for i in [1,2,4,10,20]:
#         partitioner_hyperparams["num_partitions"] = i

#         analyzer = Analyzer(torch_model)
#         analyzer.partitioner = partitioner_dict[partitioner](**partitioner_hyperparams)
#         analyzer.propagator = propagator_dict[propagator](**propagator_hyperparams)
#         output_range, analyzer_info = analyzer.get_output_range(input_range)

#         df.append()

if __name__ == '__main__':
    from partition.xiang import model_xiang_2020_robot_arm
    import numpy as np
    from partition.Partition import *
    torch_model = model_xiang_2020_robot_arm()
    input_range = np.array([ # (num_inputs, 2)
                      [np.pi/3, 2*np.pi/3], # x0min, x0max
                      [np.pi/3, 2*np.pi/3] # x1min, x1max
    ])
    partitioner = "Uniform"
    propagator = "IBP"
    partitioner_hyperparams = {}
    propagator_hyperparams = {}

    partitioner_dict = {
        "None": NoPartitioner,
        "Uniform": UniformPartitioner,
    }
    propagator_dict = {
        "IBP": IBPPropagator,
    }

    analyzer = Analyzer(torch_model)
    analyzer.partitioner = partitioner_dict[partitioner](**partitioner_hyperparams)
    analyzer.propagator = propagator_dict[propagator](**propagator_hyperparams)
    output_range, analyzer_info = analyzer.get_output_range(input_range)
    analyzer.visualize(input_range, output_range, **analyzer_info)
    print(output_range)
    print("done.")

