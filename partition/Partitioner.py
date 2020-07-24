import numpy as np
from itertools import product
from partition.xiang import sect, bisect
from partition.object_boundary import getboundary
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from partition.network_utils import get_sampled_outputs, samples_to_range
import imageio
import os

class Partitioner():
    def __init__(self):
        return

    def get_output_range(self):
        raise NotImplementedError

class NoPartitioner(Partitioner):
    def __init__(self):
        Partitioner.__init__(self)

    def get_output_range(self, input_range, propagator):
        output_range, info = propagator.get_output_range(input_range)
        return output_range, info

class UniformPartitioner(Partitioner):
    def __init__(self, num_partitions=16):
        Partitioner.__init__(self)
        self.num_partitions = num_partitions

    def get_output_range(self, input_range, propagator, num_partitions=None):
        info = {}
        num_propagator_calls = 0

        input_shape = input_range.shape[:-1]
        if num_partitions is None:
            num_partitions = np.ones(input_shape, dtype=int)
            if isinstance(self.num_partitions, np.ndarray) and input_shape == self.num_partitions.shape:
                num_partitions = self.num_partitions
            elif len(input_shape) > 1:
                num_partitions[0,0] = self.num_partitions
            else:
                num_partitions *= self.num_partitions
        slope = np.divide((input_range[...,1] - input_range[...,0]), num_partitions)
        
        ranges = []
        output_range = None
        
        for element in product(*[range(num) for num in num_partitions.flatten()]):
            element_ = np.array(element).reshape(input_shape)
            input_range_ = np.empty_like(input_range)
            input_range_[...,0] = input_range[...,0]+np.multiply(element_, slope)
            input_range_[...,1] = input_range[...,0]+np.multiply(element_+1, slope)
            output_range_, info_ = propagator.get_output_range(input_range_)
            num_propagator_calls += 1
            
            if output_range is None:
                output_range = np.empty(output_range_.shape)
                output_range[:,0] = np.inf
                output_range[:,1] = -np.inf

            tmp = np.dstack([output_range, output_range_])
            output_range[:,1] = np.max(tmp[:,1,:], axis=1)
            output_range[:,0] = np.min(tmp[:,0,:], axis=1)
            
            ranges.append((input_range_, output_range_))

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)

        return output_range, info

class SimGuidedPartitioner(Partitioner):
    def __init__(self, num_simulations=1000, tolerance_eps=0.01, interior_condition="linf", make_animation=False, show_animation=False):
        Partitioner.__init__(self)
        self.num_simulations = num_simulations
        self.tolerance_eps = tolerance_eps
        self.interior_condition = interior_condition
        self.make_animation = make_animation or show_animation
        self.show_animation = show_animation

    def get_sampled_outputs(self, input_range, propagator, N=1000):
        return get_sampled_outputs(input_range, propagator, N=N)

    def samples_to_range(self, sampled_outputs):
        return samples_to_range(sampled_outputs)

    def grab_from_M(self, M, output_range_sim):
        input_range_, output_range_ = M.pop(0) 
        return input_range_, output_range_

    def check_if_partition_within_sim_bnds(self, output_range, output_range_sim):
        if self.interior_condition == "linf":
            # Check if output_range's linf ball is within
            # output_range_sim's linf ball
            inside = np.all((output_range_sim[...,0] - output_range[...,0]) <= 0) and \
                        np.all((output_range_sim[...,1] - output_range[...,1]) >= 0)
        elif self.interior_condition == "lower_bnds":
            # Check if output_range's lower bnds are above each of
            # output_range_sim's lower bnds
            inside = np.all((output_range_sim[...,0] - output_range[...,0]) <= 0)
        elif self.interior_condition == "convex_hull":
            # Check if the rectangle of output_range lies within the
            # convex hull of the sim pts
            ndim = output_range.shape[0]
            pts = np.empty((ndim**2, ndim+1))
            pts[:,-1] = 1.
            for i, pt in enumerate(product(*output_range)):
                pts[i,:-1] = pt
            inside = np.all(np.matmul(self.sim_convex_hull.equations, pts.T) <= 0)
        else:
            raise NotImplementedError
        return inside

    def setup_visualization(self, input_range, output_range, output_range_exact, propagator):
        self.animate_fig, self.animate_axes = plt.subplots(1,2)
        num_input_dimensions_to_plot = 2
        input_shape = input_range.shape[:-1]
        lengths = input_range[...,1].flatten() - input_range[...,0].flatten()
        flat_dims = np.argpartition(lengths, -num_input_dimensions_to_plot)[-num_input_dimensions_to_plot:]
        flat_dims.sort()
        input_dims = [np.unravel_index(flat_dim, input_range.shape[:-1]) for flat_dim in flat_dims]
        self.input_dims_ = tuple([tuple([input_dims[j][i] for j in range(len(input_dims))]) for i in range(len(input_dims[0]))])

        scale = 0.05
        x_off = max((input_range[input_dims[0]+(1,)] - input_range[input_dims[0]+(0,)])*(scale), 1e-5)
        y_off = max((input_range[input_dims[1]+(1,)] - input_range[input_dims[1]+(0,)])*(scale), 1e-5)
        self.animate_axes[0].set_xlim(input_range[input_dims[0]+(0,)] - x_off, input_range[input_dims[0]+(1,)]+x_off)
        self.animate_axes[0].set_ylim(input_range[input_dims[1]+(0,)] - y_off, input_range[input_dims[1]+(1,)]+y_off)
        self.animate_axes[0].set_xlabel("Input Dim: {}".format(input_dims[0]))
        self.animate_axes[0].set_ylabel("Input Dim: {}".format(input_dims[1]))

        # Make a rectangle for the Exact boundaries
        sampled_outputs = self.get_sampled_outputs(input_range, propagator)
        self.animate_axes[1].scatter(sampled_outputs[:,0], sampled_outputs[:,1], c='k', marker='.', zorder=2)
        
        # Exact range
        color = 'black'
        linewidth = 2
        if self.interior_condition == "linf":
            output_range_exact = self.samples_to_range(sampled_outputs)
            rect = Rectangle(output_range_exact[:2,0], output_range_exact[0,1]-output_range_exact[0,0], output_range_exact[1,1]-output_range_exact[1,0],
                            fc='none', linewidth=linewidth,edgecolor=color)
            self.animate_axes[1].add_patch(rect)
            self.default_patch = [rect]
            self.default_lines = []
        elif self.interior_condition == "lower_bnds":
            output_range_exact = self.samples_to_range(sampled_outputs)
            line1 = self.animate_axes[1].axhline(output_range_exact[input_dims[1]+(0,)], linewidth=linewidth,color=color)
            line2 = self.animate_axes[1].axvline(output_range_exact[input_dims[0]+(0,)], linewidth=linewidth,color=color)
            self.default_patch = []
            self.default_lines = [line1, line2]
        elif self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull
            hull = ConvexHull(sampled_outputs)
            line = self.animate_axes[1].plot(sampled_outputs[hull.vertices,0], sampled_outputs[hull.vertices,1], color=color, linewidth=linewidth)
            self.default_patch = []
            self.default_lines = line
        else:
            raise NotImplementedError

    def visualize(self, M, interior_M, u_e, iteration=0):
        self.animate_axes[0].patches = []
        self.animate_axes[1].patches = self.default_patch.copy()
        self.animate_axes[0].lines = []
        self.animate_axes[1].lines = self.default_lines.copy()
        input_dims_ = self.input_dims_

        # Rectangles that might still be outside the sim pts
        for (input_range_, output_range_) in M:
            rect = Rectangle(output_range_[:2,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                    fc='none', linewidth=1,edgecolor='m')
            self.animate_axes[1].add_patch(rect)

            input_range__ = input_range_[input_dims_]
            rect = Rectangle(input_range__[:,0], input_range__[0,1]-input_range__[0,0], input_range__[1,1]-input_range__[1,0],
                    fc='none', linewidth=1,edgecolor='m')
            self.animate_axes[0].add_patch(rect)

        # Rectangles that are within the sim pts
        for (input_range_, output_range_) in interior_M:
            rect = Rectangle(output_range_[:2,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                    fc='none', linewidth=1,edgecolor='b')
            self.animate_axes[1].add_patch(rect)

            input_range__ = input_range_[input_dims_]
            rect = Rectangle(input_range__[:,0], input_range__[0,1]-input_range__[0,0], input_range__[1,1]-input_range__[1,0],
                    fc='none', linewidth=1,edgecolor='b')
            self.animate_axes[0].add_patch(rect)

        linewidth = 1
        color = 'green'
        if self.interior_condition == "linf":
            # Make a rectangle for the estimated boundaries
            output_range_estimate = self.squash_down_to_one_range(u_e, M)
            rect = Rectangle(output_range_estimate[:2,0], output_range_estimate[0,1]-output_range_estimate[0,0], output_range_estimate[1,1]-output_range_estimate[1,0],
                            fc='none', linewidth=linewidth,edgecolor=color)
            self.animate_axes[1].add_patch(rect)
        elif self.interior_condition == "lower_bnds":
            output_range_estimate = self.squash_down_to_one_range(u_e, M)
            self.animate_axes[1].axhline(output_range_estimate[1,0],
                linewidth=linewidth,color=color)
            self.animate_axes[1].axvline(output_range_estimate[0,0],
                linewidth=linewidth,color=color)
        elif self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull
            hull = self.squash_down_to_convex_hull(M+interior_M)
            self.animate_axes[1].plot(
                np.append(hull.points[hull.vertices,0], hull.points[hull.vertices[0],0]),
                np.append(hull.points[hull.vertices,1], hull.points[hull.vertices[0],1]),
                color=color, linewidth=linewidth)
        else:
            raise NotImplementedError

        if self.show_animation:
            plt.pause(0.01)

        animation_save_dir = "{}/results/tmp/".format(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(animation_save_dir, exist_ok=True)
        plt.savefig(animation_save_dir+"tmp_{}.png".format(str(iteration).zfill(6)))

    def compile_animation(self, iteration):
        animation_save_dir = "{}/results/tmp/".format(os.path.dirname(os.path.abspath(__file__)))
        filenames = [animation_save_dir+"tmp_{}.png".format(str(i).zfill(6)) for i in range(iteration)]
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
            if filename == filenames[-1]:
                for i in range(10):
                    images.append(imageio.imread(filename))
            os.remove(filename)

        # Save the gif in a new animations sub-folder
        animation_filename = "tmp.gif"
        animation_save_dir = "{}/results/animations/".format(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(animation_save_dir, exist_ok=True)
        animation_filename = animation_save_dir+animation_filename
        imageio.mimsave(animation_filename, images)


    def squash_down_to_one_range(self, u_e, M):
        u_e_ = u_e.copy()
        if len(M) > 0:
            # Squash all of M down to one range
            M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
            M_range = np.empty_like(u_e_)
            M_range[:,1] = np.max(M_numpy[:,1,:], axis=1)
            M_range[:,0] = np.min(M_numpy[:,0,:], axis=1)
        
            # Combine M (remaining ranges) with u_e (interior ranges)
            tmp = np.dstack([u_e_, M_range])
            u_e_[:,1] = np.max(tmp[:,1,:], axis=1)
            u_e_[:,0] = np.min(tmp[:,0,:], axis=1)
        return u_e_

    def squash_down_to_convex_hull(self, all_M):
        from scipy.spatial import ConvexHull
        ndim = all_M[0][0].shape[0]
        pts = np.empty((len(all_M)*ndim**2, ndim))
        i = 0
        for (input_range, output_range) in all_M:
            for pt in product(*output_range):
                pts[i,:] = pt
                i += 1
        hull = ConvexHull(pts)
        return hull

    def get_output_range(self, input_range, propagator):

        # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = 'max'
        input_shape = input_range.shape[:-1]
        info = {}

        num_propagator_calls = 0

        # Get initial output reachable set (Line 3)
        output_range, _ = propagator.get_output_range(input_range)
        num_propagator_calls += 1

        M = [(input_range, output_range)] # (Line 4)
        interior_M = []
        
        # Run N simulations (i.e., randomly sample N pts from input range --> query NN --> get N output pts)
        # (Line 5)
        sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (self.num_simulations,)+input_shape)
        sampled_outputs = propagator.forward_pass(sampled_inputs)

        if self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull
            self.sim_convex_hull = ConvexHull(sampled_outputs)

        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        output_range_sim = np.empty(sampled_outputs.shape[1:]+(2,))
        output_range_sim[:,1] = np.max(sampled_outputs, axis=0)
        output_range_sim[:,0] = np.min(sampled_outputs, axis=0)

        if self.make_animation:
            self.setup_visualization(input_range, output_range, output_range_sim, propagator)
        
        u_e = np.empty_like(output_range_sim)
        u_e[:,0] = np.inf
        u_e[:,1] = -np.inf
        iteration = 0
        while len(M) != 0:
            input_range_, output_range_ = self.grab_from_M(M, output_range_sim) # (Line 9)

            if self.check_if_partition_within_sim_bnds(output_range_, output_range_sim):
                # Line 11
                tmp = np.dstack([u_e, output_range_])
                u_e[:,1] = np.max(tmp[:,1,:], axis=1)
                u_e[:,0] = np.min(tmp[:,0,:], axis=1)
                interior_M.append((input_range_, output_range_))
            else:
                # Line 14
                if np.max(input_range_[...,1] - input_range_[...,0]) > self.tolerance_eps:
                    # Line 15
                    input_ranges_ = sect(input_range_, 2, select=sect_method)
                    # Lines 16-17
                    for input_range_ in input_ranges_:
                        output_range_, _ = propagator.get_output_range(input_range_)
                        num_propagator_calls += 1
                        M.append((input_range_, output_range_)) # Line 18
                else: # Lines 19-20
                    M.append((input_range_, output_range_))
                    break
            if self.make_animation:
                self.visualize(M, interior_M, u_e, iteration)
            iteration += 1

        # Line 24
        u_e = self.squash_down_to_one_range(u_e, M)

        info["all_partitions"] = M+interior_M
        info["exterior_partitions"] = M
        info["interior_partitions"] = interior_M
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = len(M) + len(interior_M)

        if self.make_animation:
            self.compile_animation(iteration)
        
        return u_e, info

class GreedySimGuidedPartitioner(SimGuidedPartitioner):
    def __init__(self, num_simulations=1000, tolerance_eps=0.01, interior_condition="linf", make_animation=False, show_animation=False):
        SimGuidedPartitioner.__init__(self, num_simulations=num_simulations, tolerance_eps=tolerance_eps, interior_condition=interior_condition, make_animation=make_animation, show_animation=show_animation)

    def grab_from_M(self, M, output_range_sim):
        # TODO(MFE): make this aware of interior_condition!!!
        if len(M) == 1:
            input_range_, output_range_ = M.pop(0)
        else:
            if self.interior_condition == "linf":
                # look thru all output_range_s and see which are furthest from sim output range
                M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
                z = np.empty_like(M_numpy)
                z[:,0,:] = (output_range_sim[:,0] - M_numpy[:,0,:].T).T
                z[:,1,:] = (M_numpy[:,1,:].T - output_range_sim[:,1]).T
                worst_index = np.unravel_index(z.argmax(), shape=z.shape)
                worst_M_index = worst_index[-1]
                input_range_, output_range_ = M.pop(worst_M_index)
            elif self.interior_condition == "lower_bnds":
                # look thru all lower bnds and see which are furthest from sim lower bnds
                M_numpy = np.dstack([output_range_[:,0] for (_, output_range_) in M])
                z = np.empty_like(M_numpy)
                z = (output_range_sim[:,0] - M_numpy.T).T
                worst_index = np.unravel_index(z.argmax(), shape=z.shape)
                worst_M_index = worst_index[-1]
                input_range_, output_range_ = M.pop(worst_M_index)
            elif self.interior_condition == "convex_hull":
                raise NotImplementedError
            else:
                raise NotImplementedError

        return input_range_, output_range_

class BoundarySimGuidedPartitioner(Partitioner):
    def __init__(self, N=20, tolerance_eps=0.01):
        Partitioner.__init__(self)
        self.N= N
        self.tolerance_eps= tolerance_eps

    def get_output_range(self, input_range, propagator):
        input_shape = input_range.shape[:-1]
        num_propagator_calls = 0

        interior_M = []
        M=[]
        M_inside=[]
        info = {}
        boundary_shape="convex"
        sect_method = 'max'
        num_inputs = input_range.shape[0]
        num_partitions = (self.N)*np.ones((num_inputs,), dtype=int)
        slope = np.divide((input_range[:,1] - input_range[:,0]), num_partitions)
        sampling_cell = np.empty(((self.N)**2,num_inputs,2))
        output_range_ = np.empty(((self.N)**2,num_inputs,2))
        sampled_output_ = np.empty(((self.N)**2,num_inputs))
        sampled_output_boundary_ = np.empty(((self.N)**2))

   
        input_range_ = np.empty((self.N**2,num_inputs,2))

        element=-1
        for idx in product(*[range(num) for num in num_partitions]):
            element = element+1
            input_range_[element,:,0] = input_range[:,0]+np.multiply(idx, slope)
            input_range_[element,:,1] = input_range[:,0]+np.multiply(np.array(idx)+1, slope)
            sampled_input_= np.random.uniform(input_range_[element,:,0], input_range_[element,:,1], (1,num_inputs,))
            sampled_output_[element,:] =  propagator.forward_pass( propagator.forward_pass(sampled_input_))
            sampled_output_boundary_[element] =0;
           # output_range_[element,:,:],_= propagator.get_output_range(input_range_[element,:,:])
           # num_propagator_calls += 1

        if boundary_shape=="convex":
            boundary_points= getboundary(sampled_output_,0.0)
        else:
            boundary_points= getboundary(sampled_output_,0.4)

        for i in range(self.N**2):
            if sampled_output_[i,:] in boundary_points:
                sampled_output_boundary_[i]=1
                propagator.get_output_range(input_range_[i,:,:])
                num_propagator_calls += 1
                M.append((input_range_[i,:,:], output_range_[i,:,:], sampled_output_boundary_[i]))# (Line 4)  

            else:
                M_inside.append((input_range_[i,:,:]))    

        

        output_range_sim = np.empty_like(output_range_[0,:,:])
        output_range_sim[:,0] = np.min(sampled_output_, axis=0)
        output_range_sim[:,1] = np.max(sampled_output_, axis=0)
            
        u_e = np.empty_like(output_range_sim)
        u_e[:,0] = np.inf
        u_e[:,1] = -np.inf


        M_e = np.empty_like(output_range_sim)
        M_e[:,0] = np.inf
        M_e[:,1] = -np.inf

        while len(M) != 0:
            input_range_, output_range_,sampled_output_boundary_ = M.pop(0) # Line 9
            if np.all((output_range_sim[:,0] - output_range_[:,0]) <= 0) and \
            np.all((output_range_sim[:,1] - output_range_[:,1]) >= 0) or sampled_output_boundary_==0:
            # Line 11
                tmp = np.dstack([u_e, output_range_])
                u_e[:,1] = np.max(tmp[:,1,:], axis=1)
                u_e[:,0] = np.min(tmp[:,0,:], axis=1)
                interior_M.append((input_range_, output_range_))
            else:
            # Line 14
                if np.max(input_range_[:,1] - input_range_[:,0]) > self.tolerance_eps:
                # Line 15
                    input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                    for input_range_ in input_ranges_:
                        output_range_,_= propagator.get_output_range(input_range_)
                        num_propagator_calls += 1
   
                        M.append((input_range_, output_range_,sampled_output_boundary_)) # Line 18
                else: # Lines 19-20
                    interior_M.append((input_range_, output_range_))
                    break


        if len(M) > 0:
            M_numpy = np.dstack([output_range_ for (_, output_range_,_) in M])
            M_range = np.empty_like(u_e)
            M_range[:,1] = np.max(M_numpy[:,1,:], axis=1)
            M_range[:,0] = np.min(M_numpy[:,0,:], axis=1)
    
            tmp1 = np.dstack([u_e, M_range])
            u_e[:,1] = np.max(tmp1[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp1[:,0,:], axis=1)
        M1=[]
        if len(M_inside) > 0:
            for (input_range) in M_inside:
                output_range_,_= propagator.get_output_range(input_range_)
                num_propagator_calls += 1
                M2 = np.dstack([M_e, output_range_])
                M_e[:,1] = np.max(M2[:,1,:], axis=1)
                M_e[:,0] = np.min(M2[:,0,:], axis=1)

            tmp2 = np.dstack([u_e, M_e])
            u_e[:,1] = np.max(tmp2[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp2[:,0,:], axis=1)
    

     
        info["all_partitions"] = M+interior_M
        info["exterior_partitions"] = M
        info["interior_partitions"] = interior_M
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = len(M) + len(interior_M)
        return u_e, info

class AdaptiveSimGuidedPartitioner(Partitioner):
    def __init__(self, num_simulations=100, k_NN =1, tolerance_eps=0.01, tolerance_expanding_step=0.001,tolerance_range=0.05):
        Partitioner.__init__(self)
        self.num_simulations= num_simulations
        self.tolerance_eps= tolerance_eps
        #self.tolerance_step= tolerance_expanding_step
        #self.k_NN= k_NN
        #self.tolerance_range= tolerance_range

   # def grab_from_M(self, M, output_range_sim):
       # input_range_, output_range_ = M.pop(0) 
        #return input_range_, output_range_
    def set_delta_step(self,range_,center_value, idx, stage=1):
        c=0.65
        if stage==1:
            k=-1
            num_inputs=idx
            delta_step =np.ones((num_inputs*2,1))
            output_range = range_
            distance = np.empty((len(range_)*2))
            output_center = center_value
            output_bounding_box = [[output_range[0,0],output_range[1,0]],[output_range[0,1],output_range[1,0]],
            [output_range[0,1],output_range[1,1]],[output_range[0,0],output_range[1,1]]]
            for (i,j) in output_bounding_box:
                k=k+1
                distance[k] = np.sqrt((center_value[0]-i)**2 + (center_value[1]-j)**2 )
            min_distance = np.min(distance)
            for i in range(len(range_)*2):
                if i % 2 == 0: 
                    delta_step[i]= -c*min_distance/np.max(output_range[:,1]-output_range[:,0])
                else:
                    delta_step[i]= c*min_distance/np.max(output_range[:,1]-output_range[:,0])
        else:
            input_range = range_
            diff_rolled = center_value
            num_inputs = len(range_)
            delta_step =np.zeros((num_inputs*2,1))
            if idx % 2 == 0: 
                delta_step[idx]= -c*diff_rolled[idx]/(np.max(input_range[:,0]))
            else:
                delta_step[idx]= c*diff_rolled[idx]/(np.max(input_range[:,1]))
        return delta_step

    def get_output_range(self, input_range, propagator):
        info = {}

       # tolerance_eps = 0.05

       ### old param tolerance_range = 0.01, c= 0.8, 2/3
        tolerance_step=0.0001
        tolerance_range=0.005
        num_propagator_calls=0
        k_NN=1
        N = 1000
        interior_M = []

        sect_method = 'max'
        num_inputs = input_range.shape[0]
        output_ = propagator.forward_pass(input_range[0,:])

        num_outputs=len(output_)
        output_range_ = np.empty((N,num_outputs,2))
        sampled_output_ = np.empty((N,num_outputs))
        sampled_input_= np.empty((N,num_inputs))
        element=-1

        sampled_input_= np.random.uniform(input_range[:,0],input_range[:,1], (N,num_inputs,))

        sampled_output_ = propagator.forward_pass(sampled_input_)
    
        output_range_sim = np.empty((num_outputs, 2))
        output_range_sim[:,0] = np.min(sampled_output_, axis=0)
        output_range_sim[:,1] = np.max(sampled_output_, axis=0)


  
        sampled_output_center=output_range_sim.mean(axis=1)


    
        input_range_initial = np.empty((num_inputs, 2))

        kdt = KDTree(sampled_output_, leaf_size=30, metric='euclidean')
        center_NN=kdt.query(sampled_output_center.reshape(1,-1), k=k_NN, return_distance=False)
    


        input_range_new=np.empty((num_inputs, 2))
        input_range_new[:,0] = np.min(sampled_input_[center_NN,:], axis=0)
        input_range_new[:,1] = np.max(sampled_input_[center_NN,:], axis=0)
 

        count =0
        M=[]
        delta_step = self.set_delta_step(output_range_sim,sampled_output_center, num_inputs, stage=1)
        prev_range =np.inf
        terminating_condition=False
        while terminating_condition==False:
 
            input_range_new= input_range_new+delta_step.reshape((num_inputs, 2))
            output_range_new, _ = propagator.get_output_range(input_range_new)
            num_propagator_calls += 1
            if np.all((output_range_sim[:,0] - output_range_new[:,0]) <= 0) and \
            np.all((output_range_sim[:,1] - output_range_new[:,1]) >= 0):
                terminating_condition==False
            else:
                input_range_new= input_range_new-delta_step.reshape((num_inputs, 2))
                delta_step=1*delta_step/2

                if np.max(abs(delta_step))<tolerance_step:

                    diff=(input_range-input_range_new)

                    max_range= (np.max(abs(diff)))
                    if max_range<tolerance_range:# or abs(max_range-prev_range)<tol
                        break
                    else:
                        tol = 5
                        diff_rolled= np.concatenate(abs(diff))
                        if abs(max_range-prev_range)<tol:
                            count  = count+1
                        idx= num_inputs*2-1-count;
                        prev_range= np.inf;
                        if idx<0:
                            break
                        else:
                            prev_range = max_range
                            delta_step =self.set_delta_step(input_range, diff_rolled,idx, stage=2)  
                            if np.max(abs(delta_step))<tolerance_step:
                                break

    
                        
        if input_range[0,0] !=input_range_new[0,0]: 
   
            input_range_ = np.array([[input_range[0,0] ,input_range_new[0,0]],[input_range[1,0], input_range[1,1]]])
            output_range_,_ = propagator.get_output_range(input_range_)
            num_propagator_calls += 1
    
            M.append((input_range_,output_range_)) 
      

        if[input_range_new[0,1]!=input_range[0,1]]:

                           #### approch2 only
            input_range_ = np.array([[input_range_new[0,1],input_range[0,1]],[input_range[1,0],input_range[1,1]]])
            output_range_,_ = propagator.get_output_range(input_range_)
            num_propagator_calls += 1
   
            M.append((input_range_,output_range_))
                       
                        
        if[input_range_new[1,1]!=input_range[1,1]]:

            input_range_ = np.array([[input_range_new[0,0],input_range_new[0,1]],[input_range_new[1,1],input_range[1,1]]])
            output_range_,_ = propagator.get_output_range(input_range_)
            num_propagator_calls += 1

            M.append((input_range_,output_range_))
                           

        if[input_range_new[1,0]!=input_range[1,0]]:
                        ### common partition between two approaches

            input_range_ = np.array([[input_range_new[0,0],input_range_new[0,1]],[input_range[1,0],input_range_new[1,0]]])
            output_range_,_ = propagator.get_output_range(input_range_)
            num_propagator_calls += 1

            M.append((input_range_,output_range_)) 
        u_e = np.empty((num_outputs, 2))
        u_e[:,0] = np.inf
        u_e[:,1] = -np.inf

        while len(M) != 0:
            input_range_, output_range_ = M.pop(0) # Line 9
            if np.all((output_range_sim[:,0] - output_range_[:,0]) <= 0) and \
               np.all((output_range_sim[:,1] - output_range_[:,1]) >= 0):
            # Line 11
                tmp = np.dstack([u_e, output_range_])
                u_e[:,1] = np.max(tmp[:,1,:], axis=1)
                u_e[:,0] = np.min(tmp[:,0,:], axis=1)
                interior_M.append((input_range_, output_range_))
            else:
            # Line 14
                if np.max(input_range_[:,1] - input_range_[:,0]) > self.tolerance_eps:
                # Line 15
                    input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                    for input_range_ in input_ranges_:
                        output_range_,_ = propagator.get_output_range(input_range_)
                        num_propagator_calls += 1
   
                        M.append((input_range_, output_range_)) # Line 18
                else: # Lines 19-20
                    interior_M.append((input_range_, output_range_))
                    break

        M.append((input_range_new,output_range_new))

    # Line 24
        if len(M) > 0:
        # Squash all of M down to one range
            M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
            M_range = np.empty((num_outputs, 2))
            M_range[:,1] = np.max(M_numpy[:,1,:], axis=1)
            M_range[:,0] = np.min(M_numpy[:,0,:], axis=1)
    
        # Combine M (remaining ranges) with u_e (interior ranges)
            tmp = np.dstack([u_e, M_range])
            u_e[:,1] = np.max(tmp[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp[:,0,:], axis=1)
        

        info["all_partitions"] = M+interior_M
        info["exterior_partitions"] = M
        info["interior_partitions"] = interior_M
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = len(M) + len(interior_M)
        
        return u_e, info
