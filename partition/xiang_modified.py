import torch, random
import numpy as np
from partition.kd_tree import *
from partition.models import *
from crown_ibp.bound_layers import BoundSequential
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import product
from partition.object_boundary import getboundary
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
def polygon_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def simulation_guided_partition_gh_plus(model, input_range, num_outputs, viz=False, boundary_shape="convex",bound_method="ibp"):
    tolerance_eps = 0.01
    tolerance_step=0.001
    tolerance_range=0.05
    num_calls_propagator=0
    k_NN=1
    N = 20
    interior_M = []

    sect_method = 'max'
    num_inputs = input_range.shape[0]

    output_range_no_partition = np.empty((num_outputs, 2))

    output_range_no_partition= get_output_range(model, input_range, num_outputs, bound_method=bound_method)
    #output_range_no_partition[:,0] = np.min(output_range, axis=0)
    #output_range_no_partition[:,1] = np.max(output_range, axis=0)
    #print(output_range_no_partition)
    num_partitions = N*np.ones((num_inputs,), dtype=int)
    slope = np.divide((input_range[:,1] - input_range[:,0]), num_partitions)
    sampling_cell = np.empty((N**2,num_inputs,2))
    output_range_ = np.empty((N**2,num_inputs,2))
    #sampled_input_ = np.empty((N,1))
    sampled_output_ = np.empty((N**2,num_inputs))

    sampled_output_boundary_ = np.empty((N**2))


    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf
    M=[]
    element=-1

    for idx in product(*[range(num) for num in num_partitions]):
       element = element+1
       sampling_cell[element,:,0] = input_range[:,0]+np.multiply(idx, slope)
       sampling_cell[element,:,1] = input_range[:,0]+np.multiply(np.array(idx)+1, slope)
       sampled_input_= np.random.uniform(sampling_cell[element,:,0], sampling_cell[element,:,1], (1,num_inputs,))
       # sampled_input_[element,0]= np.random.uniform(input_range_[element,:,0], input_range_[element,:,1], (1,num_inputs,))
     

       sampled_output_[element,:] = model(torch.Tensor(sampled_input_), method_opt=None).data.numpy()
       sampled_output_boundary_[element] =0;
       #output_range_[element,:,:]= get_output_range(model, input_range_[element,:,:], num_outputs, bound_method=bound_method)
    
    output_range_sim = np.empty((num_outputs, 2))
    #print(sampled_output_)
    output_range_sim[:,0] = np.min(sampled_output_, axis=0)
    output_range_sim[:,1] = np.max(sampled_output_, axis=0)


    #sampled_output_center=sampled_output_.mean(axis=0)



    if boundary_shape=="convex":
       #convex_hull = ConvexHull(sampled_output_,)
    #print(convex_hull.vertices)
       #boundary_points = sampled_output_[convex_hull.vertices,:]
       boundary_points= getboundary(sampled_output_,0.0)
       #sampled_output_center=boundary_points.mean(axis=0)
      # print('Runnnig partitioning with convex boundary...')
       #print(sampled_output_center)
    else:
       boundary_points= getboundary(sampled_output_,0.4)
      # print('Runnnig partitioning with concave boundary...')

   # print(convex_hull_boundary[:,0],convex_hull_boundary[:,1])
  
    sampled_output_center=boundary_points.mean(axis=0)


    
    input_range_initial = np.empty((num_outputs, 2))

    kdt = KDTree(sampled_output_, leaf_size=30, metric='euclidean')
    center_NN=kdt.query(sampled_output_center.reshape(1,-1), k=k_NN, return_distance=False)
    #print(center_NN)
    viz=0
    if viz:
        fig, axes = plt.subplots(1,2)
        plt.plot(sampled_output_[:,0],sampled_output_[:,1],'ob')
        plt.plot(boundary_points[:,0],boundary_points[:,1],'og')
        plt.plot(sampled_output_[center_NN,0], sampled_output_[center_NN,1],'or')
        plt.plot(sampled_output_center[0], sampled_output_center[1],'om') 
   # plt.show()   
    #plt.show()9
    NN_range_x=[]
    NN_range_y=[]

    for i in center_NN:
        NN_range_x.append( sampling_cell[i,0,:])
        NN_range_y.append( sampling_cell[i,1,:])

    input_range_initial[0,0] = np.min(NN_range_x)
    input_range_initial[1,0] = np.min(NN_range_y)
    input_range_initial[0,1] = np.max(NN_range_x)
    input_range_initial[1,1] = np.max(NN_range_y)
    
    terminating_condition=False
    output_range_new=np.empty((num_outputs, 2))

    input_range_new=np.empty((num_outputs, 2))
    input_range_new[0,0] = input_range_initial[0,0]
    input_range_new[0,1] = input_range_initial[0,1]
    input_range_new[1,0]  = input_range_initial[1,0]
    input_range_new[1,1] = input_range_initial[1,1]

 
    output_range = get_output_range(model, input_range_initial, num_outputs, bound_method=bound_method)

    count =0
    M=[]
    print(sampled_output_center)
    delta_step = set_delta_step(output_range_sim,sampled_output_center, num_inputs, stage=1)
    prev_range =np.inf

    while terminating_condition==False:
 
        #print(delta_step.reshape((num_inputs, 2)))
        input_range_new= input_range_new+delta_step.reshape((num_inputs, 2))
        output_range_new= get_output_range(model, input_range_new, num_outputs, bound_method=bound_method)
        num_calls_propagator += 1
        if np.all((output_range_sim[:,0] - output_range_new[:,0]) <= 0) and \
        np.all((output_range_sim[:,1] - output_range_new[:,1]) >= 0):
            terminating_condition=False
            #print('delta does not change')

        else:
            input_range_new= input_range_new-delta_step.reshape((num_inputs, 2))
            delta_step=delta_step/2

            #print('delta is decreasing')
            tol = 5
            if np.max(abs(delta_step))<tolerance_step:
               # break
                #print('final input range new')

                diff=(input_range-input_range_new)

                max_range= (np.max(abs(diff)))
                if max_range<tolerance_range:# or abs(max_range-prev_range)<tol
       
                    break
                else:
                #print(np.concatenate(diff))
                    #sort_index = np.argsort(np.concatenate(abs(diff)))
                    print("****************next stage starts..")
                    #print(sort_index[num_inputs*2-1])
                    diff_rolled= np.concatenate(abs(diff))
                    print(max_range)
                    print(prev_range)
                    if abs(max_range-prev_range)<tol:
                        count  = count+1
                    idx= num_inputs*2-1-count;
                    prev_range= np.inf;
                    if idx<0:
                        print('All boxes are explored')

       

                        break
                    else:
                        prev_range = max_range
                        delta_step =set_delta_step(input_range, diff_rolled,idx, stage=2)  
                        if np.max(abs(delta_step))<tolerance_step:

                            print('oh here!!!')
                            break

    
                        
                        #range_diff= input_range -input_range_new
    if input_range[0,0] !=input_range_new[0,0]: 

                            ##  approach 2 only
        input_range_ = np.array([[input_range[0,0] ,input_range_new[0,0]],[input_range[1,0], input_range[1,1]]])
        output_range_= get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
        num_calls_propagator += 1
    
        M.append((input_range_,output_range_)) 
      

    if[input_range_new[0,1]!=input_range[0,1]]:

                           #### approch2 only
        input_range_ = np.array([[input_range_new[0,1],input_range[0,1]],[input_range[1,0],input_range[1,1]]])
        output_range_= get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
        num_calls_propagator += 1
   
        M.append((input_range_,output_range_))
                       
                        
    if[input_range_new[1,1]!=input_range[1,1]]:

        input_range_ = np.array([[input_range_new[0,0],input_range_new[0,1]],[input_range_new[1,1],input_range[1,1]]])
        output_range_= get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
        num_calls_propagator += 1

        M.append((input_range_,output_range_))
                           

    if[input_range_new[1,0]!=input_range[1,0]]:
                        ### common partition between two approaches

        input_range_ = np.array([[input_range_new[0,0],input_range_new[0,1]],[input_range[1,0],input_range_new[1,0]]])
        output_range_= get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
        num_calls_propagator += 1

        M.append((input_range_,output_range_)) 
    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf

    while len(M) != 0:
        input_range_, output_range_ = M.pop(0) # Line 9
        print
        if np.all((output_range_sim[:,0] - output_range_[:,0]) <= 0) and \
            np.all((output_range_sim[:,1] - output_range_[:,1]) >= 0):
            # Line 11
            tmp = np.dstack([u_e, output_range_])
            u_e[:,1] = np.max(tmp[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp[:,0,:], axis=1)
            interior_M.append((input_range_, output_range_))
        else:
            # Line 14
            if np.max(input_range_[:,1] - input_range_[:,0]) > tolerance_eps:
                # Line 15
                input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                for input_range_ in input_ranges_:
                    output_range_ = get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
                    num_calls_propagator += 1
   
                    M.append((input_range_, output_range_)) # Line 18
            else: # Lines 19-20
                interior_M.append((input_range_, output_range_))
                print('break_point')
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
    N = 1000

    sampled_inputs = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
    sampled_outputs_eval = model(torch.Tensor(sampled_inputs), method_opt=None).data.numpy()
    sampled_outputs_range = np.empty((num_outputs, 2))
    sampled_outputs_range[:,0] = np.min(sampled_outputs_eval, axis=0)
    sampled_outputs_range[:,1] = np.max(sampled_outputs_eval, axis=0)
    viz=0   
    if viz:
      
        visualize_partitions(0,output_range_no_partition,sampled_outputs_eval, u_e, input_range, sampled_outputs_range, interior_M=ranges,M=None)
    
    # Line 24

 

 

   # boundary_output_partitions= getboundary(M_numpy,0.0)
    #print(polygon_area(M_numpy[0,:],M_numpy[1,:] ))

        rect = Rectangle(input_range[:,0], input_range[0,1]-input_range[0,0], input_range[1,1]-input_range[1,0],fc='none', linewidth=1,edgecolor='r')  
        axes[1].add_patch(rect)
        rect = Rectangle(input_range_initial[:,0], input_range_initial[0,1]-input_range_initial[0,0], input_range_initial[1,1]-input_range_initial[1,0],fc='none', linewidth=3,edgecolor='m')
        axes[1].add_patch(rect)
        rect = Rectangle(input_range_new[:,0], input_range_new[0,1]-input_range_new[0,0], input_range_new[1,1]-input_range_new[1,0],fc='none', linewidth=1,edgecolor='g')
        axes[1].add_patch(rect)
        axes[1].set_xlim(input_range[0,0], input_range[0,1])
        axes[1].set_ylim(input_range[1,0], input_range[1,1])
        plt.show()
    
    print('********* printing results ......')
    print('number of partitions_gh:')
    print(len(M)+len(interior_M))

   # boundary_output_partitions= getboundary(M_numpy,0.0)
    #print(polygon_area(M_numpy[0,:],M_numpy[1,:] ))
    print('bounding box area ratio_gh:')
  
    u_range = u_e[:,1]- u_e[:,0]

    sim_out_range= sampled_outputs_range[:,1]-sampled_outputs_range[:,0]
    print(u_range[0]*u_range[1]/(sim_out_range[0]*sim_out_range[1]))

    print('number of calling propagator_gh:')

    print(num_calls_propagator)
    return u_e





def simulation_guided_partition_gh(model, input_range, num_outputs, viz=False, boundary_shape="convex",bound_method="ibp"):
    # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
    tolerance_eps = 0.01
    sect_method = 'max'
    num_inputs = input_range.shape[0]
    output_range_no_partition = np.empty((num_outputs, 2))

    output_range= get_output_range(model, input_range, num_outputs, bound_method=bound_method)
    output_range_no_partition[:,0] = np.min(output_range, axis=0)
    output_range_no_partition[:,1] = np.max(output_range, axis=0)
    #print(output_range_no_partition)
    N = 10
    num_partitions = N*np.ones((num_inputs,), dtype=int)
    slope = np.divide((input_range[:,1] - input_range[:,0]), num_partitions)
    input_range_ = np.empty((N**2,num_inputs,2))
    output_range_ = np.empty((N**2,num_inputs,2))
    #sampled_input_ = np.empty((N,1))
    sampled_output_ = np.empty((N**2,num_inputs))

    sampled_output_boundary_ = np.empty((N**2))


    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf
    M=[]
    element=-1
    #print(num_partitions)
    for idx in product(*[range(num) for num in num_partitions]):
       element = element+1
      # print(element)
       input_range_[element,:,0] = input_range[:,0]+np.multiply(idx, slope)
       input_range_[element,:,1] = input_range[:,0]+np.multiply(np.array(idx)+1, slope)
       sampled_input_= np.random.uniform(input_range_[element,:,0], input_range_[element,:,1], (1,num_inputs,))

       # sampled_input_[element,0]= np.random.uniform(input_range_[element,:,0], input_range_[element,:,1], (1,num_inputs,))
     

       sampled_output_[element,:] = model(torch.Tensor(sampled_input_), method_opt=None).data.numpy()
       sampled_output_boundary_[element] =0;
       output_range_[element,:,:]= get_output_range(model, input_range_[element,:,:], num_outputs, bound_method=bound_method)
       if boundary_shape=="convex":
       #convex_hull = ConvexHull(sampled_output_,)
    #print(convex_hull.vertices)
       #boundary_points = sampled_output_[convex_hull.vertices,:]
          boundary_points= getboundary(sampled_output_,0.0)
       #sampled_output_center=boundary_points.mean(axis=0)
          print('Runnnig partitioning with convex boundary...')
       #print(sampled_output_center)
       else:
          boundary_points= getboundary(sampled_output_,0.4)
          print('Runnnig partitioning with concave boundary...')

   # print(convex_hull_boundary[:,0],convex_hull_boundary[:,1])
  
    #sampled_output_center= boundary_points[random.randrange(0, len(boundary_points), 1)]

   # print(convex_hull_boundary[:,0],convex_hull_boundary[:,1])
    plt.plot(sampled_output_[:,0],sampled_output_[:,1],'ob')
    plt.plot(boundary_points[:,0],boundary_points[:,1],'og')

    #plt.show()
    for i in range(N**2):
       #print(sampled_output_[i,0],sampled_output_[i,1])
       #print('**')
       if sampled_output_[i,:] in boundary_points:
          sampled_output_boundary_[i]=1

       M.append((input_range_[i,:,:], output_range_[i,:,:], sampled_output_boundary_[i]))# (Line 4)

    # Get initial output reachable set (Line 3)
    
    interior_M = []
    
    # Run N simulations (i.e., randomly sample N pts from input range --> query NN --> get N output pts)
    # (Line 5)
   # sampled_inputs = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
    #sampled_outputs = model(torch.Tensor(sampled_inputs), method_opt=None).data.numpy()
    
    # Compute [u_sim], aka bounds on the sampled outputs (Line 6)

    output_range_sim = np.empty((num_outputs, 2))
    output_range_sim[:,0] = np.min(sampled_output_, axis=0)
    output_range_sim[:,1] = np.max(sampled_output_, axis=0)
    
    #print(output_range_sim)
    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf
    area_partitions=np.inf

    while len(M) != 0:
        input_range_, output_range_,sampled_output_boundary_ = M.pop(0) # Line 9
    
        if np.all((output_range_sim[:,0] - output_range_[:,0]) <= 0) and \
            np.all((output_range_sim[:,1] - output_range_[:,1]) >= 0) or sampled_output_boundary_==0:
            # Line 11
            tmp = np.dstack([u_e, output_range_])
            u_e[:,0] = np.min(tmp[:,0,:], axis=1)
            u_e[:,1] = np.max(tmp[:,1,:], axis=1)
            interior_M.append((input_range_, output_range_))
        else:
      
            terminating_condition = np.max(input_range_[:,1] - input_range_[:,0])
                # Line 15
            if terminating_condition>tolerance_eps:
            # Line 14
            #if np.max(input_range_[:,1] - input_range_[:,0]) > tolerance_eps:
                # Line 15
                input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                temp_M = []
                for input_range_ in input_ranges_:
                    output_range_ = get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
                    temp_M.append((input_range_,output_range_,sampled_output_boundary_))
                prev_area_paritions = area_partitions
                area_partitions=0
                for(input_range_,output_range_,sampled_output_boundary_) in temp_M :
                    area_partitions = area_partitions+(output_range_[1,0]-output_range_[0,0])*(output_range_[1,1]-output_range_[1,0])

            # Line 14
                terminating_condition= abs(area_partitions - prev_area_paritions)
                if terminating_condition>tolerance_eps:
                    for (input_range_,output_range_,sampled_output_boundary_) in temp_M:
                        M.append((input_range_,output_range_,sampled_output_boundary_) ) # Line 18
                else:
                	break
            else: # Lines 19-20
                break
    
    # Line 24
    if len(M) > 0:
        # Squash all of M down to one range
        M_numpy = np.dstack([output_range_ for (_, output_range_,_) in M])
        M_range = np.empty((num_outputs, 2))
        M_range[:,1] = np.max(M_numpy[:,1,:], axis=1)
        M_range[:,0] = np.min(M_numpy[:,0,:], axis=1)
    
        # Combine M (remaining ranges) with u_e (interior ranges)
        tmp = np.dstack([u_e, M_range])
        u_e[:,1] = np.max(tmp[:,1,:], axis=1)
        u_e[:,0] = np.min(tmp[:,0,:], axis=1)
    
    if viz:
        visualize_partitions_gh(output_range_no_partition,sampled_output_, u_e, input_range, M=M, interior_M=interior_M, output_range_sim=output_range_sim)
    return u_e
def simulation_guided_partition(model, input_range, num_outputs, viz=False, bound_method="ibp"):
    # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
    num_calls_propagator =0

    tolerance_range =0.01
    output_range_no_partition = get_output_range(model, input_range, num_outputs, bound_method=bound_method)

    tolerance_eps = 0.01
    sect_method = 'max'
    num_inputs = input_range.shape[0]
    
    # Get initial output reachable set (Line 3)
    output_range = get_output_range(model, input_range, num_outputs, bound_method=bound_method)
    num_calls_propagator +=1
    
    M = [(input_range, output_range)] # (Line 4)
    interior_M = []
    
    # Run N simulations (i.e., randomly sample N pts from input range --> query NN --> get N output pts)
    # (Line 5)
    N = 1000
    sampled_inputs = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
    sampled_outputs = model(torch.Tensor(sampled_inputs), method_opt=None).data.numpy()
    
    # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
    #print(sampled_outputs)
   # print(np.max(sampled_outputs, axis=0))
   # print(np.min(sampled_outputs, axis=0))

    output_range_sim = np.empty((num_outputs, 2))
    output_range_sim[:,0] = np.min(sampled_outputs, axis=0)
    output_range_sim[:,1] = np.max(sampled_outputs, axis=0)
    
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
            if np.max(input_range_[:,1] - input_range_[:,0]) > tolerance_eps:
                # Line 15
                input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                for input_range_ in input_ranges_:
                    output_range_ = get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
                    num_calls_propagator+=1
                    M.append((input_range_, output_range_)) # Line 18
            else: # Lines 19-20
                interior_M.append((input_range_, output_range_)) # Line 18
                print('breaking point')
                break


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
    N = 1000

    sampled_inputs = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
    sampled_outputs_eval = model(torch.Tensor(sampled_inputs), method_opt=None).data.numpy()
    sampled_outputs_range = np.empty((num_outputs, 2))
    sampled_outputs_range[:,0] = np.min(sampled_outputs_eval, axis=0)
    sampled_outputs_range[:,1] = np.max(sampled_outputs_eval, axis=0)
    viz=0   
    if viz:
      
        visualize_partitions(0,output_range_no_partition,sampled_outputs_eval, u_e, input_range, sampled_outputs_range, interior_M=ranges,M=None)
    
    print('********* printing results ......')
    print('number of partitions_sg:')
    print(len(M)+len(interior_M))

   # boundary_output_partitions= getboundary(M_numpy,0.0)
    #print(polygon_area(M_numpy[0,:],M_numpy[1,:] ))
    print('bounding box area ratio_sg:')
  
    u_range = u_e[:,1]- u_e[:,0]
    


    sim_out_range= sampled_outputs_range[:,1]-sampled_outputs_range[:,0]
    print(u_range[0]*u_range[1]/(sim_out_range[0]*sim_out_range[1]))

    print('number of calling propagator_sg:')

    print(num_calls_propagator)
    return u_e
def simulation_guided_partition_boundary(model, input_range, num_outputs, viz=False, boundary_shape="convex",bound_method="ibp"):
    # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
    tolerance_eps = 0.01
    tolerance_step=0.001
    tolerance_range=0.05
    num_propagator_calls=0
    k_NN=1
    N = 20
    interior_M = []

    sect_method = 'max'
    num_inputs = input_range.shape[0]

    output_range_no_partition = np.empty((num_outputs, 2))

    output_range_no_partition= get_output_range(model, input_range, num_outputs, bound_method=bound_method)
    #output_range_no_partition[:,0] = np.min(output_range, axis=0)
    #output_range_no_partition[:,1] = np.max(output_range, axis=0)
    #print(output_range_no_partition)
    num_partitions = N*np.ones((num_inputs,), dtype=int)
    slope = np.divide((input_range[:,1] - input_range[:,0]), num_partitions)
    output_range_ = np.empty((N**2,num_outputs,2))
    #sampled_input_ = np.empty((N,1))
    sampled_output_ = np.empty((N**2,num_outputs))
    input_range_=np.empty((N**2,num_inputs,2))
    sampled_output_boundary_ = np.empty((N**2))


    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf
    M=[]
    element=-1

    for idx in product(*[range(num) for num in num_partitions]):
       element+=1
       input_range_[element,:] = input_range[:,0]+np.multiply(idx, slope)
       input_range_[element,:] = input_range[:,0]+np.multiply(np.array(idx)+1, slope)
       sampled_input_= np.random.uniform(input_range_[element,:,0], input_range_[element,:,1], (1,num_inputs,))

       # sampled_input_[element,0]= np.random.uniform(input_range_[element,:,0], input_range_[element,:,1], (1,num_inputs,))
     
       sampled_output_[element,:] = model(torch.Tensor(sampled_input_), method_opt=None).data.numpy()
       sampled_output_boundary_[element] =0;
       #output_range_[element,:,:]= get_output_range(model, input_range_[element,:,:], num_outputs, bound_method=bound_method)
    output_range_sim = np.empty((num_outputs, 2))
    output_range_sim[:,0] = np.min(sampled_output_, axis=0)
    output_range_sim[:,1] = np.max(sampled_output_, axis=0)


    #sampled_output_center=sampled_output_.mean(axis=0)



    if boundary_shape=="convex":
       #convex_hull = ConvexHull(sampled_output_,)
    #print(convex_hull.vertices)
       #boundary_points = sampled_output_[convex_hull.vertices,:]
       boundary_points= getboundary(sampled_output_,0.0)
       #sampled_output_center=boundary_points.mean(axis=0)
      # print('Runnnig partitioning with convex boundary...')
       #print(sampled_output_center)
    else:
       boundary_points= getboundary(sampled_output_,0.4)
   # print(convex_hull_boundary[:,0],convex_hull_boundary[:,1])
  
    #sampled_output_center= boundary_points[random.randrange(0, len(boundary_points), 1)]

   # print(convex_hull_boundary[:,0],convex_hull_boundary[:,1])
    plt.plot(sampled_output_[:,0],sampled_output_[:,1],'ob')
    plt.plot(boundary_points[:,0],boundary_points[:,1],'og')
    M_inside=[]
    #plt.show()


    for i in range(N**2):
       #print(sampled_output_[i,0],sampled_output_[i,1])
       #print('**')
      if sampled_output_[i,:] in boundary_points:
        sampled_output_boundary_[i]=1
        output_range_[i,:,:]= get_output_range(model, input_range_[i,:,:], num_outputs, bound_method=bound_method)

        M.append((input_range_[i,:,:], output_range_[i,:,:], sampled_output_boundary_[i]))# (Line 4)
      else:
        M_inside.append(input_range_[i,:,:])# (Line 4)

    # Get initial output reachable set (Line 3)
    
    interior_M = []
    
    # Run N simulations (i.e., randomly sample N pts from input range --> query NN --> get N output pts)
    # (Line 5)
   # sampled_inputs = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
    #sampled_outputs = model(torch.Tensor(sampled_inputs), method_opt=None).data.numpy()
    
    # Compute [u_sim], aka bounds on the sampled outputs (Line 6)

    output_range_sim = np.empty((num_outputs, 2))
    output_range_sim[:,0] = np.min(sampled_output_, axis=0)
    output_range_sim[:,1] = np.max(sampled_output_, axis=0)
    
    #print(output_range_sim)
    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf
    area_partitions=np.inf




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
            if np.max(input_range_[:,1] - input_range_[:,0]) > tolerance_eps:
                # Line 15
                input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                for input_range_ in input_ranges_:
                    output_range_= get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
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
            output_range_ = get_output_range(model, input_range, num_outputs, bound_method=bound_method)
            num_propagator_calls += 1
            M2 = np.dstack([M_e, output_range_])
            M_e[:,1] = np.max(M2[:,1,:], axis=1)
            M_e[:,0] = np.min(M2[:,0,:], axis=1)

        tmp2 = np.dstack([u_e, M_e])
        u_e[:,1] = np.max(tmp2[:,1,:], axis=1)
        u_e[:,0] = np.min(tmp2[:,0,:], axis=1)
    
    
    
    if viz:
        N = 1000
      
        sampled_inputs_eval = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
        sampled_outputs_eval = model(torch.Tensor(sampled_inputs_eval), method_opt=None).data.numpy()
        sampled_outputs_range = np.empty((num_outputs, 2))
        sampled_outputs_range[:,0] = np.min(sampled_outputs_eval, axis=0)
        sampled_outputs_range[:,1] = np.max(sampled_outputs_eval, axis=0)

        visualize_partitions(1, output_range_no_partition,sampled_outputs_eval, u_e, input_range, sampled_outputs_range, interior_M=interior_M,M=M)
    
    print('********* printing results ......')
    print('number of partitions_b:')
    print(len(M)+len(interior_M))

   # boundary_output_partitions= getboundary(M_numpy,0.0)
    #print(polygon_area(M_numpy[0,:],M_numpy[1,:] ))
    print('bounding box area ratio_b:')
  
    u_range = u_e[:,1]- u_e[:,0]
    


    sim_out_range= sampled_outputs_range[:,1]-sampled_outputs_range[:,0]
    print(u_range[0]*u_range[1]/(sim_out_range[0]*sim_out_range[1]))

    print('number of calling propagator_b:')

    print(num_propagator_calls)
    return u_e


def uniform_partition(model, input_range, num_outputs, viz=False, bound_method="ibp", num_partitions=None):
    num_inputs = input_range.shape[0]
    num_calls_propagator=0
    output_range_no_partition = get_output_range(model, input_range, num_outputs, bound_method=bound_method)
    n_part =16

    if num_partitions is None:
        num_partitions = n_part*np.ones((num_inputs,), dtype=int)
    slope = np.divide((input_range[:,1] - input_range[:,0]), num_partitions)    
    ranges = []
    
    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf
    for element in product(*[range(num) for num in num_partitions]):
        input_range_ = np.empty_like(input_range)
        input_range_[:,0] = input_range[:,0]+np.multiply(element, slope)
        input_range_[:,1] = input_range[:,0]+np.multiply(np.array(element)+1, slope)
        output_range_ = get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
        num_calls_propagator +=1
    
        tmp = np.dstack([u_e, output_range_])
        u_e[:,1] = np.max(tmp[:,1,:], axis=1)
        u_e[:,0] = np.min(tmp[:,0,:], axis=1)
        
        ranges.append((input_range_, output_range_))
    N = 1000

    sampled_inputs = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
    sampled_outputs_eval = model(torch.Tensor(sampled_inputs), method_opt=None).data.numpy()
    sampled_outputs_range = np.empty((num_outputs, 2))
    sampled_outputs_range[:,0] = np.min(sampled_outputs_eval, axis=0)
    sampled_outputs_range[:,1] = np.max(sampled_outputs_eval, axis=0)
    viz=0
    if viz:
      
        
        visualize_partitions(0,output_range_no_partition,sampled_outputs_eval, u_e, input_range, sampled_outputs_range, interior_M=ranges,M=None)
    
    print('********* printing results ......')

    print('number of partitions_uniform:')
    print(n_part**2)

   # boundary_output_partitions= getboundary(M_numpy,0.0)
    #print(polygon_area(M_numpy[0,:],M_numpy[1,:] ))
    print('bounding box area ratio_uniform:')
  
    u_range = u_e[:,1]- u_e[:,0]


    

    sim_out_range= sampled_outputs_range[:,1]-sampled_outputs_range[:,0]
    print(u_range[0]*u_range[1]/(sim_out_range[0]*sim_out_range[1]))

    print('number of calling propagator_uniform:')

    print(num_calls_propagator)

    return u_e


def visualize_partitions(bound,output_range_no_partition, sampled_outputs, estimated_output_range, input_range, output_range_sim=None, interior_M=None, M=None):
    fig, axes = plt.subplots(1,2)
    if interior_M is not None and len(interior_M) > 0:
        for (input_range_, output_range_) in interior_M:
            rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                    fc='none', linewidth=1,edgecolor='m')
            axes[1].add_patch(rect)

            rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
                    fc='none', linewidth=1,edgecolor='m')
            axes[0].add_patch(rect)
    if M is not None and len(M) > 0 and bound==1:
        for (input_range_, output_range_,_) in M:
            rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                    fc='none', linewidth=1,edgecolor='b')
            axes[1].add_patch(rect)

            rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
                    fc='none', linewidth=1,edgecolor='b')
            axes[0].add_patch(rect)
    else:
        if  M is not None and len(M) > 0:
            for (input_range_, output_range_) in M:
                rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                    fc='none', linewidth=1,edgecolor='b')
                axes[1].add_patch(rect)

                rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
                    fc='none', linewidth=1,edgecolor='b')
                axes[0].add_patch(rect)

    axes[1].scatter(sampled_outputs[:,0], sampled_outputs[:,1], c='k', zorder=2)

    rect = Rectangle(estimated_output_range[:,0], estimated_output_range[0,1]-estimated_output_range[0,0], estimated_output_range[1,1]-estimated_output_range[1,0],
                    fc='none', linewidth=2,edgecolor='g')
    axes[1].add_patch(rect)
    rect = Rectangle(output_range_no_partition[:,0], output_range_no_partition[0,1]-output_range_no_partition[0,0], output_range_no_partition[1,1]-output_range_no_partition[1,0],
                    fc='none', linewidth=2,edgecolor='r',linestyle='dashed')

    axes[1].add_patch(rect)

    if output_range_sim is not None:
        rect = Rectangle(output_range_sim[:,0], output_range_sim[0,1]-output_range_sim[0,0], output_range_sim[1,1]-output_range_sim[1,0],
                        fc='none', linewidth=1,edgecolor='k')
        axes[1].add_patch(rect)

    axes[0].set_xlim(input_range[0,0], input_range[0,1])
    axes[0].set_ylim(input_range[1,0], input_range[1,1])
    axes[1].set_xlim(output_range_no_partition[0,0]-1, output_range_no_partition[0,1]+1)
    axes[1].set_ylim(output_range_no_partition[1,0]-1, output_range_no_partition[1,1]+1)
    plt.show()

def visualize_partitions_gh_plus(output_range_actual, sampled_outputs, estimated_output_range, input_range, output_range_sim=None, interior_M=None, M=None):
    fig, axes = plt.subplots(1,2)
    if interior_M is not None and len(interior_M) > 0:
        for (input_range_, output_range_) in interior_M:
            rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                    fc='none', linewidth=1,edgecolor='m')
            axes[1].add_patch(rect)

            rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
                    fc='none', linewidth=1,edgecolor='m')
            axes[0].add_patch(rect)
    if M is not None and len(M) > 0:
        for (input_range_, output_range_) in M:
            rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                    fc='none', linewidth=1,edgecolor='b')
            axes[1].add_patch(rect)

            rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
                    fc='none', linewidth=1,edgecolor='b')
            axes[0].add_patch(rect)

    axes[1].scatter(sampled_outputs[:,0], sampled_outputs[:,1], c='k', zorder=2)

    rect = Rectangle(estimated_output_range[:,0], estimated_output_range[0,1]-estimated_output_range[0,0], estimated_output_range[1,1]-estimated_output_range[1,0],
                    fc='none', linewidth=2,edgecolor='g')
    axes[1].add_patch(rect)

    rect2 = Rectangle(output_range_actual[:,0], output_range_actual[0,1]-output_range_actual[0,0], output_range_actual[1,1]-output_range_actual[1,0],
                    fc='none', linewidth=1,edgecolor='r',linestyle='dashed')
    axes[1].add_patch(rect2)
   # print(output_range_actual)
   # print(output_range_sim)

    if output_range_sim is not None:
        rect = Rectangle(output_range_sim[:,0], output_range_sim[0,1]-output_range_sim[0,0], output_range_sim[1,1]-output_range_sim[1,0],
                    fc='none', linewidth=1,edgecolor='k',linestyle='dashed')
        axes[1].add_patch(rect)


    axes[0].set_xlim(input_range[0,0], input_range[0,1])
    axes[0].set_ylim(input_range[1,0], input_range[1,1])

    axes[1].set_xlim(output_range_actual[0,0]-1, output_range_actual[0,1]+1)
    axes[1].set_ylim(output_range_actual[1,0]-1, output_range_actual[1,1]+1)

    plt.show()
    fig.tight_layout()





def get_output_range(model, input_range, num_outputs, bound_method="ibp"):
    method_dict = {
        "crown": "full_backward_range",
        "ibp": "interval_range",
    }
    output_range = np.empty((num_outputs,2))
    for out_index in range(num_outputs):
        C = torch.zeros((1,1,num_outputs))
        C[0,0,out_index] = 1
        out_max, out_min = model(norm=np.inf,
                                    x_U=torch.Tensor([input_range[:,1]]),
                                    x_L=torch.Tensor([input_range[:,0]]),
                                    C=C,
                                    method_opt=method_dict[bound_method],
                                    )[:2]
        output_range[out_index,:] = [out_min, out_max]
    return output_range

def bisect(input_range):
    return sect(input_range, num_sects=2)

def sect(input_range, num_sects=3, select='random'):
    num_inputs = input_range.shape[0]
    if select == 'random':
        input_dim_to_sect = np.random.randint(0, num_inputs)
    else:
        input_dim_to_sect = np.argmax(input_range[:,1] - input_range[:,0])
    input_ranges = np.tile(input_range, (num_sects,1,1,))
    diff = (input_range[input_dim_to_sect,1]-input_range[input_dim_to_sect,0])/float(num_sects)
    for i in range(num_sects-1):
        new_endpt = input_range[input_dim_to_sect,0]+(i+1)*diff
        input_ranges[i,input_dim_to_sect,1] = new_endpt
        input_ranges[i+1,input_dim_to_sect,0] = new_endpt
    return input_ranges
def set_delta_step(range_,center_value, idx, stage=1):
    c=0.5
    if stage==1:
        k=-1
        num_inputs=idx
        delta_step =np.ones((num_inputs*2,1))

        print(len(range_))
        output_range = range_
        distance = np.empty((len(range_)*2))
        print(distance)
        output_center = center_value
        output_bounding_box = [[output_range[0,0],output_range[1,0]],[output_range[0,1],output_range[1,0]],
        [output_range[0,1],output_range[1,1]],[output_range[0,0],output_range[1,1]]]
        for (i,j) in output_bounding_box:
            k=k+1
            print(distance[k])
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
        print('hereeeee')
        delta_step =np.zeros((num_inputs*2,1))
        if idx % 2 == 0: 
            delta_step[idx]= -c*diff_rolled[idx]/(np.max(input_range[:,0]))
        else:
            delta_step[idx]= c*diff_rolled[idx]/(np.max(input_range[:,1]))
    return delta_step

def xiang2020example(input_range=None, model=None, bound_method="ibp"):
    a = np.array([1, 2, 3])   # Create a rank 1 array
    print(type(a))            # Prints "<class 'numpy.ndarray'>"
    print(a.shape)    
    if model is None:
        torch_model = model_xiang_2020_robot_arm()
        #torch_model = model_gh1()
        torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
        num_outputs = 2

    if input_range is None:
        input_range = np.array([ # (num_inputs, 2)
                          [np.pi/3, 2*np.pi/3], # x0min, x0max
                          [np.pi/3, 2*np.pi/3] # x1min, x1max
        ])
    output_range_simulation_guided_gh= simulation_guided_partition_gh_plus(torch_model_, input_range, num_outputs, viz=True, boundary_shape="convex", bound_method=bound_method)
    #output_range_simulation_guided_gh= simulation_guided_partition_gh(torch_model_, input_range, num_outputs, viz=True,  boundary_shape="convex", bound_method=bound_method)
    output_range_simulation_guided = simulation_guided_partition(torch_model_, input_range, num_outputs, viz=True, bound_method=bound_method)
    output_range_uniform = uniform_partition(torch_model_, input_range, num_outputs, viz=True, bound_method=bound_method)
    #output_range_bouandary = simulation_guided_partition_boundary(torch_model_, input_range, num_outputs, viz=True, boundary_shape="convex", bound_method=bound_method)

    #print(output_range_simulation_guided_gh)
    #print(output_range_simulation_guided)
    #print(output_range_uniform)

def xiang2017example(input_range=None, model=None, num_partitions=None):
    if model is None:
        torch_model = model_xiang_2017()
        torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
        num_outputs = 2

    if input_range is None:
        input_range = np.array([ # (num_inputs, 2)
                          [0., 1.], # x0min, x0max
                          [0., 1.] # x1min, x1max
        ])

    if num_partitions is None:
        num_inputs = input_range.shape[0]
        num_partitions = 5*np.ones((num_inputs,), dtype=int)

    bound_method = "ibp"
    output_range_uniform = uniform_partition(torch_model_, input_range, num_outputs, viz=True, bound_method=bound_method, num_partitions=num_partitions)

if __name__ == '__main__':
	#xiang2017example()

	print("CROWN...")
	xiang2020example(bound_method="crown")
	#print("IBP...")
	#xiang2020example(bound_method="ibp")

