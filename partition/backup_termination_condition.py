
    
    N=1000
    sampled_inputs_eval = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
    sampled_outputs_eval = model(torch.Tensor(sampled_inputs_eval), method_opt=None).data.numpy()
    sampled_outputs_range = np.empty((num_outputs, 2))
    sampled_outputs_range[:,0] = np.min(sampled_outputs_eval, axis=0)
    sampled_outputs_range[:,1] = np.max(sampled_outputs_eval, axis=0)
    sim_area = (sampled_outputs_range[0,1]-sampled_outputs_range[0,0])*(sampled_outputs_range[1,1]-sampled_outputs_range[1,0])
    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf



    while len(M) != 0:
        input_range_, output_range_ = M.pop(0) # Line 9
        
        if np.all((output_range_sim[:,0] - output_range_[:,0]) <= 0) and \
            np.all((output_range_sim[:,1] - output_range_[:,1]) >= 0):
            # Line 11
            tmp = np.dstack([u_e, output_range_])
            u_e[:,0] = np.min(tmp[:,0,:], axis=1)
            u_e[:,1] = np.max(tmp[:,1,:], axis=1)
            interior_M.append((input_range_, output_range_))
        else:

            # Line 14
            if np.max(input_range_[:,1] - input_range_[:,0]) > tolerance_eps:
                # Line 15
                prev_partitions_area = (output_range_[0,1]-output_range_[0,0])*(output_range_[1,1]-output_range_[1,0])

                input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                tempM=[]
                # Lines 16-17
                partition_range = np.empty((num_outputs,2))
                partition_range[:,0] = np.inf
                partition_range[:,1] = -np.inf

                for input_range_sect_ in input_ranges_:
                    output_range_sect_= get_output_range(model, input_range_sect_, num_outputs, bound_method=bound_method)
                    num_calls_propagator +=1
                    tmp = np.dstack([partition_range, output_range_sect_])
                    partition_range[:,0] = np.min(tmp[:,0,:], axis=1)
                    partition_range[:,1] = np.max(tmp[:,1,:], axis=1)
     
                    tempM.append((input_range_sect_, output_range_sect_)) # Line 18

                area_sum = (partition_range[0,1]-partition_range[0,0])*(partition_range[1,1]-partition_range[1,0])
                # Line 18
                if  abs(prev_partitions_area-area_sum)<tolerance_range and area_sum>=sim_area:

                    break
                else:
                    while len(tempM) != 0:
                        input_range_,output_range_ = tempM.pop(0)
                        M.append((input_range_,output_range_) ) # Line 18

            else: # Lines 19-20
                break
    