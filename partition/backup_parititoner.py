
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
                        output_range_,_= propagator.get_output_range(input_range_)
                        num_propagator_calls += 1
   
                        M.append((input_range_, output_range_)) # Line 18
                else: # Lines 19-20
                    interior_M.append((input_range_, output_range_))
                    break