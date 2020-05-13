import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches as ptch
from tqdm import tqdm

# Run simulation
def run_simulation(At, bt, ct,
    t_max, init_state_range, goal_state_range,
    u_min, u_max, num_states,
    collect_data=False,
    show_bounds=False, all_bs=None, A_in=None, bnd_colors=None,
    model=None,
    save_plot=False, plot_name='sim.png',
    num_samples=None, clip_control=True, show_dataset=False):

    colors = [cm.get_cmap("tab10")(i) for i in range(t_max+1)]

    if collect_data:
        show_bounds = False
    if num_samples is None:
        # num_samples = 100
        num_samples = 2420
        xs = np.zeros((num_samples, num_states))
        us = np.zeros((num_samples))
        np.random.seed(0)
    else:
        if num_samples is None:
            num_samples = 100
        np.random.seed(1)
        dataset_index = 0

    if show_dataset:
        xs, us = load_dataset()
        plt.scatter(xs[:,0], xs[:,1], c='0.2')

    with tqdm(total=num_samples) as pbar:

        while dataset_index < num_samples:

        # Initial state
        num_states = At.shape[0]
        x = np.zeros((int((t_max)/dt)+1, num_states))
        x[0,:] = np.random.uniform(
        low=init_state_range[:,0], 
        high=init_state_range[:,1])

        t = 0
        step = 0
        while t < t_max:
            t += dt
            if collect_data:
                u = control_mpc(x0=x[step,:], A=At, b=bt, Q=Q, R=R, P=Pinf, u_min=u_min, u_max=u_max)
            else:
                u = control_nn(x=x[step,:], model=model)
            if clip_control:
                u = np.clip(u, u_min, u_max)
            if collect_data:
                xs[dataset_index, :] = x[step,:]
                us[dataset_index] = u
            x[step+1,:] = np.dot(At, x[step, :]) + np.dot(bt,u)[:,0]
            step += 1
            dataset_index += 1
            pbar.update(1)
            if dataset_index == num_samples:
                break

        plt.scatter(x[:,0], x[:,1], c=colors)
        # plt.plot(x[:,0], x[:,1])

    if show_bounds:
        for i in range(len(all_bs)):
            this_A_in = A_in[i]
            this_all_bs = all_bs[i]
                for this_bs in this_all_bs:
                vertices = pypoman.compute_polygon_hull(this_A_in, this_bs)
                bnd_color = bnd_colors[i]
                plt.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]], bnd_color)

    # vertices = pypoman.compute_polygon_hull(A_in, bs)
    # plt.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]], 'r')

    # vertices = pypoman.compute_polygon_hull(A_in, bs2)
    # plt.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]], 'b')

    # Input state rectangle
    rect = ptch.Rectangle(init_state_range[:,0],
    init_state_range[0,1]-init_state_range[0,0], 
    init_state_range[1,1]-init_state_range[1,0],
    fill=False, ec='k')
    plt.gca().add_patch(rect)

    #   # Goal state rectangle
    #   rect = ptch.Rectangle(goal_state_range[:,0],
    #                         goal_state_range[0,1]-goal_state_range[0,0], 
    #                         goal_state_range[1,1]-goal_state_range[1,0],
    #                         fill=False, ec='green')
    #   plt.gca().add_patch(rect)



    #   plt.gca().axhline(y=-1, color='red')

    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')

    if save_plot:
        plt.savefig(plot_name)

    plt.show()



if __name__ == '__main__':
    run_simulation(At, bt, ct,
               t_max, init_state_range, goal_state_range,
               u_min, u_max, num_states,
               collect_data=False,
               show_bounds=False, all_bs=None, A_in=None,
               model=model, save_plot=True)