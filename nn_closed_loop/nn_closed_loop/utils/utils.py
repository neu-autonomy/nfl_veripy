import pickle
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def save_dataset(xs, us, system="DoubleIntegrator", dataset_name="default"):
    path = "{}/../../datasets/{}/{}".format(dir_path, system, dataset_name)
    os.makedirs(path, exist_ok=True)
    with open(path+"/dataset.pkl", "wb") as f:
        pickle.dump([xs, us], f)


def load_dataset(system="DoubleIntegrator", dataset_name="default"):
    path = "{}/../../datasets/{}/{}".format(dir_path, system, dataset_name)
    with open(path+"/dataset.pkl", "rb") as f:
        xs, us = pickle.load(f)
    return xs, us


def range_to_polytope(state_range):
    num_states = state_range.shape[0]
    A = np.vstack([np.eye(num_states), -np.eye(num_states)])
    b = np.hstack([state_range[:, 1], -state_range[:, 0]])
    return A, b


def get_polytope_A(num):
    theta = np.linspace(0, 2 * np.pi, num=num + 1)
    A_out = np.dstack([np.cos(theta), np.sin(theta)])[0][:-1]
    return A_out


def get_next_state(xt, ut, At, bt, ct):
    return np.dot(At, xt.T) + np.dot(bt, ut.T)


def plot_polytope_facets(A, b, ls='-', show=True):
    import matplotlib.pyplot as plt
    cs = ['r','g','b','brown','tab:red', 'tab:green', 'tab:blue', 'tab:brown']
    ls = ['-', '-', '-', '-', '--', '--', '--', '--']
    num_facets = b.shape[0]
    x = np.linspace(1, 5, 2000)
    for i in range(num_facets):
        alpha = 0.2
        if A[i, 1] == 0:
            offset = -0.1*np.sign(A[i, 0])
            plt.axvline(x=b[i]/A[i, 0], ls=ls[i], c=cs[i])
            plt.fill_betweenx(y=np.linspace(-2, 2, 2000), x1=b[i]/A[i, 0], x2=offset+b[i]/A[i, 0], fc=cs[i], alpha=alpha)
        else:
            offset = -0.1*np.sign(A[i, 1])
            y = (b[i] - A[i, 0]*x)/A[i, 1]
            plt.plot(x, y, ls=ls[i], c=cs[i])
            plt.fill_between(x, y, y+offset, fc=cs[i], alpha=alpha)
    if show:
        plt.show()

def get_polytope_verts(A, b):
    import pypoman
    # vertices = pypoman.duality.compute_polytope_vertices(A, b)
    vertices = pypoman.polygon.compute_polygon_hull(A, b)
    print(vertices)



def plot_time_data(info):
    labels = {
        'br_lp' : 'LPs (Backreach)', 
        'bp_lp' : 'LPs (BReach)', 
        'nstep_bp_lp' : 'LPs (ReBReach)', 
        'crown' : 'CROWN (BReach)', 
        'nstep_crown' : 'CROWN (ReBReach)', 
        'other' : 'Other (BReach)', 
        'nstep_other' : 'Other (ReBReach)'
    }
    num_entries = {}
    vals = []
    sums = []
    for dict in info['per_timestep']:
        step_values = {}
        for key in labels:
            if key in dict:
                step_values[key] = sum(dict[key])
                if key in num_entries:
                    num_entries[key] += len(dict[key])
                else:
                    num_entries[key] = len(dict[key])
            else:
                step_values[key] = 0
                if key in num_entries:
                    num_entries[key] += 0
                else:
                    num_entries[key] = 0

        
        vals.append(step_values)

    summed_vals = {
        'br_lp' : [0], 
        'bp_lp' : [0], 
        'nstep_bp_lp' : [0], 
        'crown' : [0], 
        'nstep_crown' : [0], 
        'other' : [0], 
        'nstep_other' : [0]
    }
    for i in range(len(vals)):
        for key in vals[i]:
            
            summed_vals[key].append(summed_vals[key][i]+vals[i][key])
    summed_value_list = list(summed_vals.values())

    if num_entries['bp_lp'] > 0:
        print('Number of LPs solved (BReach): {}'.format(num_entries['bp_lp']))
        print('Time per LP solved (BReach): {0:.4f}'.format(summed_vals['bp_lp'][-1]/num_entries['bp_lp']))
    if num_entries['nstep_bp_lp'] > 0:
        print('Number of LPs solved (ReBReach): {}'.format(num_entries['nstep_bp_lp']))
        print('Time per LP solved (ReBReach): {0:.4f}'.format(summed_vals['nstep_bp_lp'][-1]/num_entries['nstep_bp_lp']))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    from colour import Color
    orange = Color("orange")
    colors = list(orange.range_to(Color("purple"),len(vals)))
    cm = plt.cm.get_cmap('tab20')
    # import pdb; pdb.set_trace()
    for i,dict in enumerate(vals):
        # ax.bar(labels.values(), dict.values(), color=colors[i].hex_l, bottom=[x[i] for x in summed_value_list])
        ax.bar(labels.values(), dict.values(), color=cm.colors[i], bottom=[x[i] for x in summed_value_list])
    ax.set_ylabel('Time (s)', fontsize=12)
    
    try:
        textstr = '\n'.join((
            'Number of LPs (BReach): {}'.format(num_entries['bp_lp']),
            'Time per LP (BReach): {0:.4f}'.format(summed_vals['bp_lp'][-1]/num_entries['bp_lp']),
            'Number of LPs (ReBReach): {}'.format(num_entries['nstep_bp_lp']),
            'Time per LP (ReBReach): {0:.4f}'.format(summed_vals['nstep_bp_lp'][-1]/num_entries['nstep_bp_lp'])))

        # these are matplotlib.patch.Patch properties
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top')
    except:
        try:
            textstr = '\n'.join((
            'Number of LPs (BReach): {}'.format(num_entries['bp_lp']),
            'Time per LP (BReach): {0:.4f}'.format(summed_vals['bp_lp'][-1]/num_entries['bp_lp'])))
            ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top')
        except:
            pass



    plt.xticks(rotation=60,fontsize=12)
    plt.subplots_adjust(bottom=0.36)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    plt.show()


if __name__ == "__main__":
    # save_dataset(xs, us)
    # xs, us = load_dataset()

    import matplotlib.pyplot as plt

    A = np.array([
              [1, 1],
              [0, 1],
              [-1, -1],
              [0, -1]
    ])
    b = np.array([2.8, 0.41, -2.7, -0.39])

    A2 = np.array([
                  [1, 1],
                  [0, 1],
                  [-0.97300157, -0.95230697],
                  [0.05399687, -0.90461393]
    ])
    b2 = np.array([2.74723146, 0.30446292, -2.64723146, -0.28446292])

    # get_polytope_verts(A, b)
    plot_polytope_facets(A, b)
    # get_polytope_verts(A2, b2)
    plot_polytope_facets(A2, b2, ls='--')
    plt.show()
