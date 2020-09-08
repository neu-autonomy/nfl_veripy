import pickle
import numpy as np
import pypoman

def save_dataset(xs, us):
    with open("dataset.pkl", "wb") as f:
        pickle.dump([xs, us], f)

def load_dataset():
    with open("/Users/mfe/Downloads/dataset.pkl", "rb") as f:
        xs, us = pickle.load(f)
    return xs, us

def init_state_range_to_polytope(init_state_range):
    num_states = init_state_range.shape[0]
    pts = []
    for i in range(num_states):
        for j in range(num_states):
            pts.append([init_state_range[0,i], init_state_range[1,j]])
    vertices = np.array(pts)
    A_inputs, b_inputs = pypoman.compute_polytope_halfspaces(vertices)
    return A_inputs, b_inputs

def get_polytope_A(num):
    theta = np.linspace(0,2*np.pi,num=num)
    A_out = np.dstack([np.cos(theta), np.sin(theta)])[0][:-1]
    return A_out

def get_next_state(xt, ut, At, bt, ct):
    return np.dot(At, xt.T) + np.dot(bt,ut.T)

if __name__ == '__main__':
    save_dataset(xs, us)
    xs, us = load_dataset()