import pickle
import numpy as np
import pypoman
import itertools

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
    A_inputs = np.vstack([np.eye(num_states), -np.eye(num_states)])
    b_inputs = np.hstack([init_state_range[:,1], -init_state_range[:,0]])
    return A_inputs, b_inputs

def get_polytope_A(num):
    theta = np.linspace(0,2*np.pi,num=num+1)
    A_out = np.dstack([np.cos(theta), np.sin(theta)])[0][:-1]
    return A_out

def get_next_state(xt, ut, At, bt, ct):
    return np.dot(At, xt.T) + np.dot(bt,ut.T)

if __name__ == '__main__':
    save_dataset(xs, us)
    xs, us = load_dataset()