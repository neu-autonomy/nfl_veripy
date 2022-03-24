from turtle import color
import numpy as np
import scipy.io
from nn_closed_loop.utils.nn import create_and_train_model, save_model, load_controller, create_model
import os
import matplotlib.pyplot as plt
import nn_closed_loop.dynamics as dynamics
import nn_closed_loop.constraints as constraints
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

# Policy used to debug CROWN
def random_weight_controller():
    neurons_per_layer = [10,10]
    xs = np.zeros((10,1))
    us = np.zeros((10,1))
        
    model = create_model(neurons_per_layer, xs.shape[1:], us.shape[1:])
    save_model(model, name="model", dir=dir_path+"/controllers/random_weight_controller/")

# Policy used to debug CROWN
def zero_input_controller():
    neurons_per_layer = [10,10]
    state_range = np.array(
        [
            [-20, 20],
            [-15, 15]
        ]
    )
    xs = np.random.uniform(low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2))
    us = np.zeros(xs.shape)
    for i,x in enumerate(xs):
        vy = 0
        vx = 0

        us[i] = np.array([vx,vy])
        
    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(model, name="model", dir=dir_path+"/controllers/zero_input_controller/")

def potential_field_controller():
    neurons_per_layer = [10,10]
    state_range = np.array(
        [
            [-15, 15],
            [-15, 15]
        ]
    )
    xs = np.random.uniform(low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2))
    us = np.zeros(xs.shape)
    for i,x in enumerate(xs):
        vx = max(min(1+2*x[0]/(x[0]**2+x[1]**2), 1), -1)
        vy = max(min(x[1]/(x[0]**2+x[1]**2), 1), -1)
        us[i] = np.array([vx,vy])
        
    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(model, name="model", dir=dir_path+"/controllers/small_potential_field/")


# Control policy used for CDC 2022 paper
def complex_potential_field_controller():
    neurons_per_layer = [10,10]
    state_range = np.array(
        [
            [-10, 10],
            [-10, 10]
        ]
    )
    xs = np.random.uniform(low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2))
    us = np.zeros(xs.shape)
    for i,x in enumerate(xs):
        vx = max(min(1+2*x[0]/(x[0]**2+x[1]**2), 1), -1)
        vy = max(min(x[1]/(x[0]**2+x[1]**2)+np.sign(x[1])*2*(1+np.exp(-(0.5*x[0]+2)))**-2*np.exp(-(0.5*x[0]+2)), 1), -1)
        us[i] = np.array([vx,vy])
        
    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(model, name="model", dir=dir_path+"/controllers/complex_potential_field/")


def display_ground_robot_control_field(name = 'avoid_origin_controller_simple', ax=None):
    controller = load_controller(system='GroundRobotSI', model_name=name, model_type='keras')
    x,y = np.meshgrid(np.linspace(-6,4,20), np.linspace(-7,7,20))
    # import pdb; pdb.set_trace()
    inputs = np.hstack((x.reshape(len(x)*len(x[0]),1), y.reshape(len(y)*len(y[0]),1)))
    us = controller.predict(inputs)
    
    if ax is None:
        # import pdb; pdb.set_trace()
        plt.quiver(x,y,us[:,0].reshape(len(x),len(y)),us[:,1].reshape(len(x),len(y)),color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.rc('font', size=12)
        plt.show()
    else:
        ax.quiver(x,y,us[:,0].reshape(len(x),len(y)),us[:,1].reshape(len(x),len(y)))

def build_controller_from_matlab(filename = "quad_mpc_data.mat"):
    neurons_per_layer = [25,25,25]
    
    file = dir_path+"/controllers/MATLAB_data/"+filename
    mat = scipy.io.loadmat(file)
    xs = mat['data'][0][0][0][:,0:6]
    us = mat['data'][0][0][1]

    model = create_and_train_model(neurons_per_layer, xs, us, epochs=50, verbose=True, validation_split=0.1)
    save_model(model, name="model", dir=dir_path+"/controllers/quadrotor_matlab_3/")

def generate_mpc_data_quadrotor(num_samples=100):
    dyn = dynamics.Quadrotor()
    state_range = np.array(
        [  # (num_inputs, 3)
            [-2.8, -0.8, -0.8, -0.5, -0.5, -0.5],
            [0.2, 0.8, 0.8, 0.5, 0.5, 0.5],
        ]
    ).T
    input_constraint = constraints.LpConstraint(range=state_range)
    # import pdb; pdb.set_trace()
    xs, us = dyn.collect_data(t_max=1, input_constraint=input_constraint, num_samples=num_samples)
    with open('xs.pickle', 'wb') as handle:
        pickle.dump(xs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('us.pickle', 'wb') as handle:
        pickle.dump(xs, handle, protocol=pickle.HIGHEST_PROTOCOL)




def main():
    # random_weight_controller()
    # avoid_origin_controller_simple()
    # stop_at_origin_controller()
    # zero_input_controller()
    # complex_potential_field_controller()
    # display_ground_robot_control_field(name='complex_potential_field')
    build_controller_from_matlab("quad_mpc_data_paths_small.mat")
    # generate_mpc_data_quadrotor()

if __name__ == "__main__":
    main()
