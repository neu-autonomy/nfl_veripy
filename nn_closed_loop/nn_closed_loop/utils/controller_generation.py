from turtle import color
import numpy as np
from nn_closed_loop.utils.nn import create_and_train_model, save_model, load_controller, create_model
import os
import matplotlib.pyplot as plt

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



def main():
    # random_weight_controller()
    # avoid_origin_controller_simple()
    # stop_at_origin_controller()
    # zero_input_controller()
    # complex_potential_field_controller()
    display_ground_robot_control_field(name='complex_potential_field')

if __name__ == "__main__":
    main()
