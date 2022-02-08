import numpy as np
from nn_closed_loop.utils.nn import create_and_train_model, save_model, load_controller
import os
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

def avoid_origin_controller():
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
        vy = 5*(1+np.exp(-(0.5*x[0]+2)))**-2*np.exp(-(0.5*x[0]+2))-5*(1+np.exp(-(-0.5*x[0]+2)))**-2*np.exp(-(-0.5*x[0]+2))
        vx = 1.0 - 2/(1+2*np.sqrt((x[0]/2)**2+(x[1]/2)**2))
        if x[1] < 0:
            vy = -vy
        us[i] = np.array([vx,vy])
        
    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(model, name="model", dir=dir_path+"/controllers/avoid_origin_controller/")

def avoid_origin_controller_simple():
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
        vy = 5*(1+np.exp(-(0.5*x[0]+2)))**-2*np.exp(-(0.5*x[0]+2))-5*(1+np.exp(-(-0.5*x[0]+2)))**-2*np.exp(-(-0.5*x[0]+2))
        vx = (1 + 2*np.exp(-np.abs(1/(4*x[1]))))/3
        if x[1] < 0:
            vy = -vy
        us[i] = np.array([vx,vy])
        
    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(model, name="model", dir=dir_path+"/controllers/avoid_origin_controller_simple/")

def stop_at_origin_controller():
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
        vy = 5*(1+np.exp(-(0.5*x[0]+2)))**-2*np.exp(-(0.5*x[0]+2))-5*(1+np.exp(-(-0.5*x[0]+2)))**-2*np.exp(-(-0.5*x[0]+2))
        vx = 1
        if x[1] < 0:
            vy = -vy
        if np.linalg.norm(x) < 3:
            vx = 0
            vy = 0
        us[i] = np.array([vx,vy])
        
    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(model, name="model", dir=dir_path+"/controllers/stop_at_origin_controller/")



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


def display_ground_robot_control_field(name = 'avoid_origin_controller_simple'):
    controller = load_controller(system='GroundRobotSI', model_name=name)
    x,y = np.meshgrid(np.linspace(-7,-4.5,20), np.linspace(-0.5,2,20))
    # import pdb; pdb.set_trace()
    inputs = np.hstack((x.reshape(len(x)*len(x[0]),1), y.reshape(len(y)*len(y[0]),1)))
    us = controller.predict(inputs)
    
    
    # import pdb; pdb.set_trace()
    plt.quiver(x,y,us[:,0].reshape(len(x),len(y)),us[:,1].reshape(len(x),len(y)))
    plt.show()



def main():
    display_ground_robot_control_field()
    # stop_at_origin_controller()
    # zero_input_controller()

if __name__ == "__main__":
    main()
