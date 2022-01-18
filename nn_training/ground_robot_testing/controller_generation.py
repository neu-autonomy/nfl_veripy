import numpy as np
from nn_closed_loop.utils.nn import create_and_train_model, save_model
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def drive_in_circle_controller():
    neurons_per_layer = [10,10]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [0, 2*np.pi]
        ]
    )
    xs = np.random.uniform(low=state_range[:,0], high=state_range[:,1], size=(10000,3))
    us = np.array([[1.0, np.pi/6] for i in range(10000)], dtype='float32')

    model= create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(model, name="model", dir = dir_path+"/controllers/drive_in_circle_controller/")

def main():
    drive_in_circle_controller()

if __name__ == "__main__":
    main()