import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from crown_ibp.conversions.keras2torch import keras2torch, get_keras_model
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))

def create_model(neurons_per_layer):
    model = Sequential()
    model.add(Dense(neurons_per_layer[0], input_shape=(num_states,), activation='relu'))
    for neurons in neurons_per_layer[1:]:
        model.add(Dense(neurons, activation='relu'))
        model.add(Dense(num_inputs))
        model.compile(optimizer='rmsprop', loss='mse')
    return model

def create_and_train_model(neurons_per_layer, xs, us, epochs=20, batch_size=32, verbose=0):
    model = create_model(neurons_per_layer)
    model.fit(xs, us, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model(name='double_integrator_mpc'):
    path = '{}/models/{}'.format(dir_path, name)
    with open(path+'/model.json', 'r') as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(path+"/model.h5")
    torch_model = keras2torch(model, "torch_model")
    return torch_model

def control_nn(x, model=None, use_torch=True):
    if model is None:
        model = load_model()
    if x.ndim == 1:
        batch_x = np.expand_dims(x, axis=0)
    else:
        batch_x = x
    if use_torch:
        us = model.forward(torch.Tensor(batch_x)).data.numpy()[0][0]
        return us
    else:
        us = model.predict(batch_x)
        if x.ndim == 1:
            return us[0][0]
        else:
            return us

if __name__ == '__main__':
    neurons_per_layer = [10,5]
    model = create_model(neurons_per_layer)

    model = create_and_train_model(neurons_per_layer, xs, us)