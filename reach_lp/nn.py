from keras.models import Sequential
from keras.layers import Dense

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

def load_model():
    # load json and create model
    try:
        json_file = open('model.json', 'r')
    except:
        pass
    try:
        json_file = open('/Users/mfe/Downloads/model.json', 'r')
    except:
        pass
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    try:
        model.load_weights("model.h5")
    except:
        pass
    try:
        model.load_weights("/Users/mfe/Downloads/model.h5")
    except:
        pass
    print("Loaded model from disk")
    return model

def control_nn(x, model):
    return model.predict(np.expand_dims(x, axis=0))[0][0]

if __name__ == '__main__':
    neurons_per_layer = [10,5]
    model = create_model(neurons_per_layer)

    model = create_and_train_model(neurons_per_layer, xs, us)