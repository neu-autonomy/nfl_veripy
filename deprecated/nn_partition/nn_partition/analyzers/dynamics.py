import numpy as np
import gym
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json

from nn_partition.models.models import model_dynamics
import nn_partition.analyzers as analyzers

np.set_printoptions(suppress=True)
simple_envs = ['CartPole-v0', 'Pendulum-v0']
not_simple_envs = ['HandManipulateBlock-v0']

save_dir = "{}/results/dynamics/".format(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(save_dir, exist_ok=True)

class DynamicsAnalyzer(analyzers.Analyzer):
    def __init__(self, torch_model, env_name):
        analyzers.Analyzer.__init__(self, torch_model=torch_model)
        self.env_name = env_name

    def get_output_range(self, input_range):
        output_range, info = super().get_output_range(input_range)

        # Use NN to get estimated next states for those samples (according to the NN)
        num_samples = 100
        sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (num_samples,)+input_range.shape[:-1])
        next_states_nn_samples = self.propagator.forward_pass(sampled_inputs)

        # Use gym to get exact next states for those samples
        env = gym.make(self.env_name)
        state = env.reset()

        next_states_true = np.empty((num_samples,)+env.observation_space.shape)
        obs_last_ind = input_range.shape[0] - env.action_space.shape[0]
        for i, sampled_input in enumerate(sampled_inputs):
            state, action, next_state = get_env_sample(env, sampled_input, obs_last_ind)
            next_states_true[i,...] = next_state

        info["sampled_inputs"] = sampled_inputs
        info["next_states_nn_samples"] = next_states_nn_samples
        info["next_states_true"] = next_states_true

        return output_range, info

    def visualize(self, input_range, output_range_estimate, **kwargs):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(input_range, output_range_estimate, self.propagator, show_samples=False,
            outputs_to_highlight=kwargs.get("outputs_to_highlight", None),
            inputs_to_highlight=kwargs.get("inputs_to_highlight", None))
        self.partitioner.visualize(kwargs["exterior_partitions"], kwargs["interior_partitions"], output_range_estimate)

        if "sampled_inputs" in kwargs:
            samples_ = kwargs["sampled_inputs"][...,self.partitioner.input_dims_].squeeze()
            self.partitioner.animate_axes[0].scatter(samples_[:,0], samples_[:,1],
                c='tab:pink', marker='.', zorder=2,
                label="Current State/Action Samples")

        if "next_states_nn_samples" in kwargs:
            samples_ = kwargs["next_states_nn_samples"][...,self.partitioner.output_dims_].squeeze()
            self.partitioner.animate_axes[1].scatter(samples_[:,0], samples_[:,1], c='tab:red', marker='.', zorder=3,
                label="Next State Samples (Learned NN)")

        if "next_states_true" in kwargs:
            samples_ = kwargs["next_states_true"][...,self.partitioner.output_dims_].squeeze()
            self.partitioner.animate_axes[1].scatter(samples_[:,0], samples_[:,1], c='tab:olive', marker='.', zorder=3,
                label="Next State Samples (True Env.)")

        self.partitioner.animate_axes[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
        self.partitioner.animate_axes[1].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)

        self.partitioner.animate_fig.tight_layout()

        if "save_name" in kwargs and kwargs["save_name"] is not None:
            plt.savefig(kwargs["save_name"])
        plt.show()

def get_env_sample(env, sampled_input, obs_last_ind):
    state = sampled_input[:obs_last_ind]

    if env.spec.id in simple_envs:
        env.env.state = state.copy()

        action = sampled_input[obs_last_ind:]
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = action[0].astype(int)

    elif env.spec.id in not_simple_envs:
        # hand
        env.env.state['observation'] = state.copy()
        action = sampled_input[obs_last_ind:]

    else:
        raise NotImplementedError

    next_state, _, _, _ = env.step(action)
    return state, action, next_state

def random_sample(env, states, actions, next_states, num_pts):
    state = env.reset()
    for i in range(num_pts):
        action = env.action_space.sample()
        next_state, _, _, _ = env.step(action) # take a random action
        states[i,...] = state
        actions[i,...] = action
        next_states[i,...] = next_state
        state = next_state
    env.close()
    return states, actions, next_states

def sample_within_range(env, states, actions, next_states, num_pts, input_range):
    state = env.reset()
    sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (num_pts,)+input_range.shape[:-1])
    obs_last_ind = states.shape[1]

    for i, sampled_input in enumerate(sampled_inputs):
        state, action, next_state = get_env_sample(env, sampled_input, obs_last_ind)
        states[i,...] = state
        actions[i,...] = action
        next_states[i,...] = next_state
    env.close()
    return states, actions, next_states

def collect(env_name='CartPole-v0', num_pts=100000, input_range=None):
    # open an env, run a bunch of actions and store a database of (s,a,s')
    env = gym.make(env_name)
    if isinstance(env.observation_space, gym.spaces.Dict):
        next_state_shape = env.observation_space['observation'].shape
    else:
        next_state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    state_shape = (input_range.shape[0] - action_shape[0],)
    states = np.empty((num_pts,) + state_shape)
    actions = np.empty((num_pts,) + action_shape)
    next_states = np.empty((num_pts,) + next_state_shape)

    # states, actions, next_states = random_sample(env, states, actions, next_states, num_pts)

    if input_range is None:
        input_range = np.array([ # (num_inputs, 2)
                          [np.pi/3, 2*np.pi/3], # x0min, x0max
                          [np.pi/3, 2*np.pi/3], # x1min, x1max
                          [np.pi/3, np.pi/3], # x1min, x1max
                          [np.pi/3, np.pi/3], # x1min, x1max
                          [0, 0], # amin, amax
        ])
    states, actions, next_states = sample_within_range(env, states, actions, next_states, num_pts, input_range)

    data = {
        'states': states,
        'actions': actions,
        'next_states': next_states,
    }

    with open(env_name+"_data.pkl", "wb") as f:
        pickle.dump(data, f)

def train(env_name='CartPole-v0', neurons_per_layer=[3,5], activation='relu', epochs=20):
    # open the database, create a keras model, train, save model
    with open(env_name+"_data.pkl", "rb") as f:
        data = pickle.load(f)

    states = data['states']
    actions = data['actions']
    next_states = data['next_states']

    if actions.ndim == 1:
        actions = np.expand_dims(actions, axis=-1)
    if states.ndim == 1:
        states = np.expand_dims(states, axis=-1)
    if next_states.ndim == 1:
        next_states = np.expand_dims(next_states, axis=-1)
    inputs = np.hstack([states, actions])
    outputs = next_states

    model = create_and_train_model(neurons_per_layer, inputs, outputs, verbose=1, activation=activation, epochs=epochs)
    save_model(model, env_name+"_model")

def create_model(neurons_per_layer, input_shape, output_shape, activation='relu'):
    model = Sequential()
    model.add(Dense(neurons_per_layer[0], input_shape=input_shape[1:], activation=activation))
    for neurons in neurons_per_layer[1:]:
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(output_shape[1]))
    model.compile(optimizer='rmsprop', loss='mean_absolute_percentage_error')
    return model

def create_and_train_model(neurons_per_layer, inputs, outputs, epochs=20, batch_size=32, verbose=0, activation='relu'):
    model = create_model(neurons_per_layer, inputs.shape, outputs.shape, activation=activation)
    model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

def save_model(model, name="model"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")
    print("Saved model to disk")

def load_model(name="model"):
    # load json and create model
    try:
        json_file = open(name+'.json', 'r')
        print('loaded '+ name)
    except:
        json_file = open('/Users/mfe/Downloads/model.json', 'r')
        print('loaded downloads')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    try:
        model.load_weights(name+".h5")
        print('loaded '+ name)
    except:
        model.load_weights("/Users/mfe/Downloads/model.h5")
        print('loaded downloads')
    print("Loaded model from disk")
    return model

def test(env_name='CartPole-v0', input_range=None, outputs_to_highlight=None, inputs_to_highlight=None):
    torch_model = model_dynamics(env_name)

    partitioner_hyperparams = {
        "type": "GreedySimGuided",
        "termination_condition_type": "num_propagator_calls",
        "termination_condition_value": 50,
        # "interior_condition": "lower_bnds",
        # "interior_condition": "linf",
        "interior_condition": "convex_hull",
        "make_animation": False,
        "show_animation": False,
    }
    propagator_hyperparams = {
        "type": "IBP_LIRPA",
        "input_shape": input_range.shape[:-1],
    }

    analyzer = DynamicsAnalyzer(torch_model, env_name)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams
    output_range, analyzer_info = analyzer.get_output_range(input_range)
    analyzer_info['outputs_to_highlight'] = outputs_to_highlight
    analyzer_info['inputs_to_highlight'] = inputs_to_highlight

    pars = '_'.join([str(key)+"_"+str(value) for key, value in partitioner_hyperparams.items() if key not in ["make_animation", "show_animation", "type"]])
    pars2 = '_'.join([str(key)+"_"+str(value) for key, value in propagator_hyperparams.items() if key not in ["input_shape", "type"]])
    input_range_str = '__'.join(['_'.join([str(ll) for ll in l]) for l in input_range])    
    analyzer_info["save_name"] = save_dir+env_name+'_'+partitioner_hyperparams['type']+"_"+propagator_hyperparams['type']+"_"+pars+"_"+pars2+"_"+"input_range"+input_range_str+".pdf"
    analyzer.visualize(input_range, output_range, **analyzer_info)

if __name__ == '__main__':
    np.random.seed(0)

    # env_name = 'CartPole-v0'
    # nominal_state = np.array([0., 0., 0., 0.]) # cart_pos, cart_vel, pole_ang, pole_ang_vel
    # state_uncertainty = np.zeros_like(nominal_state)
    # state_uncertainty[0:2] = 0.5
    # state_range = np.dstack([nominal_state-state_uncertainty, nominal_state+state_uncertainty])[0]
    # nominal_action = np.zeros((1,))
    # action_uncertainty = np.zeros_like(nominal_action)
    # action_range = np.dstack([nominal_action-action_uncertainty, nominal_action+action_uncertainty])[0]
    # input_range = np.vstack([state_range, action_range])

    env_name = 'Pendulum-v0'
    nominal_state = np.array([0., 0.]) # theta, thetadot
    state_uncertainty = np.zeros_like(nominal_state)
    state_uncertainty[0:2] = np.array([0.5, 0.1])
    state_range = np.dstack([nominal_state-state_uncertainty, nominal_state+state_uncertainty])[0]
    nominal_action = np.zeros((1,))
    action_uncertainty = np.array([0.1])
    action_range = np.dstack([nominal_action-action_uncertainty, nominal_action+action_uncertainty])[0]
    input_range = np.vstack([state_range, action_range])
    outputs_to_highlight = [
        {
            'dim': (1,),
            'name': r"NN Output: $\mathrm{sin}(\theta)$",
        },
        {
            'dim': (0,),
            'name': r"NN Output: $\mathrm{cos}(\theta)$",
        },
    ]
    inputs_to_highlight = [
        {
            'dim': (0,),
            'name': r"NN Input: $\theta$",
        },
        # {
        #     'dim': (1,),
        #     'name': r"NN Input: $\dot{\theta}$",
        # },
        {
            'dim': (2,),
            'name': r"NN Input: Motor Torque",
        },
    ]

    # env_name = 'HandManipulateBlock-v0'
    # nominal_state = np.array([-0.17429765, -0.19256321, -0.00017184,  0.76340606,  0.65941135,
    #     0.60334368,  0.00100149,  0.76291025,  0.65937461,  0.60330303,
    #     0.00315491,  0.73850659,  0.64370053,  0.58566994,  0.3424603 ,
    #    -0.00335944,  0.78199216,  0.66582407,  0.61014436,  0.0028371 ,
    #     0.58120427, -0.00714528, -0.00155769, -0.77619994, -0.0001154 ,
    #    -0.01654544,  0.00014496,  0.11174153,  0.55045009,  0.66218901,
    #    -0.00086641,  0.11708907,  0.55045439,  0.66218585,  0.02442837,
    #    -0.04898517,  0.41595827,  0.50335792,  0.10552045, -0.00845265,
    #     0.25397413,  0.59595225,  0.71478779,  0.00067487,  0.07929348,
    #    -0.00169097, -0.00040312, -0.10771166, -0.00522898,  0.01470529,
    #    -0.00565713,  0.13404697,  0.21461683,  0.3318141 ,  1.0111169 ,
    #     0.87614412,  0.16893923,  0.22959695,  0.43640261, -0.64513616,
    #    -0.58364145])
    # state_uncertainty = np.zeros_like(nominal_state)
    # state_uncertainty[0:2] = 0.5
    # state_range = np.dstack([nominal_state-state_uncertainty, nominal_state+state_uncertainty])[0]
    # nominal_action = np.zeros((20,))
    # action_uncertainty = np.zeros_like(nominal_action)
    # action_range = np.dstack([nominal_action-action_uncertainty, nominal_action+action_uncertainty])[0]
    # input_range = np.vstack([state_range, action_range])

    collect(env_name=env_name, input_range=input_range)
    train(env_name=env_name, neurons_per_layer=[3,5], activation='relu', epochs=20)
    test(env_name=env_name, input_range=input_range, outputs_to_highlight=outputs_to_highlight, inputs_to_highlight=inputs_to_highlight)

    # comparison(env_name=env_name)


def comparison(env_name="Pendulum-v0"):
    env_name = 'Pendulum-v0'
    nominal_state = np.array([0., 0.]) # theta, thetadot
    state_uncertainty = np.zeros_like(nominal_state)
    state_uncertainty[0:2] = np.array([0.5, 0.1])
    state_range = np.dstack([nominal_state-state_uncertainty, nominal_state+state_uncertainty])[0]
    nominal_action = np.zeros((1,))
    action_uncertainty = np.array([0.1])
    action_range = np.dstack([nominal_action-action_uncertainty, nominal_action+action_uncertainty])[0]
    input_range = np.vstack([state_range, action_range])
    outputs_to_highlight = [
        {
            'dim': (1,),
            'name': r"NN Output: $\mathrm{sin}(\theta)$",
        },
        {
            'dim': (0,),
            'name': r"NN Output: $\mathrm{cos}(\theta)$",
        },
    ]
    inputs_to_highlight = [
        {
            'dim': (0,),
            'name': r"NN Input: $\theta$",
        },
        # {
        #     'dim': (1,),
        #     'name': r"NN Input: $\dot{\theta}$",
        # },
        {
            'dim': (2,),
            'name': r"NN Input: Motor Torque",
        },
    ]

    torch_model = model_dynamics(env_name)

    partitioner_hyperparams = {
        "type": "GreedySimGuided",
        # "termination_condition_type": "verify",
        # "termination_condition_value": [np.array([1., 0., 0.]), np.array([0.5])],
        # "interior_condition": "lower_bnds",
        # "interior_condition": "linf",
        "interior_condition": "convex_hull",
        # "interior_condition": "verify",
        "make_animation": False,
        "show_animation": False,
    }
    propagator_hyperparams = {
        "type": "IBP_LIRPA",
        "input_shape": input_range.shape[:-1],
    }

    analyzer = DynamicsAnalyzer(torch_model, env_name)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams
    output_range, analyzer_info = analyzer.get_output_range(input_range)
    analyzer_info['outputs_to_highlight'] = outputs_to_highlight
    analyzer_info['inputs_to_highlight'] = inputs_to_highlight

    pars = '_'.join([str(key)+"_"+str(value) for key, value in partitioner_hyperparams.items() if key not in ["make_animation", "show_animation", "type"]])
    pars2 = '_'.join([str(key)+"_"+str(value) for key, value in propagator_hyperparams.items() if key not in ["input_shape", "type"]])
    input_range_str = '__'.join(['_'.join([str(ll) for ll in l]) for l in input_range])    
    analyzer_info["save_name"] = save_dir+env_name+'_'+partitioner_hyperparams['type']+"_"+propagator_hyperparams['type']+"_"+pars+"_"+pars2+"_"+"input_range"+input_range_str+".pdf"
    analyzer.visualize(input_range, output_range, **analyzer_info)    


