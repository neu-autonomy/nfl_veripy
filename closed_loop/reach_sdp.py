import numpy as np
import cvxpy as cp
import itertools
from tqdm import tqdm
from torch.nn import Linear

def getE_in(num_states, num_neurons, num_inputs):
    # Set up E_in to change the basis of P to NN coordinates
    E_in = np.zeros((num_states+1, num_states+num_neurons+2*num_inputs+1))
    E_in[:num_states, :num_states] = np.eye(num_states)
    E_in[-1, -1] = 1
    return E_in

def getE_out(num_states, num_neurons, num_inputs, At, bt, ct):
    # Set up E_out to change the basis of S_i to NN coordinates
    E_out = np.zeros((num_states+1, num_states+num_neurons+2*num_inputs+1))
    E_out[:num_states, :num_states] = At
    E_out[:num_states, -1-num_inputs:-1] = bt
    E_out[:num_states, -1] = ct
    E_out[-1,-1] = 1
    return E_out

def getE_mid(num_states, num_neurons, num_inputs, model, u_min, u_max):
    # Set up E_mid to change the basis of Q to NN coordinates

    # Keras:
    # Ws = model.get_weights()

    # Pytorch:
    weights = [layer.weight.data.numpy().T for layer in model if isinstance(layer, Linear)]
    biases = [layer.bias.data.numpy() for layer in model if isinstance(layer, Linear)]
    Ws = [None]*(len(weights)+len(biases))
    Ws[::2] = weights
    Ws[1::2] = biases

    num_layers = int(len(Ws)/2)

    A_ = np.zeros((num_neurons+2*num_inputs,num_states+num_neurons+2*num_inputs))
    B_ = np.zeros((num_neurons+2*num_inputs,num_states+num_neurons+2*num_inputs))
    a_ = np.zeros((num_neurons+2*num_inputs,))
    b_ = np.zeros((num_neurons+2*num_inputs,))
    i = 0; j = 0
    for layer in range(num_layers):
        W_i = Ws[2*layer].T
        b_i = Ws[2*layer+1]

        A_[i:i+W_i.shape[0], j:j+W_i.shape[1]] = W_i

        a_[i:i+W_i.shape[0]] = b_i

        i += W_i.shape[0]; j += W_i.shape[1]

    A_[i:i+num_inputs, j:j+num_inputs] = -np.eye(num_inputs)

    a_[-2*num_inputs:-num_inputs] -= u_min
    a_[-num_inputs:] = u_max
    b_[-2*num_inputs:-num_inputs] = -u_min
    b_[-num_inputs:] = u_max

    a_ = np.expand_dims(a_, axis=-1)
    b_ = np.expand_dims(b_, axis=-1)

    B_[-num_inputs:, -num_inputs:] = -np.eye(num_inputs)
    B_[-2*num_inputs:-num_inputs, -2*num_inputs:-num_inputs] = np.eye(num_inputs)
    B_[0:num_neurons, num_states:num_states+num_neurons] = np.eye(num_neurons)

    E_mid = np.vstack([
            np.hstack([A_, a_]),
            np.hstack([B_, b_]),
            np.zeros((1,A_.shape[1]+a_.shape[1]))
            ])
    E_mid[-1,-1] = 1
    return E_mid

def mat_constr(i,j,val,mat):
    d = mat.shape[0]
    Xtr = np.zeros((d,d))
    Xtr[j,i]=1
    return cp.trace(mat @ Xtr) == val

def getInputConstraints(num_states, m, A_inputs, b_inputs):
    """ Set up M_in(P) which describes the input constraint """
    # Set up P, the polyhedron constraint in state coordinates
    P = cp.Variable((num_states+1, num_states+1), symmetric=True)
    Gamma = cp.Variable((m,m), symmetric=True)
    Gamma2 = cp.Variable((m,m), nonneg=True)

    input_set_constrs = []
    for i,j in itertools.combinations_with_replacement(range(m),2):
        input_set_constrs += [
        # Ensure each term in Gamma >= 0
            mat_constr(i=i,j=j,val=Gamma2[i,j],mat=Gamma)
        ]
    input_set_constrs += [
        mat_constr(i=0,j=0,val=cp.quad_form(A_inputs, Gamma)[0,0],mat=P),
        mat_constr(i=0,j=1,val=cp.quad_form(A_inputs, Gamma)[0,1],mat=P),
        mat_constr(i=1,j=1,val=cp.quad_form(A_inputs, Gamma)[1,1],mat=P),
        mat_constr(i=0,j=2,val=(-A_inputs.T*Gamma*b_inputs)[0],mat=P),
        mat_constr(i=1,j=2,val=(-A_inputs.T*Gamma*b_inputs)[1],mat=P),
        mat_constr(i=2,j=2,val=cp.quad_form(b_inputs, Gamma),mat=P),
    ]
    return P, input_set_constrs


def getNNConstraints(num_neurons, num_inputs):
    """ Set up M_mid(Q) which describes the ReLU constraint """
    nn_constrs = []

    # Set up T
    d = num_neurons+2*num_inputs
    T = cp.Variable((d,d), PSD=True)
    lamb_ij = cp.Variable((d,d), nonneg=True)
    lamb_i = cp.Variable((d))
    pairs = list(itertools.combinations(range(d), 2))

    for i in range(d):
        if i == 0:
            first = 0
        else:
            first = cp.sum(lamb_ij[:i,i])
        if i == d-1:
            second = 0
        else:
            second = cp.sum(lamb_ij[i, i+1:])
        val = first+second+lamb_i[i]
        nn_constrs += [mat_constr(i=i,j=i,val=val,mat=T)]

    for i,j in pairs:
        val = -lamb_ij[i,j]
        nn_constrs += [mat_constr(i=i,j=j,val=val,mat=T)]

    # Set up Q (consists of T, eta, nu)
    Q = cp.Variable((2*d+1, 2*d+1))
    eta = cp.Variable((d), nonneg=True)
    nu = cp.Variable((d), nonneg=True)

    # Zero block
    nn_constrs += [mat_constr(i=i,j=j,val=0,mat=Q) for (i,j) in itertools.product(range(d), repeat=2)]
    # T block (top middle)
    nn_constrs += [mat_constr(i=i,j=j+d,val=T[i,j],mat=Q) for (i,j) in itertools.product(range(d), repeat=2)]
    # T block (left middle)
    nn_constrs += [mat_constr(i=i+d,j=j,val=T[i,j],mat=Q) for (i,j) in itertools.product(range(d), repeat=2)]
    # -2T block (middle)
    nn_constrs += [mat_constr(i=i+d,j=j+d,val=-2*T[i,j],mat=Q) for (i,j) in itertools.product(range(d), repeat=2)]
    # -v block (top right)
    nn_constrs += [mat_constr(i=i,j=-1,val=-nu[i],mat=Q) for i in range(d)]
    # v+nu block (middle right)
    nn_constrs += [mat_constr(i=i+d,j=-1,val=(nu+eta)[i],mat=Q) for i in range(d)]
    # zero block (bottom right)
    nn_constrs += [mat_constr(i=-1,j=-1,val=0,mat=Q)]
    # vT+nuT block (bottom middle)
    nn_constrs += [mat_constr(i=-1,j=i+d,val=(nu+eta)[i],mat=Q) for i in range(d)]
    # -vT block (bottom left)
    nn_constrs += [mat_constr(i=-1,j=i,val=-nu[i],mat=Q) for i in range(d)]
    return Q, nn_constrs

def getOutputConstraints(num_states, a_i):
    """ Set up M_out(S_i) which describes one halfplane of the reachable set """

    # Set up S_i for one halfplane in state coordinates
    b_i = cp.Variable(1)
    S_i = cp.Variable((num_states+1, num_states+1), symmetric=True)

    reachable_set_constrs = []
    reachable_set_constrs += [
        mat_constr(i=0,j=0,val=0,mat=S_i),
        mat_constr(i=0,j=1,val=0,mat=S_i),
        mat_constr(i=1,j=1,val=0,mat=S_i),
        mat_constr(i=0,j=2,val=a_i[0],mat=S_i),
        mat_constr(i=1,j=2,val=a_i[1],mat=S_i),
        mat_constr(i=2,j=2,val=-2*b_i,mat=S_i),
    ]
    return S_i, reachable_set_constrs, b_i


def reachSDP_1(model, A_inputs, b_inputs, At, bt, ct, A_in, u_min, u_max):

    # Count number of units in each layer, except last layer
    num_neurons = np.sum([layer.get_config()['units'] for layer in model.layers][:-1])

    # Number of vertices in input polyhedron
    m = A_inputs.shape[0]
    num_states = At.shape[0]
    num_inputs = bt.shape[1]

    # Get change of basis matrices
    E_in = getE_in(num_states, num_neurons, num_inputs)
    E_mid = getE_mid(num_states, num_neurons, num_inputs, model, u_min, u_max)
    E_out = getE_out(num_states, num_neurons, num_inputs, At, bt, ct)

    # Get P,Q,S and constraint lists
    P, input_set_constrs = getInputConstraints(num_states, m, A_inputs, b_inputs)
    Q, nn_constrs = getNNConstraints(num_neurons, num_inputs)

    # M_in describes the input set in NN coords
    M_in = cp.quad_form(E_in, P)
    M_mid = cp.quad_form(E_mid, Q)

    num_facets = A_in.shape[0]
    bs = np.zeros((num_facets))
    for i in tqdm(range(num_facets)):
        S_i, reachable_set_constrs, b_i = getOutputConstraints(num_states, A_in[i,:])
        M_out = cp.quad_form(E_out, S_i)

        constraints = input_set_constrs + nn_constrs + reachable_set_constrs

        constraints.append(M_in + M_mid + M_out << 0)

        objective = cp.Minimize(b_i)
        prob = cp.Problem(objective,
                          constraints)
        prob.solve()
        # print("status:", prob.status)
        bs[i] = b_i.value

    # print("status:", prob.status)
    # print("optimal value", prob.value)
    # print("b_{i}: {val}".format(i=i, val=b_i.value))
    # print("S_{i}: {val}".format(i=i, val=S_i.value))

    return bs

def reachSDP_n(n, model, A_inputs, b_inputs, At, bt, ct, A_in, u_min, u_max):
    all_bs = []
    bs = reachSDP_1(model, A_inputs, b_inputs, At, bt, ct, A_in, u_min, u_max)
    all_bs.append(bs)
    for i in range(1,n):
        bs = reachSDP_1(model, A_in, bs, At, bt, ct, A_in, u_min, u_max)
        all_bs.append(bs)
    return all_bs

def save_dataset(bs):
    with open("bs.pkl", "wb") as f:
        pickle.dump(bs, f)

def load_dataset():
    with open("bs.pkl", "rb") as f:
        bs = pickle.load(f)
    return bs

# save_dataset(all_bs)
# all_bs = load_dataset()

if __name__ == '__main__':
    # Reach SDP
    np.set_printoptions(precision=1)

    # Initial state constraints
    x0_min, x0_max = init_state_range[0,:]
    x1_min, x1_max = init_state_range[1,:]
    # A_inputs: vectors of the facets of the input reachable set polyhedron
    A_inputs = np.array([[-1, 0],
                         [1, 0],
                         [0, -1],
                         [0, 1]
    ])
    b_inputs = np.array([-x0_min, x0_max, -x1_min, x1_max])

    # A_in: vectors of the facets of the output reachable set polyhedron
    A_in = np.array([[1,-1,0,0,1,-1,1,-1],[0,0,1,-1,-1,1,1,-1]]).T

    bs = reachSDP_1(model, A_inputs, b_inputs, At, bt, ct, A_in, u_min, u_max)
    # bs2 = reachSDP_1(model, A_in, bs, At, bt, ct, A_in)

    # all_bs = reachSDP_n(6, model, A_inputs, b_inputs, At, bt, ct, A_in)






