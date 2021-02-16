import julia
julia.install()
import julia.NeuralVerification as NV
import numpy as np
import torch

def weights_to_txt(weights, f):
    num_rows = weights.shape[0]
    for row in weights:
        f.write(','.join(map(str, row))+"\n")

def bias_to_txt(bias, f):
    for i in range(0,len(bias)):
        f.write(str(bias[i])+"\n")

def torch2julianet(torch_model):

    dims = []
    act = None
    for idx, m in enumerate(torch_model.modules()):
        if isinstance(m, torch.nn.Sequential): continue
        elif isinstance(m, torch.nn.ReLU):
            if act is None or act == 'relu':
                act = 'relu'
            else:
                print('Multiple types of activations in your model --- unsuported by robust_sdp.')
                assert(0)
        elif isinstance(m, torch.nn.Linear):
            dims.append(m.in_features)
        else:
            print("That layer isn't supported.")
            assert(0)
    dims.append(m.out_features)
    if len(dims) != 3:
        print("robust sdp only supports 1 hidden layer (for now).")
        assert(0)

    tmp_filename = "tmp.txt"
    with open(tmp_filename, "w+") as f:
        f.write(str(dims[0])+"\n")
        f.write(str(dims)[1:-1]+"\n")
        for i in range(5):
            f.write("0\n")

        for name, param in torch_model.named_parameters():
            layer, typ = name.split('.')
            layer = int(int(layer)/2)
            if typ == 'weight':
                weights_to_txt(param.data.numpy(), f)
            elif typ == 'bias':
                bias_to_txt(param.data.numpy(), f)
            else:
                print('this layer isnt a weight or bias ???')
    net = NV.read_nnet(tmp_filename)
    return net

# nnet = read_nnet("/Users/mfe/.julia/packages/NeuralVerification/IdHBn/examples/networks/cartpole_nnet.nnet")
# input_range = np.array([
#     [-0.01, 0.01],
#     [-0.01, 0.01],
#     [-0.01, 0.01],
#     [-0.01, 0.01],
#     ])
# output_set = Hyperrectangle(low = [0., 0.], high = [0., 0.])
# num_outputs = 2

def query_model(net, data=None):
    if data is None:
        data = np.linspace(input_range[:,0], input_range[:,1], num=50)
    out = []
    for d in data:
        out.append(NV.compute_output(net, d))
    out = np.stack(out, axis=0)
    return out

def julia_output_range(net=None, input_range=None, verbose=True, bound_method="MaxSens"):

    if net is None:
        # net = NV.read_nnet("/Users/mfe/.julia/packages/NeuralVerification/IdHBn/examples/networks/small_nnet.nnet")
        # input_range = np.array([
        #     [-0.1, 0.1]
        #     ])
        # num_outputs = 1

        net = NV.read_nnet("tmp.txt")
        input_range = np.array([
            [-0.1, 0.1],
            [-0.1, 0.1],
            ])

    num_outputs = net.layers[-1].weights.shape[0]

    # TODO(MFE): I can't figure out how to set the resolution s.t. this code
    # will use a single input set (not splitting into 4)
    solver = getattr(NV, bound_method)(resolution=0.1)
    input_set  = NV.Hyperrectangle(low = input_range[:,0], high = input_range[:,1])
    output_set = NV.Hyperrectangle(low = np.zeros((num_outputs,)), high = np.zeros((num_outputs,)))
    problem = NV.Problem(net, input_set, output_set)
    result = NV.solve(solver, problem)
    num_boxes = len(result.reachable)
    radii = np.stack([r.radius for r in result.reachable], axis=0)
    center = np.stack([r.center for r in result.reachable], axis=0)

    output_range = np.empty((num_outputs, 2))
    if num_boxes == 1:
        output_range[:,0] = center - radii
        output_range[:,1] = center + radii
    else:
        output_range[:,0] = np.min(center - radii, axis=0)
        output_range[:,1] = np.max(center + radii, axis=0)

    return output_range

if __name__ == '__main__':
    output_range = julia_output_range()
    print(output_range)

