from nn_closed_loop.utils.nn import save_model
import tensorflow as tf
import os
import torch
from PIL import Image
import numpy as np
# import torch.nn.modules.batchnorm as batchnorm
import onnx
from onnx2torch import convert
from crown_ibp.conversions.keras2torch import keras2torch
# from onnx2pytorch import ConvertModel


# import onnx
# from onnx2keras import onnx_to_keras
# from onnx2pytorch import ConvertModel

from tensorflow.python.platform import gfile

dir_path = os.path.dirname(os.path.realpath(__file__))

stride = 16             # Size of square of pixels downsampled to one grayscale value
numPix = 16             # During downsampling, average the numPix brightest pixels in each square
width  = 256//stride    # Width of downsampled grayscale image
height = 128//stride    # Height of downsampled grayscale image
# GRAPH_PB_PATH = dir_path+'/../../models/Taxinet/from_pb/saved_model.pb'
# ONNX_PATH = dir_path+'/../../models/Taxinet/from_pb/TinyTaxiNet.onnx'

def main():
    
    path = dir_path+'/TinyTaxiNet.pt'
    onnx_model = onnx.load('./TinyTaxiNet.onnx')
    model = convert(onnx_model)

    ez_image = Image.open(dir_path+"/downsampled_images/smallsubset_1traj/MWH_Runway04_morning_overcast_1_2000.png")
    ez_arr = np.array(ez_image)[:,:,0]/255.0
    test_input = torch.from_numpy(np.array(ez_arr.flatten(), dtype='float32'))
    pred = model(test_input)
    input_set = np.vstack((test_input-0.001, test_input+0.001))

    from crown_ibp.bound_layers import BoundSequential
    

    # ktest_input = []
    # for i in range(128):
    #     ktest_input.append(tf.convert_to_tensor(ez_arr.flatten()[i]))
    # import pdb; pdb.set_trace()

    

    keras_model = tf.keras.models.load_model(dir_path+'/TinyTaxiNet.h5')
    torch_model = keras2torch(keras_model, "converted_model")
    pred2 = model(test_input)

    network = BoundSequential.convert(
        torch_model, {"zero-lb": True}
    )

    x_max = np.array(ez_arr.flatten()+0.001, dtype='float32')
    x_min = np.array(ez_arr.flatten()-0.001, dtype='float32')

    x_max_torch = torch.Tensor([x_max])
    x_min_torch = torch.Tensor([x_min])

    method_opt = 'full_backward_range'

    # Compute the NN output matrices (for this xt partition)
    C = torch.eye(2).unsqueeze(0)
    lower_A, upper_A, lower_sum_b, upper_sum_b = network(
        method_opt=method_opt,
        norm=np.inf,
        x_U=x_max_torch,
        x_L=x_min_torch,
        upper=True,
        lower=True,
        C=C,
        return_matrices=True,
    )
    upper_A = upper_A.detach().numpy()[0]
    lower_A = lower_A.detach().numpy()[0]
    upper_sum_b = upper_sum_b.detach().numpy()[0]
    lower_sum_b = lower_sum_b.detach().numpy()[0]


    import pdb; pdb.set_trace()






    # with Image.open(dir_path+"/original_images/smallsubset_1traj/MWH_Runway04_morning_overcast_1_2000.png") as im:
    #     img = np.array(im)

    #     # Remove yellow/orange lines
    #     mask = ((img[:,:,0].astype('float')-img[:,:,2].astype('float'))>60) & ((img[:,:,1].astype('float')-img[:,:,2].astype('float'))>30) 
    #     img[mask] = 0
        
    #     # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so 
    #     # values range between 0 and 1
    #     # img = np.array(Image.fromarray(img).convert('L').crop(
    #     #     (55, 5, 360, 135)).resize((256, 128)))/255.0
    #     img = np.array(Image.fromarray(img).convert('L').resize((256, 128)))/255.0


    #     # Downsample image
    #     # Split image into stride x stride boxes, average numPix brightest pixels in that box
    #     # As a result, img2 has one value for every box
    #     img2 = np.zeros((height,width))
    #     for i in range(height):
    #         for j in range(width):
    #             img2[i,j] = np.mean(np.sort(img[stride*i:stride*(i+1),stride*j:stride*(j+1)].reshape(-1))[-numPix:])

    #     test_input = torch.from_numpy(np.array(img2.flatten(), dtype='float32'))
    #     pred = model(test_input)
    #     import pdb; pdb.set_trace()
    #     print('ok')
    # model = torch.load(path)
    # pytorch_model = ConvertModel(onnx_model)
    import pdb; pdb.set_trace()

    # if not os.path.isfile(path+'model.h5'):
        
    #     input_list = ['{}'.format(i) for i in range(128)]
    #     onnx_model = onnx.load(ONNX_PATH)
    #     input_all = [node.name for node in onnx_model.graph.output]
    #     print('here I am')
    #     print(input_all)
    #     pytorch_model = ConvertModel(onnx_model)
    #     import pdb; pdb.set_trace()
    #     k_model = onnx_to_keras(onnx_model, ['X'])

    #     import pdb; pdb.set_trace()


    #     # tf.compat.v1.disable_v2_behavior()
    #     # with tf.compat.v1.Session() as sess:
    #     #     print("load graph")
    #     #     with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    #     #         graph_def = tf.compat.v1.GraphDef()
    #     #         graph_def.ParseFromString(f.read())
    #     #         sess.graph.as_default()
    #     #         tf.import_graph_def(graph_def, name='')
    #     #         graph_nodes=[n for n in graph_def.node]
        
    #     # wts = [n for n in graph_nodes if n.op=='Const']
    #     # from tensorflow.python.framework import tensor_util
    #     # from keras.models import Sequential
    #     # from keras.layers import Dense, Activation

    #     # keras_model = Sequential()
    #     # keras_model.add(Dense())

    #     for n in wts:
    #         print("Name of the node - %s" % n.name)
    #         print("Value - ")
            
    #         print(tensor_util.MakeNdarray(n.attr['value'].tensor).shape)
    #         print(tensor_util.MakeNdarray(n.attr['value'].tensor))
    #     import pdb; pdb.set_trace()

    #     # old_model = tf.saved_model.load(dir_path+'/../../models/Taxinet/from_pb')
    #     # tf.keras.models.save_model(old_model, dir_path+'/../../models/Taxinet/to_pb/', save_format='h5')




    #     # tf.saved_model.save(old_model, dir_path+'/../../models/Taxinet/to_pb/')
        
    #     # tf.compat.v1.disable_v2_behavior()

    #     model = tf.keras.models.load_model(dir_path+'/../../models/Taxinet/to_pb')
    #     model.save("my_model")
    #     # tf.keras.models.save_model(model, dir_path+'/../../models/Taxinet/to_h5/model.h5'),
    #     import pdb; pdb.set_trace()
    #     save_model(model, system="Taxinet", model_name="default")
    #     import pdb; pdb.set_trace()
    
    # print('haha')


if __name__ == "__main__":
    main()
