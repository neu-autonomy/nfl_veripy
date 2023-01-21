from nn_closed_loop.nn_closed_loop.utils.nn import save_model
import tensorflow as tf
import os
import torch

import onnx
from onnx2keras import onnx_to_keras
from onnx2pytorch import ConvertModel

from tensorflow.python.platform import gfile

dir_path = os.path.dirname(os.path.realpath(__file__))
GRAPH_PB_PATH = dir_path+'/../../models/Taxinet/from_pb/saved_model.pb'
ONNX_PATH = dir_path+'/../../models/Taxinet/from_pb/TinyTaxiNet.onnx'

def main():
    
    path = dir_path+'/../../models/Taxinet/default/'
    if not os.path.isfile(path+'model.h5'):
        
        input_list = ['{}'.format(i) for i in range(128)]
        onnx_model = onnx.load(ONNX_PATH)
        input_all = [node.name for node in onnx_model.graph.output]
        print('here I am')
        print(input_all)
        pytorch_model = ConvertModel(onnx_model)
        import pdb; pdb.set_trace()
        k_model = onnx_to_keras(onnx_model, ['X'])

        import pdb; pdb.set_trace()


        # tf.compat.v1.disable_v2_behavior()
        # with tf.compat.v1.Session() as sess:
        #     print("load graph")
        #     with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        #         graph_def = tf.compat.v1.GraphDef()
        #         graph_def.ParseFromString(f.read())
        #         sess.graph.as_default()
        #         tf.import_graph_def(graph_def, name='')
        #         graph_nodes=[n for n in graph_def.node]
        
        # wts = [n for n in graph_nodes if n.op=='Const']
        # from tensorflow.python.framework import tensor_util
        # from keras.models import Sequential
        # from keras.layers import Dense, Activation

        # keras_model = Sequential()
        # keras_model.add(Dense())

        for n in wts:
            print("Name of the node - %s" % n.name)
            print("Value - ")
            
            print(tensor_util.MakeNdarray(n.attr['value'].tensor).shape)
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        import pdb; pdb.set_trace()

        # old_model = tf.saved_model.load(dir_path+'/../../models/Taxinet/from_pb')
        # tf.keras.models.save_model(old_model, dir_path+'/../../models/Taxinet/to_pb/', save_format='h5')




        # tf.saved_model.save(old_model, dir_path+'/../../models/Taxinet/to_pb/')
        
        # tf.compat.v1.disable_v2_behavior()

        model = tf.keras.models.load_model(dir_path+'/../../models/Taxinet/to_pb')
        model.save("my_model")
        # tf.keras.models.save_model(model, dir_path+'/../../models/Taxinet/to_h5/model.h5'),
        import pdb; pdb.set_trace()
        save_model(model, system="Taxinet", model_name="default")
        import pdb; pdb.set_trace()
    
    print('haha')


if __name__ == "__main__":
    main()
