from os.path import dirname, join, exists
import tensorflow as tf
import subprocess


if tf.test.is_built_with_cuda():
    shared_object = join(dirname(__file__), '../../bazel-bin/pyronn_layers/pyronn_layers.so')
    if not exists(shared_object):
        print('Trying to build bazel-bin/pyronn_layers/pyronn_layers.so')
        subprocess.run(['bazel', 'build', '//pyronn_layers:pyronn_layers.so'])


    _pyronn_layers_module = tf.load_op_library(shared_object)

        
    print('Loading shared object...')
    for obj in dir(_pyronn_layers_module):
        globals()[obj] = getattr(_pyronn_layers_module, obj)
        print('\t' + str(obj))
else:
    print('Not built with CUDA')
