from os.path import dirname, join, exists
import tensorflow as tf
import subprocess


if tf.test.is_built_with_cuda():
    shared_object = join(dirname(__file__), '../../bazel-bin/lme_custom_ops/lme_custom_ops.so')
    if not exists(shared_object):
        print('Trying to build bazel-bin/lme_custom_ops/lme_custom_ops.so')
        subprocess.run(['bazel', 'build', '//lme_custom_ops:lme_custom_ops.so'])


    _lme_custom_ops_module = tf.load_op_library(shared_object)

        
    print('Loading shared object...')
    for obj in dir(_lme_custom_ops_module):
        globals()[obj] = getattr(_lme_custom_ops_module, obj)
        print('\t' + str(obj))
else:
    print('Not built with CUDA')
