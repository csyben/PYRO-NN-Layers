import os.path
import tensorflow as tf
import pyronn_layers


if tf.test.is_built_with_cuda():
    _pyronn_layers_module = tf.load_op_library(os.path.join(
        tf.resource_loader.get_data_files_path(), 'pyronn_layers.so'))
    ''' TODO: Improve the getattr method to add only real kernel methods and not everything '''
    for obj in dir(_pyronn_layers_module):
        setattr(pyronn_layers, obj, getattr(_pyronn_layers_module, obj))



#
 # Makes every implemented operator in python available under the namespace pyronn_layers
 # PYRO-NN is developed as an Open Source project under the GNU General Public License (GPL).
#