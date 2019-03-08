import os.path
import tensorflow as tf
import lme_custom_ops


if tf.test.is_built_with_cuda():
    _lme_custom_ops_module = tf.load_op_library(os.path.join(
        tf.resource_loader.get_data_files_path(), 'lme_custom_ops.so'))
    ''' TODO: Improve the getattr method to add only real kernel methods and not everything '''
    for obj in dir(_lme_custom_ops_module):
        setattr(lme_custom_ops, obj, getattr(_lme_custom_ops_module, obj))



#
 # Makes every implemented operator in python available under the namespace PyRo-ML
 # PyRo-ML is developed as an Open Source project under the GNU General Public License (GPL).
 # Copyright (C) 2019  Christopher Syben
#