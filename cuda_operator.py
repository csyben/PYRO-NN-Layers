import os.path
import tensorflow as tf
import lme_custom_ops


print("HELLO from lme_custom_op operator")

if tf.test.is_built_with_cuda():
    _lme_custom_ops_module = tf.load_op_library(os.path.join(
        tf.resource_loader.get_data_files_path(), 'lme_custom_ops.so'))

    for obj in dir(_lme_custom_ops_module):
        setattr(lme_custom_ops, obj, getattr(_lme_custom_ops_module, obj))
