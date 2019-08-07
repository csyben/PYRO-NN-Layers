# Copyright [2019] [Christopher Syben]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Makes every implemented operator in python available under the namespace pyronn_layers
# PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
#
import os.path
import tensorflow as tf
import pyronn_layers


if tf.test.is_built_with_cuda():
    _pyronn_layers_module = tf.load_op_library(os.path.dirname(__file__)+'/pyronn_layers.so')
    ''' TODO: Improve the getattr method to add only real kernel methods and not everything '''
    for obj in dir(_pyronn_layers_module):
        setattr(pyronn_layers, obj, getattr(_pyronn_layers_module, obj))


