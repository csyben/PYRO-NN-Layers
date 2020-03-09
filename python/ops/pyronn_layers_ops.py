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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

import pyronn_layers 

pyronn_layers_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_pyronn_layers_ops.so'))
for obj in dir(pyronn_layers_ops):
        setattr(pyronn_layers, obj, getattr(pyronn_layers_ops, obj))
