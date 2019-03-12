FRAMEWORK
==========

PYRO-NN-Layers
========

Python Reconstruction Operators in Machine Learning (PYRO-NN-Layers) brings state-of-the-art reconstruction algorithm to
neuralnetwork integrated into Tensorflow. This Repository contains the actual Layer implementation as CUDA kernels and 
the necessary C++ Information Control Classes according to Tensorflow API.

For convinient use of the layers also install https://github.com/csyben/PYRO-NN

The publication can be found under (https://frameworkpaper)


Installation
============

To build the reconstruction operators into the Tensorflow package, the Tensorflow sources need to be prepared.

To build the sources following tools are necessary: Python, Bazel, CUDA.
Please prepare the system according to the Tensorflow 'Build from sources' guidlines: https://www.tensorflow.org/install/source . 
 
If all necessary tools are installed the build process can start:

First clone the Tensorflow repository:

.. code-block:: bash

    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow

Checkout a release branch from Tensorflow:

.. code-block:: bash

    git checkout branch_name  # r1.9, r1.12, etc.

Now the reconstruction operators need to be added to the build process.
To achieve this, the PRYO-NN-Layers repository need to be cloned into a 'pyronn_layers' subfolder withing the Tensorflow source directory:

.. code-block:: bash

    git clone https://github.com/csyben/PYRO-NN-Layers pyronn_layers

Next step is to patch the Tensorflow build process such that all C++ and CUDA files in the pyronn_layers folder are compiled and
made available under the pyronn_layers namespace at the python level. Select the respective patch for the choosen release version of Tensorflow.

.. code-block:: bash

    cd pyronn_layers/patches/
    python3 patch_tf_1_12.py

Now everything is setup to build Tensorflow and the reconstruction operators. For this change back to the source directory of Tensorflow. 

.. code-block:: bash

    cd ../..

The Tensorflow build process need to be configured, for that type:

.. code-block:: bash

    ./configure

For a detailed description follow the Tensorflow guidlines itself (https://www.tensorflow.org/install/source). 
In short, choose the python interpreter and the CUDA version which is used to create the package.

After the confguration the sources can be compiled with

.. code-block:: bash

    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

The pip_package can be then build with 

.. code-block:: bash

    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./pip_package/

The Tensorflow wheele file including the reconstruction operators can be found in the pip_package folder.
This wheele package can be now installed via pip:

.. code-block:: bash

    pip3 install ./pip_package/<FileName>

Now verything is setup and the reconstruction operators can be found under pyronn_layers namespace. 
For a more convinient use of these operators the pyronn pip package is provided under:

https://github.com/csyben/PYRO-NN

or use

.. code-block:: bash

    pip3 install pyronn

Potential Challenges
====================




Changelog
=========

Can be found `CHANGELOG.md `_.
