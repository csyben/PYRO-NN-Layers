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
To achieve this, the PRYO-NN-Layers repository need to be cloned into a the 'pyronn_layers' subfolder withing the Tensorflow source directory:

.. code-block:: bash

    git clone https://github.com/csyben/PYRO-NN-Layers pyronn_layers





Tests
=====


Changelog
=========

Can be found `CHANGELOG.md `_.
