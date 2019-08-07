FRAMEWORK
==========

PYRO-NN-Layers
========

Python Reconstruction Operators in Machine Learning (PYRO-NN-Layers) brings state-of-the-art reconstruction algorithms to
neural networks integrated into Tensorflow. This repository contains the actual Layer implementation as CUDA kernels and 
the necessary C++ information control classes according to the Tensorflow API.

For convenient use of the layers also install https://github.com/csyben/PYRO-NN

If you find this helpful, we would kindly ask you to reference our article.
The arXiv preprint can be found under https://arxiv.org/abs/1904.13342


Installation
============

To build the reconstruction operators into the Tensorflow package, the Tensorflow sources need to be prepared.

To build the sources following tools are necessary: Python, Bazel, CUDA.
Please prepare the system according to the Tensorflow 'Build from sources' guidelines: https://www.tensorflow.org/install/source . 
 
If all necessary tools are installed the build process can start:

First, clone the Tensorflow repository:

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
    python3  patch_version # patch_tf_1_9.py, patch_tf_1_12.py, etc.

Now everything is setup to build Tensorflow and the reconstruction operators. For this change back to the source directory of Tensorflow. 

.. code-block:: bash

    cd ../..

The Tensorflow build process need to be configured, for that type:

.. code-block:: bash

    ./configure

For a detailed description follow the Tensorflow guidelines itself (https://www.tensorflow.org/install/source). 
In short, choose the python interpreter and the CUDA version which sould be used to create the package.

After the confguration the sources can be compiled with

.. code-block:: bash

    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

The pip_package can be then build with 

.. code-block:: bash

    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./pip_package/

The Tensorflow wheel file including the reconstruction operators can be found in the pip_package folder.
This wheel package can be now installed via pip:

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

Memory consumption on the graphics card can be a problem with CT datasets. For the reconstruction operators the input data is passed via a Tensorflow tensor,
which is already allocated on the graphicscard by Tensorflow itself. In fact without any manual configuration Tensorflow will allocate most of
the graphics card memory and handle the memory management internally. This leads to the problem that CUDA malloc calls in the operators itself will allocate
memory outside of the Tensorflow context, which can easily lead to out of memory errors, although the memory is not full.

There exist two ways of dealing with this problem:

1. A convenient way is to reduce the initially allocated memory by Tensorflow itself and allow a memory growth. We suggest to always use this mechanism 
to minimize the occurrence of out of memory errors:

.. code-block:: python

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    # ------------------ Call Layers ------------------
    with tf.Session(config=config) as sess:
        ...

2. The memory consuming operators like 3D cone-beam projection and back-projection have a so called hardware_interp flag. This means that the
interpolation for both operators are either done by the CUDA texture or based on software interpolation. To use the CUDA texture, 
and thus have a fast hardware_interpolation, the input data need to be copied into a new CUDA array, thus consuming the double amount of memory. 
In the case of large data or deeper networks it could be favorable to switch to the software interpolation mode. In this case the actual Tensorflow pointer
can directly be used in the kernel without any duplication of the data. The downside is that the interpolation takes nearly 10 times longer.



Changelog
=========

Can be found `CHANGELOG.md <https://github.com/csyben/PYRO-NN-Layers/blob/master/CHANGELOG.md>`_.


Reference
=========

`PYRO-NN: Python Reconstruction Operators in Neural Networks <https://arxiv.org/abs/1904.13342>`_.

Applications
============
.. [GCPR2018] `Deriving Neural Network Architectures using Precision Learning: Parallel-to-fan beam Conversion <https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2018/Syben18-DNN.pdf>`_.
.. [CTMeeting18] `Precision Learning: Reconstruction Filter Kernel Discretization <https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2018/Syben18-PLR.pdf>`_.



