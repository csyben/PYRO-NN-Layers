FRAMEWORK
==========

PYRO-NN-Layers
========

Python Reconstruction Operators in Machine Learning (PYRO-NN-Layers) brings state-of-the-art reconstruction algorithms to
neural networks integrated into Tensorflow. This repository contains the actual Layer implementation as CUDA kernels and 
the necessary C++ information control classes according to the Tensorflow API.

For convenient use of the layers also install https://github.com/csyben/PYRO-NN

Open access paper available under:
https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.13753

If you find this helpful, we would kindly ask you to reference our article published by medical physics:
.. code-block:: 

   @article{PYRONN2019,
   author = {Syben, Christopher and Michen, Markus and Stimpel, Bernhard and Seitz, Stephan and Ploner, Stefan and Maier, Andreas K.},
   title = {Technical Note: PYRO-NN: Python reconstruction operators in neural networks},
   year = {2019},
   journal = {Medical Physics},
   }



Installation - Pip
============

pyronn_layers are automatically installed with pyronn via pip:

.. code-block:: bash

    pip install pyronn


The pyronn_layers itself can be installed via pip:

.. code-block:: bash

    pip install pyronn_layers

Installation - From Source
============

From pyronn_layers 0.1.0 onwards the docker imager provided by tensorflow can be used to built the reconstruction operators.
In this procedure, the operators are built so that they match the latest Tensorflow version that is distributed via pip.

To build the sources following tools are necessary: Docker.
Please prepare the system according to the Tensorflow repository: https://github.com/tensorflow/custom-op . 
 
If all necessary tools are installed the build process can start:

First, clone the Tensorflow custom-op repository:

.. code-block:: bash

    git clone https://github.com/tensorflow/custom-op <folder-name>
    cd <folder-name>

Now the reconstruction operators need to be added to the build process.
To achieve this, the PRYO-NN-Layers repository need to be cloned into a 'pyronn_layers' subfolder withing the directory:

.. code-block:: bash

    git clone https://github.com/csyben/PYRO-NN-Layers pyronn_layers

In the next step, the pyronn_layers need to be added to the build process (The TF examples can be removed at the same time).
Change the following files:

.. code-block:: bash

build_pip_pkg.sh -->
	remove zero_out & time_two
	add:   rsync -avm -L --exclude='*_test.py' ${PIP_FILE_PREFIX}pyronn_layers "${TMPDIR}"
	change python to python3 (or change the default python path in the docker image)

.. code-block:: bash

BUILD -->
	remove zero_out & time_two
	add: "//pyronn_layers:pyronn_layers_py",

.. code-block:: bash

setup.py -->set project name:
	project_name = 'pyronn-layers'

.. code-block:: bash

MANIFEST.in --> add pyronn 
	remove zero_out & add_two
	recursive-include pyronn_layers/ *.so

Now everything is setup to build the reconstruction operators.

The Tensorflow build process need to be configured, for that type:

.. code-block:: bash

    ./configure.sh
    bazel build build_pip_pkg
    bazel-bin/build_pip_pkg artifacts

Thats it. The wheel file containts the reconstruction operators. 
This wheel package can be now installed via pip:

.. code-block:: bash

    pip3 install ./artifacts/<FileName>

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

1. With the new pyronn version of 0.1.0 pyronn will automatically set memory growth for Tensorflow to true. The following code allows the memory growth:

.. code-block:: python

    gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RunetimeError as e:
                print(e)

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



