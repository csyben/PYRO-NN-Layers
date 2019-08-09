## 0.0.4
* [x]   Added shape_inference::ShapeHandle; to allow Tensorflow to evaluate the shape of the propived operators
* [x]   Batch dimension is now handled correctly
* [x]   All Operators require the batch_size stored in the first dimension of the Tensor ! Batch dimensions is now mandatory
* [x]   Currently all operators are only supporting a batch_size of 1 ! A Batch_size >1 will lead to an Error.
* [x]   Preparation to Tensorflow 2.0 and Eager execution are included

## 0.0.3
* [x]   Final license: Apache 2.0

## 0.0.2
* [x]   Critical Bugfix: 3D Layers missing Attributes for gradient registration

## 0.0.1

* [x]  Initial package setup:
* [x]  2D parallel, fan and 3D cone beam projectors and back-projectors computing A and A^T on the fly
* [x]  Tensorflow patch for Tensorflow 1.9 up to Tensorflow 1.12
* [x]  BUILD file to include custom kernels into the Tensorflow .whl package
* [x]  Add license
