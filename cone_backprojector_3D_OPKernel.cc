/*
 * Copyright [2019] [Christopher Syben]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Links the cone-beam projector layer from python to the actual kernel implementation. Implemented according to Tensorflow API.
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <typeinfo>
using namespace tensorflow; // NOLINT(build/namespaces)
using shape_inference::ShapeHandle; 

#define CUDA_OPERATOR_KERNEL "ConeBackprojection3D"

REGISTER_OP(CUDA_OPERATOR_KERNEL)
    .Input("sinogram: float")
    .Attr("sinogram_shape: shape")
    .Attr("volume_shape: shape")
    .Attr("volume_origin : tensor")
    .Attr("volume_spacing : tensor")
    .Attr("projection_multiplier : float")
    .Attr("projection_matrices : tensor")
    .Attr("hardware_interp : bool = false")
    .Attr("step_size : float")
    .Output("output: float")
    .SetShapeFn( []( ::tensorflow::shape_inference::InferenceContext* c )
    {
      TensorShapeProto sp;
      ShapeHandle sh;
      ShapeHandle batch;
      ShapeHandle out;
      auto status = c->GetAttr( "volume_shape", &sp );
      status.Update( c->MakeShapeFromShapeProto( sp, &sh ) );
      c->Subshape(c->input(0),0,1,&batch);
      c->Concatenate(batch, sh, &out);
      c->set_output( 0, out );
      return status;
    } )
    .Doc(R"doc(
Computes the 3D cone backprojection of the input sinogram on the given the trajectory

output: A Tensor.
  output = A_cone^T * p
)doc");

void Cone_Backprojection3D_Kernel_Tex_Interp_Launcher(const float *sinogram_ptr, float *out, const float *projection_matrix, const int number_of_projections,
                                    const int volume_width, const int volume_height, const int volume_depth, 
                                    const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                    const float volume_origin_x, const float volume_origin_y, const float volume_origin_z,
                                    const int detector_width, const int detector_height, const float projection_multiplier);

void Cone_Backprojection3D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *projection_matrix, const int number_of_projections,
                                    const int volume_width, const int volume_height, const int volume_depth, 
                                    const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                    const float volume_origin_x, const float volume_origin_y, const float volume_origin_z,
                                    const int detector_width, const int detector_height, const float projection_multiplier);

class ConeBackprojection3DOp : public OpKernel
{
    TensorShape volume_shape;
    int volume_width, volume_height, volume_depth;

    TensorShape projection_shape;
    int detector_size_x, detector_size_y, number_of_projections;

    float volume_origin_x, volume_origin_y, volume_origin_z;

    float volume_spacing_x, volume_spacing_y, volume_spacing_z;

    float projection_multiplier;

    Eigen::Tensor<float, 3, Eigen::RowMajor> projection_matrices;

    bool hardware_interp;
    float step_size;

  public:
    explicit ConeBackprojection3DOp(OpKernelConstruction *context) : OpKernel(context)
    {        
        //get detector shape from attributes
        OP_REQUIRES_OK(context, context->GetAttr("sinogram_shape", &projection_shape));
        number_of_projections = projection_shape.dim_size(0);
        detector_size_y = projection_shape.dim_size(1);
        detector_size_x = projection_shape.dim_size(2);
        //get volume shape from attributes
        OP_REQUIRES_OK(context, context->GetAttr("volume_shape", &volume_shape));
        volume_depth = volume_shape.dim_size(0);
        volume_height = volume_shape.dim_size(1);
        volume_width = volume_shape.dim_size(2);
        //get volume origin from attributes
        Tensor volume_origin_tensor;
        OP_REQUIRES_OK(context, context->GetAttr("volume_origin", &volume_origin_tensor));
        auto volume_origin_eigen = volume_origin_tensor.tensor<float, 1>();
        volume_origin_z = volume_origin_eigen(0);
        volume_origin_y = volume_origin_eigen(1);
        volume_origin_x = volume_origin_eigen(2);

        //get volume spacing from attributes
        Tensor volume_spacing_tensor;
        OP_REQUIRES_OK(context, context->GetAttr("volume_spacing", &volume_spacing_tensor));
        auto volume_spacing_eigen = volume_spacing_tensor.tensor<float, 1>();
        volume_spacing_z = volume_spacing_eigen(0);
        volume_spacing_y = volume_spacing_eigen(1);
        volume_spacing_x = volume_spacing_eigen(2);

        //get ray vectors from attributes
        Tensor projection_matrices_tensor;
        OP_REQUIRES_OK(context, context->GetAttr("projection_matrices", &projection_matrices_tensor));
        auto projection_matrices_eigen = projection_matrices_tensor.tensor<float, 3>();
        projection_matrices = Eigen::Tensor<float, 3, Eigen::RowMajor>( number_of_projections, 3, 4 );

        //get hardware interpolation flag
        OP_REQUIRES_OK(context, context->GetAttr("hardware_interp", &hardware_interp));

        //get discretization invariant and constant part of distance
        OP_REQUIRES_OK(context, context->GetAttr("projection_multiplier", &projection_multiplier));

        //set_size for projection operator as gradient
        OP_REQUIRES_OK(context, context->GetAttr("step_size", &step_size));

        //TODO: Loop is not neseccary anymore, refactor to direct conversion
        //for each projection
        for (int n = 0; n < number_of_projections; n++)
        {   
            // loop over matrix
            for( int i = 0; i < 3; ++i )
            {
                for( int j = 0; j < 4; ++j )
                {
                    projection_matrices(n,i,j) = projection_matrices_eigen(n,i,j);
                }
            }
        }
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<float>();        
        // Create an output tensor
        Tensor *output_tensor = nullptr;
        
        TensorShape out_shape = TensorShape(
            {input_tensor.shape().dim_size(0), volume_shape.dim_size(0), volume_shape.dim_size(1), volume_shape.dim_size(2)});

        // Check Batch size. Batch > 1 is not supported currently.
        OP_REQUIRES(context, input_tensor.shape().dim_size(0)  == 1,
                errors::InvalidArgument("Batch dimension is mandatory ! Batch size > 1 is not supported in the current PYRO-NN-layers."));

        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                         &output_tensor));
        auto output = output_tensor->template flat<float>();

        // Call the cuda kernel launcherst
        if(hardware_interp)
        {
            Cone_Backprojection3D_Kernel_Tex_Interp_Launcher(input.data(), output.data(), projection_matrices.data(), number_of_projections,
                                        volume_width, volume_height, volume_depth, volume_spacing_x, volume_spacing_y, volume_spacing_z,
                                        volume_origin_x, volume_origin_y, volume_origin_z,
                                        detector_size_x, detector_size_y,
                                        projection_multiplier);
        }
        else
        {
            Cone_Backprojection3D_Kernel_Launcher(input.data(), output.data(), projection_matrices.data(), number_of_projections,
                                        volume_width, volume_height, volume_depth, volume_spacing_x, volume_spacing_y, volume_spacing_z,
                                        volume_origin_x, volume_origin_y, volume_origin_z,
                                        detector_size_x, detector_size_y,
                                        projection_multiplier);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name(CUDA_OPERATOR_KERNEL).Device(DEVICE_GPU), ConeBackprojection3DOp);