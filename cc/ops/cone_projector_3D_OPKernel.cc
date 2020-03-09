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
 * Implementation partially adapted from CONRAD
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "helper_headers/helper_geometry_cpu.h"
#include "helper_headers/helper_eigen.h"
#include <Eigen/QR>
#include <typeinfo>
using namespace tensorflow; // NOLINT(build/namespaces)
using shape_inference::ShapeHandle; 

#define CUDA_OPERATOR_KERNEL "ConeProjection3D"
REGISTER_OP(CUDA_OPERATOR_KERNEL)
    .Input("volume: float")
    .Attr("volume_shape: shape")
    .Attr("projection_shape: shape")
    .Attr("volume_origin : tensor")
    .Attr("volume_spacing : tensor")
    .Attr("projection_matrices : tensor")
    .Attr("hardware_interp : bool = false")
    .Attr("step_size: float = 1.0")
    .Attr("projection_multiplier : float")
    .Output("output: float")
    .SetShapeFn( []( ::tensorflow::shape_inference::InferenceContext* c )
    {
      TensorShapeProto sp;
      ShapeHandle sh;
      ShapeHandle batch;
      ShapeHandle out;
      auto status = c->GetAttr( "projection_shape", &sp );
      status.Update( c->MakeShapeFromShapeProto( sp, &sh ) );
      c->Subshape(c->input(0),0,1,&batch);
      c->Concatenate(batch, sh, &out);
      c->set_output( 0, out );
      return status;
    } )
    .Doc(R"doc(
Computes the 3D cone forward projection of the input based on the given the trajectory

output: A Tensor.
  output = A_cone * x
)doc");

void Cone_Projection_Kernel_Launcher(const float *volume_ptr, float *out, const float *inv_AR_matrix, const float *src_points, const int number_of_projections,
                                    const int volume_width, const int volume_height, const int volume_depth, 
                                    const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                    const int detector_width, const int detector_height, const float step_size, tensorflow::OpKernelContext *context);

void Cone_Projection_Kernel_Tex_Interp_Launcher(const float *volume_ptr, float *out, const float *inv_AR_matrix, const float *src_points, const int number_of_projections,
                                                const int volume_width, const int volume_height, const int volume_depth, 
                                                const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                                const int detector_width, const int detector_height, const float step_size);

class ConeProjection3DOp : public OpKernel
{
    TensorShape volume_shape;
    int volume_width, volume_height, volume_depth;

    TensorShape projection_shape;
    int detector_size_x, detector_size_y, number_of_projections;

    float volume_origin_x, volume_origin_y, volume_origin_z;

    float volume_spacing_x, volume_spacing_y, volume_spacing_z;

    float step_size;
    bool hardware_interp;

    float projection_multiplier;

    Eigen::Tensor<float, 3, Eigen::RowMajor> projection_matrices;
    //TensorShape projection_matrices_shape;

    Eigen::Tensor<float, 3, Eigen::RowMajor> inv_AR_matrix;
    Eigen::Tensor<float, 2, Eigen::RowMajor> src_points;

    

  public:

    
    explicit ConeProjection3DOp(OpKernelConstruction *context) : OpKernel(context)
    {
        //get volume shape from attributes
        OP_REQUIRES_OK(context, context->GetAttr("volume_shape", &volume_shape));
        volume_depth = volume_shape.dim_size(0);
        volume_height = volume_shape.dim_size(1);
        volume_width = volume_shape.dim_size(2);
        //get detector shape from attributes
        OP_REQUIRES_OK(context, context->GetAttr("projection_shape", &projection_shape));
        number_of_projections = projection_shape.dim_size(0);
        detector_size_y = projection_shape.dim_size(1);
        detector_size_x = projection_shape.dim_size(2);
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
        // projection_matrices_shape = projection_matrices_tensor.shape();
        //Init src_point and inv_ar_matrix tensors
        //get stepsize
        OP_REQUIRES_OK(context, context->GetAttr("step_size", &step_size));

        //get hardware interpolation flag
        OP_REQUIRES_OK(context, context->GetAttr("hardware_interp", &hardware_interp));

        //Projectionmultiplier for backprojection as gradient
        OP_REQUIRES_OK(context, context->GetAttr("projection_multiplier", &projection_multiplier));

        src_points = Eigen::Tensor<float, 2, Eigen::RowMajor>(number_of_projections,3);
        inv_AR_matrix = Eigen::Tensor<float, 3, Eigen::RowMajor>(number_of_projections,3,3);
        
        /*********************************************************************************************************************************************************************
         * 
         *  P = [M | -MC] 
         *  M = KR
         * 1. Extract Source Position (C) from P using SVD to calc right null space
         * 2. Calculate M^-1 and multiply with a 3x3 Matrix containing 1/voxel_spacing on the diagonal matrix
         * 3. Put src_points and inv_ar_matrix into the CUDA Kernel, like Cone-Projector from Conrad
         * 
         * WARNING: The following code is not created under memory and runtime performance point of view.
         *          A better conversion from Tensorflow Tensor to Eigen::Tensor and Eigen::Matrizes are probably neccessary !!!!
         * ********************************************************************************************************************************************************************/

        Eigen::Matrix3f scaling_matrix(3,3);
        scaling_matrix.setZero();
        scaling_matrix(0,0) = 1.0/volume_spacing_x;
        scaling_matrix(1,1) = 1.0/volume_spacing_y;
        scaling_matrix(2,2) = 1.0/volume_spacing_z;
        src_points.setZero();
        inv_AR_matrix.setZero();

        //for each projection
        for (int n = 0; n < number_of_projections; n++)
        {            
            Eigen::Matrix<float,3,4,Eigen::RowMajor> proj_mat(3,4);
            proj_mat << projection_matrices_eigen(n,0,0), projection_matrices_eigen(n,0,1), projection_matrices_eigen(n,0,2), projection_matrices_eigen(n,0,3),
                        projection_matrices_eigen(n,1,0), projection_matrices_eigen(n,1,1), projection_matrices_eigen(n,1,2), projection_matrices_eigen(n,1,3),
                        projection_matrices_eigen(n,2,0), projection_matrices_eigen(n,2,1) ,projection_matrices_eigen(n,2,2), projection_matrices_eigen(n,2,3); 

            auto c = (Geometry::getCameraCenter(proj_mat) * -1).eval();

            src_points(n,0) = -((volume_origin_x * scaling_matrix(0,0)) + c(0) * scaling_matrix(0,0));
            src_points(n,1) = -((volume_origin_y * scaling_matrix(1,1)) + c(1) * scaling_matrix(1,1));
            src_points(n,2) = -((volume_origin_z * scaling_matrix(2,2)) + c(2) * scaling_matrix(2,2));

            Eigen::Matrix<float,3,3, Eigen::RowMajor> inverted_scaled_result = (scaling_matrix * proj_mat.block<3,3>(0,0).inverse()).eval();

            //TODO: dont copy element-wise use Eigen::Map to map eigen::matrix to eigen::tensor
            for(int j = 0; j < inverted_scaled_result.cols();++j){
                for(int i = 0; i < inverted_scaled_result.rows(); ++i){
                    inv_AR_matrix(n,j,i) = inverted_scaled_result(j,i);
                }
            }           
        }
    }

    /*
    https://github.com/tensorflow/tensorflow/issues/5902
    tensorflow::GPUBFCAllocator* allocator = new tensorflow::GPUBFCAllocator(0, sizeof(float) * height * width * 3);
    tensorflow::Tensor input_tensor = tensorflow::Tensor(allocator, tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape( { 1, height, width, 3 }));
    <copy output data from program A into the GPU memory allocated by input_tensor using a GPU->GPU copy>


    https://stackoverflow.com/questions/39797095/tensorflow-custom-allocator-and-accessing-data-from-tensor

    */

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor        
        const Tensor &input_tensor = context->input(0);        
        auto input = input_tensor.flat<float>();        
        // Create an output tensor
        TensorShape out_shape = TensorShape(
          {input_tensor.shape().dim_size(0), projection_shape.dim_size(0), projection_shape.dim_size(1), projection_shape.dim_size(2)});
        Tensor *output_tensor = nullptr;

        // Check Batch size. Batch > 1 is not supported currently.
        OP_REQUIRES(context, input_tensor.shape().dim_size(0) == 1,
                errors::InvalidArgument("Batch dimension is mandatory ! Batch size > 1 is not supported in the current PYRO-NN-layers."));

        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                         &output_tensor));
        
        auto output = output_tensor->template flat<float>();

        if(hardware_interp){
            Cone_Projection_Kernel_Tex_Interp_Launcher(input.data(), output.data(), inv_AR_matrix.data(), src_points.data(), number_of_projections,
                                        volume_width, volume_height, volume_depth, volume_spacing_x, volume_spacing_y, volume_spacing_z,
                                        detector_size_x, detector_size_y, step_size);
        }
        else{
            //TODO:
            // allocate inv_ar_matrix, src_points with tensorflow context as temp memory.
            // Call the cuda kernel launcher
            Cone_Projection_Kernel_Launcher(input.data(), output.data(), inv_AR_matrix.data(), src_points.data(), number_of_projections,
                                        volume_width, volume_height, volume_depth, volume_spacing_x, volume_spacing_y, volume_spacing_z,
                                        detector_size_x, detector_size_y,step_size, context);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name(CUDA_OPERATOR_KERNEL).Device(DEVICE_GPU), ConeProjection3DOp);