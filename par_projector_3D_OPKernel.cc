#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <typeinfo>
using namespace tensorflow; // NOLINT(build/namespaces)

#define CUDA_OPERATOR_KERNEL "ParallelProjection3D"
REGISTER_OP(CUDA_OPERATOR_KERNEL)
    .Input("volume: float")
    .Attr("volume_shape: shape")
    .Attr("projection_shape: shape")
    .Attr("volume_origin : tensor")
    .Attr("volume_spacing : tensor")
    .Attr("detector_origin : tensor")
    .Attr("detector_spacing : tensor")
    .Attr("ray_vectors : tensor")
    .Attr("hardware_interp : bool = false")
    .Attr("step_size: float = 0.2")
    .Output("output: float")
    .Doc(R"doc(
Computes the 3D parallel forward projection of the input based on the given the trajectory

output: A Tensor.
  output = A_parallel * x
)doc");

// void Parallel_Projection_Kernel_Launcher(const float *volume_ptr, float *out, const float *ray_vector, const int number_of_projections,
//                                     const int volume_width, const int volume_height, const int volume_depth, 
//                                     const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
//                                     const int detector_width, const int detector_height,
//                                     const float detector_spacing_x,const float detector_spacing_y,
//                                     const float detector_origin_x, const float detector_origin_y,
//                                     const float step_size, tensorflow::OpKernelContext *context);

void Parallel_Projection_Kernel_Tex_Interp_Launcher(const float *volume_ptr, float *out, const float *ray_vector, const int number_of_projections,
                                                const int volume_width, const int volume_height, const int volume_depth, 
                                                const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                                const float volume_origin_x, const float volume_origin_y, const float volume_origin_z,
                                                const int detector_width, const int detector_height, 
                                                const float detector_spacing_x,const float detector_spacing_y,
                                                const float detector_origin_x, const float detector_origin_y,
                                                const float step_size);

class ParallelProjection3DOp : public OpKernel
{
    TensorShape volume_shape;
    int volume_width, volume_height, volume_depth;

    TensorShape projection_shape;
    int detector_size_x, detector_size_y, number_of_projections;

    float volume_origin_x, volume_origin_y, volume_origin_z;

    float volume_spacing_x, volume_spacing_y, volume_spacing_z;

    float detector_origin_x, detector_origin_y;

    float detector_spacing_x, detector_spacing_y;

    float step_size;
    bool hardware_interp;

    Eigen::Tensor<float, 2, Eigen::RowMajor> ray_vectors;

  public:

    
    explicit ParallelProjection3DOp(OpKernelConstruction *context) : OpKernel(context)
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

        //get detector origin from attributes
        Tensor detector_origin_tensor;
        OP_REQUIRES_OK(context, context->GetAttr("detector_origin", &detector_origin_tensor));
        auto detector_origin_eigen = detector_origin_tensor.tensor<float, 1>();
        detector_origin_y = detector_origin_eigen(0);
        detector_origin_x = detector_origin_eigen(1);

        //get volume spacing from attributes
        Tensor volume_spacing_tensor;
        OP_REQUIRES_OK(context, context->GetAttr("volume_spacing", &volume_spacing_tensor));
        auto volume_spacing_eigen = volume_spacing_tensor.tensor<float, 1>();
        volume_spacing_z = volume_spacing_eigen(0);
        volume_spacing_y = volume_spacing_eigen(1);
        volume_spacing_x = volume_spacing_eigen(2);

        //get volume spacing from attributes
        Tensor detector_spacing_tensor;
        OP_REQUIRES_OK(context, context->GetAttr("detector_spacing", &detector_spacing_tensor));
        auto detector_spacing_eigen = volume_spacing_tensor.tensor<float, 1>();
        detector_spacing_y = detector_spacing_eigen(0);
        detector_spacing_x = detector_spacing_eigen(1);

        //get ray vectors from attributes
        Tensor ray_vectors_tensor;
        OP_REQUIRES_OK(context, context->GetAttr("ray_vectors", &ray_vectors_tensor));
        auto ray_vectors_eigen = ray_vectors_tensor.tensor<float, 2>();
        ray_vectors = Eigen::Tensor<float, 2, Eigen::RowMajor>(ray_vectors_eigen);
        // projection_matrices_shape = projection_matrices_tensor.shape();
        //Init src_point and inv_ar_matrix tensors
        //get stepsize
        OP_REQUIRES_OK(context, context->GetAttr("step_size", &step_size));

        //get hardware interpolation flag
        OP_REQUIRES_OK(context, context->GetAttr("hardware_interp", &hardware_interp));
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
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, projection_shape,
                                                         &output_tensor));
        
        auto output = output_tensor->template flat<float>();
        //std::cout << "hardware_interp: " << output_tensor->shape() << std::endl;
        //if(hardware_interp){
        Parallel_Projection_Kernel_Tex_Interp_Launcher(input.data(), output.data(), ray_vectors.data(), number_of_projections,
                                    volume_width, volume_height, volume_depth, volume_spacing_x, volume_spacing_y, volume_spacing_z, volume_origin_x, volume_origin_y, volume_origin_z,
                                    detector_size_x, detector_size_y, detector_spacing_x, detector_spacing_y, detector_origin_x, detector_origin_y,
                                    step_size);
        //}
        //else{
            //TODO:
            // allocate inv_ar_matrix, src_points with tensorflow context as temp memory.

            // Tensor* inv_AR_matrix_tensor = nullptr;
            // std::cout << "befor temp alloc" << std::endl;
            // TensorShape matrix_shape = TensorShape({number_of_projections,4,3});
            // // // temparily use this space
            // OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,matrix_shape , inv_AR_matrix_tensor));
            // auto test = inv_AR_matrix_tensor->flat<float>().data();
            // std::cout << "tmp alloc finished, launch kernel" << std::endl;
        // Call the cuda kernel launcher
            // Parallel_Projection_Kernel_Launcher(input.data(), output.data(), ray_vectors.data(), number_of_projections,
            //                             volume_width, volume_height, volume_depth, volume_spacing_x, volume_spacing_y, volume_spacing_z,
            //                             detector_size_x, detector_size_y, detector_spacing_x, detector_spacing_y, detector_origin_x, detector_origin_y, 
            //                             step_size, context);
        //}
    }
};

REGISTER_KERNEL_BUILDER(Name(CUDA_OPERATOR_KERNEL).Device(DEVICE_GPU), ParallelProjection3DOp);