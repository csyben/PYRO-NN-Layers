#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow; // NOLINT(build/namespaces)

#define CUDA_OPERATOR_KERNEL "AddOne"

REGISTER_OP(CUDA_OPERATOR_KERNEL)
    .Input("input: int32")
    .Output("output: int32")
    .Doc(R"doc(
Adds 1 to all elements of the tensor.

output: A Tensor.
  output = input + 1
)doc");

void AddOneKernelLauncher(const int *in, const int N, int *out);

class AddOneOp : public OpKernel
{
  public:
    explicit AddOneOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<int32>();

        // Create an output tensor
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output = output_tensor->template flat<int32>();

        // Set all but the first element of the output tensor to 0.
        const int N = input.size();
        // Call the cuda kernel launcher
        AddOneKernelLauncher(input.data(), N, output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name(CUDA_OPERATOR_KERNEL).Device(DEVICE_GPU), AddOneOp);