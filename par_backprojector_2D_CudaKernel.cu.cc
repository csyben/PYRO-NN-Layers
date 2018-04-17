#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//Make float2 inline helper Methods
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
//Vector helper methods
// dot product
inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
//Grid Calculations
inline __host__ __device__ float index_to_physical(float index, float origin, float spacing)
{
    return index * spacing + origin;
}

inline __host__ __device__ float physical_to_index(float physical, float origin, float spacing)
{
    return (physical - origin) / spacing;
}

inline __host__ __device__ float2 index_to_physical(float2 index, float2 origin, float2 spacing)
{
    return make_float2(index.x * spacing.x + origin.x, index.y * spacing.y + origin.y);
}

inline __host__ __device__ float2 physical_to_index(float2 physical, float2 origin, float2 spacing)
{
    return make_float2((physical.x - origin.x) / spacing.x, (physical.y - origin.y) / spacing.y);
}

texture<float, cudaTextureType2D, cudaReadModeElementType> sinogram_as_texture;
#define CUDART_INF_F __int_as_float(0x7f800000)

__global__ void backproject_par_beam_kernel(float *pVolume, const float2 *d_rays, const int number_of_projections, const float sampling_step_size,
                                            const int2 volume_size, const float2 volume_spacing, const float2 volume_origin,
                                            const int detector_size, const float detector_spacing, const float detector_origin)
{
    unsigned int volume_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int volume_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (volume_x >= volume_size.x || volume_y >= volume_size.y)
    {
        return;
    }
    //Preparations:
    const float2 pixel_coordinate = index_to_physical(make_float2(volume_x, volume_y), volume_origin, volume_spacing);
    /* const float2 pixel_coordinate = make_flaot2(index_to_physical(volume_x, volume_origin.x, volume_spacing.x),
                                                 index_to_physical(volume_y, volume_origin.y, volume_spacing.y));
    */
    float pixel_value = 0.0f;

    for (int n = 0; n < number_of_projections; n++)
    {
        float2 detector_normal = d_rays[n];
        float2 detector_vec = make_float2(-detector_normal.y, detector_normal.x);

        float s = dot(pixel_coordinate, detector_vec);
        unsigned int s_idx = physical_to_index(s, detector_origin, detector_spacing);

        pixel_value += tex2D(sinogram_as_texture, s_idx + 0.5f, n + 0.5f);
        if (volume_x == 511 && volume_y == 511)
        {
            printf("volume_x=%d, volume_y=%d, physic.x=%f, physic.y=%f, s=%f, n=%d, pixel_value=%f\n", volume_x, volume_y, pixel_coordinate.x,pixel_coordinate.y,s, n, pixel_value);
        }
    }

    const unsigned volume_linearized_idx = volume_y * volume_size.x + volume_x;
    pVolume[volume_linearized_idx] = pixel_value / number_of_projections;

    return;
}

void Backprojection_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *ray_vectors, const int number_of_projections,
                                    const int volume_width, const int volume_height, const float volume_spacing_x, const float volume_spacing_y,
                                    const float volume_origin_x, const float volume_origin_y,
                                    const int detector_size, const float detector_spacing, const float detector_origin)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    sinogram_as_texture.addressMode[0] = cudaAddressModeBorder;
    sinogram_as_texture.addressMode[1] = cudaAddressModeBorder;
    sinogram_as_texture.filterMode = cudaFilterModeLinear;
    sinogram_as_texture.normalized = false;

    cudaArray *sinogram_array;
    cudaMallocArray(&sinogram_array, &channelDesc, detector_size, number_of_projections);
    cudaMemcpyToArray(sinogram_array, 0, 0, sinogram_ptr, detector_size * number_of_projections * sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTextureToArray(sinogram_as_texture, sinogram_array, channelDesc);
    printf("Number of Projections __host__: %d\n",number_of_projections);
    auto ray_size_b = number_of_projections * sizeof(float2);
    float2 *d_rays;
    cudaMalloc(&d_rays, ray_size_b);
    cudaMemcpy(d_rays, ray_vectors, ray_size_b, cudaMemcpyHostToDevice);

    float sampling_step_size = 1;

    int2 volume_size = make_int2(volume_width, volume_height);
    float2 volume_spacing = make_float2(volume_spacing_x, volume_spacing_y);
    float2 volume_origin = make_float2(volume_origin_x, volume_origin_y);

    const unsigned block_size = 16;
    const dim3 threads_per_block = dim3(block_size,block_size);
    const dim3 num_blocks = dim3(volume_width/threads_per_block.x+1, volume_height/threads_per_block.y+1);
    
    backproject_par_beam_kernel<<<num_blocks, threads_per_block>>>(out, d_rays, number_of_projections, sampling_step_size,
                                                         volume_size, volume_spacing, volume_origin,
                                                         detector_size, detector_spacing, detector_origin);
}

#endif
