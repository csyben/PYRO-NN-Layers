#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "helper_headers/helper_grid.h"
#include "helper_headers/helper_math.h"
#include "tensorflow/core/framework/types.pb.h"
texture<float, 3, cudaReadModeElementType> volume_as_texture;
#define CUDART_INF_F __int_as_float(0x7f800000)

#define BLOCKSIZE_X           16
#define BLOCKSIZE_Y           16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {        
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}


inline __device__ float kernel_project3D(const float* volume_ptr, const float3 source_point, const float3 ray_vector,
                                         const float step_size, const int3 volume_size,
                                         const float3 volume_origin, const float3 volume_spacing)
{
    float pixel = 0.0f;
    // Step 1: compute alpha value at entry and exit point of the volume
    float min_alpha, max_alpha;
    min_alpha = 0;
    max_alpha = CUDART_INF_F;
    


    if (0.0f != ray_vector.x)
    {
        float volume_min_edge_point = index_to_physical(0, volume_origin.x, volume_spacing.x);
        float volume_max_edge_point = index_to_physical(volume_size.x, volume_origin.x, volume_spacing.x);

        float reci = 1.0f / ray_vector.x;
        float alpha0 = (volume_min_edge_point - source_point.x) * reci;
        float alpha1 = (volume_max_edge_point - source_point.x) * reci;
        min_alpha = fmin(alpha0, alpha1);
        max_alpha = fmax(alpha0, alpha1);
    }

    if (0.0f != ray_vector.y)
    {
        float volume_min_edge_point = index_to_physical(0, volume_origin.y, volume_spacing.y);
        float volume_max_edge_point = index_to_physical(volume_size.y, volume_origin.y, volume_spacing.y);

        float reci = 1.0f / ray_vector.y;

        float alpha0 = (volume_min_edge_point - source_point.y) * reci;
        float alpha1 = (volume_max_edge_point - source_point.y) * reci;
        min_alpha = fmax(min_alpha, fmin(alpha0, alpha1));
        max_alpha = fmin(max_alpha, fmax(alpha0, alpha1));
    }

    if (0.0f != ray_vector.z)
    {
        float volume_min_edge_point = index_to_physical(0, volume_origin.z, volume_spacing.z);
        float volume_max_edge_point = index_to_physical(volume_size.z, volume_origin.z, volume_spacing.z);

        float reci = 1.0f / ray_vector.z;

        float alpha0 = (volume_min_edge_point - source_point.z) * reci;
        float alpha1 = (volume_max_edge_point - source_point.z) * reci;
        min_alpha = fmax(min_alpha, fmin(alpha0, alpha1));
        max_alpha = fmin(max_alpha, fmax(alpha0, alpha1));
    }
    float init_min = min_alpha;
    float init_max = max_alpha;
    // we start not at the exact entry point
    // => we can be sure to be inside the volume
    min_alpha += step_size * 0.5f;
    // Step 2: Cast ray if it intersects the volume
    // Trapezoidal rule (interpolating function = piecewise linear func)
    float3 point = make_float3(0,0,0);
    // Entrance boundary
    // For the initial interpolated value, only a half stepsize is
    //  considered in the computation.
    if (min_alpha < max_alpha)
    {
        point.x = source_point.x + min_alpha * ray_vector.x;
        point.y = source_point.y + min_alpha * ray_vector.y;
        point.z = source_point.z + min_alpha * ray_vector.z;
        float3 index = physical_to_index(point, volume_origin, volume_spacing);
        pixel += 0.5f * tex3D(volume_as_texture, index.x+0.5f , index.y+0.5f, index.z+0.5f );
        min_alpha += step_size;        
    }

    while (min_alpha < max_alpha)
    {
        point.x = source_point.x + min_alpha * ray_vector.x;
        point.y = source_point.y + min_alpha * ray_vector.y;
        point.z = source_point.z + min_alpha * ray_vector.z;
        float3 index = physical_to_index(point, volume_origin, volume_spacing);
        pixel += tex3D(volume_as_texture, index.x+0.5f , index.y+0.5f, index.z+0.5f);
        min_alpha += step_size;
    }    
    // Scaling by stepsize;
    pixel *= step_size;

    //Last segment of the line
    if (pixel > 0.0f)
    {   
        float3 index = physical_to_index(point, volume_origin, volume_spacing);
        pixel -= 0.5f * step_size * tex3D(volume_as_texture, index.x+0.5f, index.y+0.5f, index.z+0.5f);
        min_alpha -= step_size;
        float last_step_size = max_alpha - min_alpha;

        pixel += 0.5f * last_step_size* tex3D(volume_as_texture, index.x+0.5f, index.y+0.5f, index.z+0.5f);

        point.x = source_point.x + max_alpha * ray_vector.x;
        point.y = source_point.y + max_alpha * ray_vector.y;
        point.z = source_point.z + max_alpha * ray_vector.z;
        index = physical_to_index(point, volume_origin, volume_spacing);
        // The last segment of the line integral takes care of the
        // varying length.
        pixel += 0.5f * last_step_size * tex3D(volume_as_texture, index.x+0.5f, index.y+0.5f, index.z+0.5f);
    }
    int2 detector_idx = make_int2( blockIdx.x * blockDim.x + threadIdx.x,  blockIdx.y* blockDim.y + threadIdx.y  );
    uint projection_number = blockIdx.z;
    
    return pixel;
}
__global__ void project_3Dparallel_beam_kernel_tex_interp( const float* volume_ptr, float *pSinogram, 
                                            const float3 *d_ray_vectors, const float sampling_step_size,
                                            const int3 volume_size, const float3 volume_spacing, const float3 volume_origin,
                                            const int2 detector_size, const float2 detector_spacing, const float2 detector_origin,
                                            const int number_of_projections, const float ray_length)
{
    //return;
    int2 detector_idx = make_int2( blockIdx.x * blockDim.x + threadIdx.x,  blockIdx.y* blockDim.y + threadIdx.y  );
    uint projection_number = blockIdx.z;
    if (detector_idx.x >= detector_size.x || detector_idx.y >= detector_size.y || blockIdx.z >= number_of_projections)
    {
        return;
    }  
    float3 ray_vector = (d_ray_vectors[projection_number]);
    //TODO: should the initial vector be set from python level as the rotation axis to get consistent coordinate systems on the detector ?    
    float3 tmp = make_float3(0.0,0.0,1.0); 
    float3 u_vec = normalize(cross(tmp, ray_vector)); 
	float3 v_vec = normalize(cross(u_vec, ray_vector)); 

    float2 detector_coordinate = index_to_physical(make_float2(detector_idx), detector_origin, detector_spacing);
    
    float3 source_point = ray_vector * (-ray_length) + u_vec * detector_coordinate.x + v_vec * detector_coordinate.y;

    float pixel = kernel_project3D(
        volume_ptr,
        source_point,
        ray_vector,
        sampling_step_size,
        volume_size,
        volume_origin,
        volume_spacing);
   
    unsigned sinogram_idx = projection_number * detector_size.y * detector_size.x +  detector_idx.y * detector_size.x + detector_idx.x;

    pixel *= sqrt(  (ray_vector.x * volume_spacing.x) * (ray_vector.x * volume_spacing.x) +
                    (ray_vector.y * volume_spacing.y) * (ray_vector.y * volume_spacing.y) +
                    (ray_vector.z * volume_spacing.z) * (ray_vector.z * volume_spacing.z)  );

    pSinogram[sinogram_idx] = pixel;
    return;
}

void Parallel_Projection_Kernel_Tex_Interp_Launcher(const float* __restrict__ volume_ptr, float *out, const float *ray_vectors, const int number_of_projections,
                                         const int volume_width, const int volume_height, const int volume_depth, 
                                         const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                         const float volume_origin_x, const float volume_origin_y, const float volume_origin_z,
                                         const int detector_width, const int detector_height,
                                         const float detector_spacing_x,const float detector_spacing_y,
                                         const float detector_origin_x, const float detector_origin_y,
                                         const float step_size)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    volume_as_texture.addressMode[0] = cudaAddressModeBorder;
    volume_as_texture.addressMode[1] = cudaAddressModeBorder;
    volume_as_texture.addressMode[2] = cudaAddressModeBorder;
    volume_as_texture.filterMode = cudaFilterModeLinear;
    volume_as_texture.normalized = false;
    
    //COPY inv AR matrix to graphics card as float array
    auto ray_size_b = number_of_projections * sizeof(float3);
    float3 *d_ray_vectors;
    gpuErrchk(cudaMalloc(&d_ray_vectors, ray_size_b));
    gpuErrchk(cudaMemcpy(d_ray_vectors, ray_vectors, ray_size_b, cudaMemcpyHostToDevice));

     //COPY volume to graphics card
    //Malloc cuda array for texture
    cudaExtent volume_extent = make_cudaExtent(  volume_width, volume_height, volume_depth );
    cudaExtent volume_extent_byte = make_cudaExtent( sizeof(float)*volume_width, volume_height, volume_depth );

    cudaPitchedPtr d_volumeMem = make_cudaPitchedPtr( const_cast<float*>( volume_ptr ),
                                                volume_width*sizeof(float),
                                                volume_width,
                                                volume_height
                                            );
   
    cudaArray *volume_array;
    gpuErrchk(cudaMalloc3DArray(&volume_array, &channelDesc, volume_extent));
    
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = d_volumeMem;
    copyParams.dstArray = volume_array;
    copyParams.extent = volume_extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;

    gpuErrchk(cudaMemcpy3D(&copyParams)); 

    gpuErrchk(cudaBindTextureToArray(volume_as_texture, volume_array, channelDesc))

    int3 volume_size = make_int3(volume_width, volume_height, volume_depth);
    float3 volume_spacing = make_float3(volume_spacing_x, volume_spacing_y, volume_spacing_z);
    float3 volume_origin = make_float3(volume_origin_x, volume_origin_y, volume_origin_z);

    int2 detector_size = make_int2(detector_width, detector_height);
    float2 detector_spacing = make_float2(detector_spacing_x, detector_spacing_y);
    float2 detector_origin = make_float2(detector_origin_x, detector_origin_y);
    float ray_length = -1000;//ceil( sqrt(pow(volume_width*volume_spacing_x*0.5,2)+pow(volume_height*volume_spacing_y*0.5,2)+pow(volume_depth*volume_spacing_z*0.5,2)) * 1.1 );

    const dim3 blocksize = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, 1 );
    const dim3 gridsize = dim3( detector_size.x / blocksize.x + 1, detector_size.y / blocksize.y + 1 , number_of_projections+1);

    project_3Dparallel_beam_kernel_tex_interp<<<gridsize, blocksize>>>(volume_ptr, out, d_ray_vectors, step_size,
                                        volume_size, volume_spacing, volume_origin,
                                        detector_size, detector_spacing, detector_origin, 
                                        number_of_projections,ray_length);

    cudaDeviceSynchronize();

    // check for errors
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaFreeArray(volume_array));
    gpuErrchk(cudaUnbindTexture(volume_as_texture));
    gpuErrchk(cudaFree(d_ray_vectors));
}

#endif