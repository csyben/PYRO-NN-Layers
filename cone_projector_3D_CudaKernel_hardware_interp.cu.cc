#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "helper_headers/helper_grid.h"
#include "helper_headers/helper_math.h"
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

__device__ float kernel_project3D_tex_interp(const float3 source_point, const float3 ray_vector, const float step_size, const uint3 volume_size)
{   
    float pixel = 0.0f;
    // Step 1: compute alpha value at entry and exit point of the volume
    float min_alpha, max_alpha;
    min_alpha = 0;
    max_alpha = CUDART_INF_F;

    if (0.0f != ray_vector.x)
    {
        float volume_min_edge_point = 0 - 0.5f;
        float volume_max_edge_point = volume_size.x - 0.5f;


        float reci = 1.0f / ray_vector.x;
        float alpha0 = (volume_min_edge_point - source_point.x) * reci;
        float alpha1 = (volume_max_edge_point - source_point.x) * reci;
        min_alpha = fmin(alpha0, alpha1);
        max_alpha = fmax(alpha0, alpha1);
    }

    if (0.0f != ray_vector.y)
    {
        float volume_min_edge_point = 0 - 0.5f;
        float volume_max_edge_point = volume_size.y - 0.5f;

        float reci = 1.0f / ray_vector.y;
        float alpha0 = (volume_min_edge_point - source_point.y) * reci;
        float alpha1 = (volume_max_edge_point - source_point.y) * reci;
        min_alpha = fmax(min_alpha, fmin(alpha0, alpha1));
        max_alpha = fmin(max_alpha, fmax(alpha0, alpha1));
    }

    if (0.0f != ray_vector.z)
    {
        float volume_min_edge_point = 0 - 0.5f;
        float volume_max_edge_point = volume_size.z - 0.5f;

        float reci = 1.0f / ray_vector.z;
        float alpha0 = (volume_min_edge_point - source_point.z) * reci;
        float alpha1 = (volume_max_edge_point - source_point.z) * reci;
        min_alpha = fmax(min_alpha, fmin(alpha0, alpha1));
        max_alpha = fmin(max_alpha, fmax(alpha0, alpha1));
    }
    // we start not at the exact entry point
    // => we can be sure to be inside the volume
    min_alpha += step_size * 0.5f;

    // Step 2: Cast ray if it intersects the volume
    // Trapezoidal rule (interpolating function = piecewise linear func)

    float px, py, pz;
    
    // Entrance boundary
    // For the initial interpolated value, only a half stepsize is
    //  considered in the computation.
    if (min_alpha < max_alpha)
    {
        px = source_point.x + min_alpha * ray_vector.x;
        py = source_point.y + min_alpha * ray_vector.y;
        pz = source_point.z + min_alpha * ray_vector.z;

        pixel += 0.5f * tex3D(volume_as_texture, px , py , pz );

        min_alpha += step_size;
    }
    // Mid segments
    while (min_alpha < max_alpha)
    {
        px = source_point.x + min_alpha * ray_vector.x;
        py = source_point.y + min_alpha * ray_vector.y;
        pz = source_point.z + min_alpha * ray_vector.z;
        //float3 interp_point = physical_to_index(make_float3(px, py, pz), volume_origin, volume_spacing);

        pixel += tex3D(volume_as_texture, px, py, pz );

        min_alpha += step_size;
    }
    // Scaling by stepsize;
    pixel *= step_size;

    // Last segment of the line
    if (pixel > 0.0f)
    {   
        pixel -= 0.5f * step_size * tex3D(volume_as_texture,px ,py ,pz );
        
        min_alpha -= step_size;
        float last_step_size = max_alpha - min_alpha;

        pixel += 0.5f * last_step_size* tex3D(volume_as_texture, px, py, pz );

        px = source_point.x + max_alpha * ray_vector.x;
        py = source_point.y + max_alpha * ray_vector.y;
        pz = source_point.z + max_alpha * ray_vector.z;
        
        // The last segment of the line integral takes care of the
        // varying length.
        pixel += 0.5f * last_step_size * tex3D(volume_as_texture, px, py, pz);
    }
    return pixel;
}
//number_of_projections for boundary check neccessary ??
__global__ void project_3Dcone_beam_kernel_tex_interp(float *pSinogram, const float *d_inv_AR_matrices, const float3 *d_src_points, const float sampling_step_size,
                                          const uint3 volume_size, const float3 volume_spacing,
                                          const uint2 detector_size, const int number_of_projections)
{
    //return;
    uint2 detector_idx = make_uint2( blockIdx.x * blockDim.x + threadIdx.x,  blockIdx.y* blockDim.y + threadIdx.y  );
    uint projection_number = blockIdx.z;
    if (detector_idx.x >= detector_size.x || detector_idx.y >= detector_size.y || blockIdx.z >= number_of_projections)
    {
        return;
    }
    // //Preparations:
	d_inv_AR_matrices += projection_number * 9;
    float3 source_point = d_src_points[projection_number];
    
    //Compute ray direction
    const float rx = d_inv_AR_matrices[2] + detector_idx.y * d_inv_AR_matrices[1] + detector_idx.x * d_inv_AR_matrices[0];
    const float ry = d_inv_AR_matrices[5] + detector_idx.y * d_inv_AR_matrices[4] + detector_idx.x * d_inv_AR_matrices[3];
    const float rz = d_inv_AR_matrices[8] + detector_idx.y * d_inv_AR_matrices[7] + detector_idx.x * d_inv_AR_matrices[6];

    float3 ray_vector = make_float3(rx,ry,rz);
    ray_vector = normalize(ray_vector);

    float pixel = kernel_project3D_tex_interp(
        source_point,
        ray_vector,
        sampling_step_size,// * fmin(volume_spacing.x, volume_spacing.y),
        volume_size);

    //TODO: Check sacling
    pixel *= sqrt((ray_vector.x * volume_spacing.x) * (ray_vector.x * volume_spacing.x) +
            (ray_vector.y * volume_spacing.y) * (ray_vector.y * volume_spacing.y) +
            (ray_vector.z * volume_spacing.z) * (ray_vector.z * volume_spacing.z));

    unsigned sinogram_idx = projection_number * detector_size.y * detector_size.x +  detector_idx.y * detector_size.x + detector_idx.x;
    
    pSinogram[sinogram_idx] = pixel;
    return;
}

void Cone_Projection_Kernel_Tex_Interp_Launcher(const float* __restrict__ volume_ptr, float *out, const float *inv_AR_matrix, const float *src_points, 
                                    const int number_of_projections, const int volume_width, const int volume_height, const int volume_depth, 
                                    const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                    const int detector_width, const int detector_height,const float step_size)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    volume_as_texture.addressMode[0] = cudaAddressModeBorder;
    volume_as_texture.addressMode[1] = cudaAddressModeBorder;
    volume_as_texture.addressMode[2] = cudaAddressModeBorder;
    volume_as_texture.filterMode = cudaFilterModeLinear;
    volume_as_texture.normalized = false;

    //COPY inv AR matrix to graphics card as float array
    auto matrices_size_b = number_of_projections * 9 * sizeof(float);
    float *d_inv_AR_matrices;
    gpuErrchk(cudaMalloc(&d_inv_AR_matrices, matrices_size_b));
    gpuErrchk(cudaMemcpy(d_inv_AR_matrices, inv_AR_matrix, matrices_size_b, cudaMemcpyHostToDevice));
    //COPY source points to graphics card as float3
    auto src_points_size_b = number_of_projections * sizeof(float3);
    float3 *d_src_points;
    gpuErrchk(cudaMalloc(&d_src_points, src_points_size_b));
    gpuErrchk(cudaMemcpy(d_src_points, src_points, src_points_size_b, cudaMemcpyHostToDevice));

    //COPY volume to graphics card
    //Malloc cuda array for texture
    cudaExtent volume_extent = make_cudaExtent(  volume_width, volume_height, volume_depth );
    cudaExtent volume_extent_byte = make_cudaExtent( sizeof(float)*volume_width, volume_height, volume_depth );
    //TODO: pitched ptr from float *volumePtr does work, but the cudaArray below is still allocated using CUDA,
    //which means still the texture volume lives outside of the tensorflow allocated memory.

    cudaPitchedPtr d_volumeMem = make_cudaPitchedPtr( const_cast<float*>( volume_ptr ),
                                                volume_width*sizeof(float),
                                                volume_width,
                                                volume_height
                                            );
    //gpuErrchk(cudaMalloc3D(&d_volumeMem, volume_extent_byte));
    //size_t size = d_volumeMem.pitch * volume_height * volume_depth;
    //cudaMemcpy(d_volumeMem.ptr, volume_ptr, size, cudaMemcpyHostToDevice);
    
    /*************** WARNING ******************./
    * 
    *   Tensorflow is allocating the whole GPU memory for itself and just leave a small slack memory
    *   using cudaMalloc and cudaMalloc3D will allocate memory in this small slack memory !
    *   Therefore, currently only small volumes can be used (they have to fit into the slack memory which TF does not allocae !)
    *   The solution is to use the Tensorflow context to allocate memory within the gpu memory occupied by Tensorflow
    * 
    *   TODO: use context->allocate_tmp and context->allocate_persistent instead of cudaMalloc !
    *       : https://stackoverflow.com/questions/48580580/tensorflow-new-op-cuda-kernel-memory-managment
    * 
    */

   
    cudaArray *volume_array;
    gpuErrchk(cudaMalloc3DArray(&volume_array, &channelDesc, volume_extent));
    
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = d_volumeMem;
    copyParams.dstArray = volume_array;
    copyParams.extent = volume_extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;

    gpuErrchk(cudaMemcpy3D(&copyParams)); 

    gpuErrchk(cudaBindTextureToArray(volume_as_texture, volume_array, channelDesc))

    uint3 volume_size = make_uint3(volume_width, volume_height, volume_depth);
    float3 volume_spacing = make_float3(volume_spacing_x, volume_spacing_y, volume_spacing_z);


    uint2 detector_size = make_uint2(detector_width, detector_height);
    
    const dim3 blocksize = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, 1 );
    const dim3 gridsize = dim3( detector_size.x / blocksize.x + 1, detector_size.y / blocksize.y + 1 , number_of_projections+1);

    project_3Dcone_beam_kernel_tex_interp<<<gridsize, blocksize>>>(out, d_inv_AR_matrices, d_src_points, step_size,
                                        volume_size,volume_spacing, detector_size,number_of_projections);

    cudaDeviceSynchronize();


    // check for errors
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaFreeArray(volume_array));
    gpuErrchk(cudaUnbindTexture(volume_as_texture));

    //gpuErrchk(cudaFree(d_volumeMem.ptr)); //no need to deconstruct pitched pointer ????

    gpuErrchk(cudaFree(d_inv_AR_matrices));
    gpuErrchk(cudaFree(d_src_points));
}

#endif