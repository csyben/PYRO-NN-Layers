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
 * Voxel-driven parllel-beam back-projector CUDA kernel
 * Implementation partially adapted from CONRAD
 * PYRO-NN is developed as an Open Source project under the Apache License, Version 2.0.
*/
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "helper_headers/helper_grid.h"
#include "helper_headers/helper_math.h"

#define BLOCKSIZE_X           16
#define BLOCKSIZE_Y           4
#define BLOCKSIZE_Z           4

texture<float, cudaTextureType2DLayered> sinogram_as_texture;

#define CUDART_INF_F __int_as_float(0x7f800000)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void backproject_3Dpar_beam_kernel(float *pVolume, const float3 *d_rays, const float3 rotation_axis, const int number_of_projections,
                                              const int3 volume_size, const float3 volume_spacing, const float3 volume_origin,
                                              const int2 detector_size, const float2 detector_spacing, const float2 detector_origin)
{
    const float pi = 3.14159265359f;

    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    
    if( i >= volume_size.x  || j >= volume_size.y || k >= volume_size.z )
        return;
    
    const float3 voxel_coordinate = index_to_physical(make_float3(i,j,k),volume_origin,volume_spacing); 

    float pixel_value = 0.0f;
    
    for( int n = 0; n < number_of_projections; ++n )
    {
        float3 detector_normal = d_rays[n];
        float3 detector_vector_u = cross(rotation_axis,detector_normal);
        float3 detector_vector_v = cross(detector_vector_u, detector_normal);

        float2 detector_coordinate = make_float2(dot(voxel_coordinate,detector_vector_u),dot(voxel_coordinate,detector_vector_v)); 

        float2 detector_indices = physical_to_index(detector_coordinate, detector_origin, detector_spacing );
        
        pixel_value += tex2DLayered( sinogram_as_texture, detector_indices.x + 0.5, detector_indices.y + 0.5, n );
    }

    // linear volume address
    const unsigned int l = volume_size.x * ( k*volume_size.y + j ) + i;
    pVolume[l] = 2 * pi * pixel_value / number_of_projections;

}
/*************** WARNING ******************./
* 
*   Tensorflow is allocating the whole GPU memory for itself and just leave a small slack memory
*   using cudaMalloc and cudaMalloc3D will allocate memory in this small slack memory !
*   Therefore, currently only small volumes can be used (they have to fit into the slack memory which TF does not allocae !)
* 
*   This is the kernel based on texture interpolation, thus, the allocations are not within the Tensorflow managed memory.
*   If memory errors occure:
*    1. start Tensorflow with less gpu memory and allow growth
*    2. TODO: no software interpolation based 2D verions are available yet
* 
*   TODO: use context->allocate_tmp and context->allocate_persistent instead of cudaMalloc for the ray_vectors array
*       : https://stackoverflow.com/questions/48580580/tensorflow-new-op-cuda-kernel-memory-managment
* 
*/
void Parallel_Backprojection3D_Kernel_Launcher(const float *sinogram_ptr, float *out, const float *ray_vectors, const int number_of_projections,
                                               const int volume_width, const int volume_height, const int volume_depth,
                                               const float volume_spacing_x, const float volume_spacing_y, const float volume_spacing_z,
                                               const float volume_origin_x, const float volume_origin_y, const float volume_origin_z,
                                               const int detector_size_x, const int detector_size_y,
                                               const float detector_spacing_x, const float detector_spacing_y,
                                               const float detector_origin_x, const float detector_origin_y)
{
    auto ray_size_b = number_of_projections * sizeof(float3);
    float3 *d_rays;
    gpuErrchk( cudaMalloc(&d_rays, ray_size_b) );
    gpuErrchk( cudaMemcpy(d_rays, ray_vectors, ray_size_b, cudaMemcpyHostToDevice) );

    int3 volume_size = make_int3(volume_width, volume_height, volume_depth);
    float3 volume_spacing = make_float3(volume_spacing_x, volume_spacing_y, volume_spacing_z);
    float3 volume_origin = make_float3(volume_origin_x, volume_origin_y, volume_origin_z);

    int2 detector_size = make_int2(detector_size_x, detector_size_y);
    float2 detector_spacing = make_float2(detector_spacing_x, detector_spacing_y);
    float2 detector_origin = make_float2(detector_origin_x, detector_origin_y);

    //COPY volume to graphics card
    // set texture properties
    sinogram_as_texture.addressMode[0] = cudaAddressModeBorder;
    sinogram_as_texture.addressMode[1] = cudaAddressModeBorder;
    sinogram_as_texture.addressMode[2] = cudaAddressModeBorder;
    sinogram_as_texture.filterMode = cudaFilterModeLinear;
    sinogram_as_texture.normalized = false;

    // malloc cuda array for texture
    cudaExtent projExtent = make_cudaExtent( detector_size.x,
                                             detector_size.y,
                                             number_of_projections );
    
    cudaArray *projArray;
    
    static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    gpuErrchk( cudaMalloc3DArray( &projArray, &channelDesc, projExtent, cudaArrayLayered ) );

    auto pitch_ptr = make_cudaPitchedPtr( const_cast<float*>( sinogram_ptr ),
                                                detector_size.x*sizeof(float),
                                                detector_size.x,
                                                detector_size.y
                                            );
    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = pitch_ptr;
    copyParams.dstArray = projArray;
    copyParams.extent   = projExtent;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    gpuErrchk( cudaMemcpy3D( &copyParams ) );

    // bind texture reference
    gpuErrchk( cudaBindTextureToArray( sinogram_as_texture, projArray, channelDesc ) );

    // launch kernel
    const unsigned int gridsize_x = (volume_size.x-1) / BLOCKSIZE_X + 1;
    const unsigned int gridsize_y = (volume_size.y-1) / BLOCKSIZE_Y + 1;
    const unsigned int gridsize_z = (volume_size.z-1) / BLOCKSIZE_Z + 1;
    const dim3 grid = dim3( gridsize_x, gridsize_y, gridsize_z );
    const dim3 block = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z );

    //TODO:
    // Define rotation axis on python level as geometry 
    float3 rotation_axis = make_float3(0,0,1);

    backproject_3Dpar_beam_kernel<<< grid, block >>>(out, d_rays, rotation_axis, number_of_projections,
                                                                     volume_size, volume_spacing, volume_origin,
                                                                     detector_size, detector_spacing, detector_origin);

    cudaUnbindTexture(sinogram_as_texture);
    cudaFreeArray(projArray);
    cudaFree(d_rays);
}

#endif