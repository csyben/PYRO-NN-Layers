/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#ifndef HELPER_GRID_H
#define HELPER_GRID_H


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

inline __host__ __device__ float3 index_to_physical(float3 index, float3 origin, float3 spacing)
{
    return make_float3(index.x * spacing.x + origin.x, index.y * spacing.y + origin.y, index.z * spacing.z + origin.z);
}

inline __host__ __device__ float3 physical_to_index(float3 physical, float3 origin, float3 spacing)
{
    return make_float3((physical.x - origin.x) / spacing.x, (physical.y - origin.y) / spacing.y, (physical.z - origin.z) / spacing.z);
}

#endif


