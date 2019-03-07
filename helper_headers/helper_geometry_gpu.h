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
#pragma once
#ifndef HELPER_GEOMETRY_GPU_H
#define HELPER_GEOMETRY_GPU_H

#include "helper_math.h"
#include <Eigen/Dense>

inline __device__ float2 intersectLines2D(float2 p1, float2 p2, float2 p3, float2 p4)
{
    float dNom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);

    if (dNom < 0.000001f && dNom > -0.000001f)
    {
        float2 retValue = {NAN, NAN};
        return retValue;
    }
    float x = (p1.x * p2.y - p1.y * p2.x) * (p3.x - p4.x) - (p1.x - p2.x) * (p3.x * p4.y - p3.y * p4.x);
    float y = (p1.x * p2.y - p1.y * p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x * p4.y - p3.y * p4.x);

    x /= dNom;
    y /= dNom;
    float2 isectPt = {x, y};
    return isectPt;
}

#endif


