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


/*
 * Computes line intersection for projector kernel
 * Implementation is adapted from CONRAD
 * PYRO-NN is developed as an Open Source project under the GNU General Public License (GPL).
*/