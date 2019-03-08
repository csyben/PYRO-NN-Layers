
#ifndef HELPER_GEOMETRY_CPU_H
#define HELPER_GEOMETRY_CPU_H
#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace Geometry{
    
    /// Compute right null-space of A
    Eigen::VectorXf nullspace(const Eigen::MatrixXf& A)
    {
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto V=svd.matrixV();
        return V.col(V.cols()-1);
    }
    /// Extract the world coordinates of the camera center from a projection matrix. (SVD based implementation)
    Eigen::Vector4f getCameraCenter(const Eigen::MatrixXf& P)
    {
        Eigen::Vector4f C = Geometry::nullspace(P);
        if (C(3)<-1e-12 || C(3)>1e-12)
            C=C/C(3); // Def:Camera centers are always positive.
        return C;
    }
}
#endif

/*
 * Helper methods to prepare projection matrices for projector kernel
 * PyRo-ML is developed as an Open Source project under the GNU General Public License (GPL).
*/