#include <unsupported/Eigen/CXX11/Tensor>

template<typename T>
using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic>;

template<typename Scalar,int rank, typename sizeType>
MatrixType<Scalar> Tensor_to_Matrix(const Eigen::Tensor<Scalar,rank> &tensor,const sizeType rows,const sizeType cols)
{
    return Eigen::Map<const MatrixType<Scalar>> (tensor.data(), rows,cols);
}

template<typename Scalar, typename... Dims>
Eigen::Tensor< Scalar, sizeof... (Dims)> Matrix_to_Tensor(const MatrixType<Scalar> &matrix, Dims... dims)
{
    constexpr int rank = sizeof... (Dims);
    return Eigen::TensorMap<Eigen::Tensor<const Scalar, rank>>(matrix.data(), {dims...});
}

/*
 * Conversion from: https://stackoverflow.com/questions/48795789/eigen-unsupported-tensor-to-eigen-matrix
 * PyRo-ML is developed as an Open Source project under the GNU General Public License (GPL).
*/