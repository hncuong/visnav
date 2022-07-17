#pragma once

#include <Eigen/Dense>
// #include <pangolin/image/managed_image.h>
#include <visnav/image/image.h>
#include <visnav/patterns.h>

namespace visnav {

template <typename Scalar, typename Pattern>
struct OpticalFlowPatch {
  static constexpr int PATTERN_SIZE = Pattern::PATTERN_SIZE;

  typedef Eigen::Matrix<int, 2, 1> Vector2i;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 1, 2> Vector2T;  // Transposed
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 1> VectorP;
  typedef Eigen::Matrix<Scalar, 2, PATTERN_SIZE> Matrix2P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 2> MatrixP2;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 3> MatrixP3;
  typedef Eigen::Matrix<Scalar, 3, PATTERN_SIZE> Matrix3P;
  typedef Eigen::Matrix<Scalar, PATTERN_SIZE, 4> MatrixP4;
  typedef Eigen::Matrix<int, 2, PATTERN_SIZE> Matrix2Pi;

  static const Matrix2P pattern2;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OpticalFlowPatch() { mean = 0; }

  OpticalFlowPatch(const visnav::Image<const uint8_t>& img,
                   const Vector2& pos) {
    setFromImage(img, pos);
  }

  void setFromImage(const visnav::Image<const uint8_t>& img,
                    const Vector2& pos) {
    this->pos = pos;

    int num_valid_points = 0;
    Scalar sum = 0;
    Vector2 grad_sum(0, 0);

    MatrixP2 grad;

    // Derived from the function setDataJacSe2 in BASALT
    for (int i = 0; i < PATTERN_SIZE; i++) {
      Vector2 p = pos + pattern2.col(i);
      if (img.InBounds(p.x(), p.y(), 3)) {
        Vector3 valGrad = img.interpGrad<Scalar>(p.x(), p.y());
        data[i] = valGrad[0];
        sum += valGrad[0];
        grad.row(i) = valGrad.template tail<2>();
        grad_sum += valGrad.template tail<2>();
        num_valid_points++;
      } else {
        data[i] = -1;
      }
    }

    mean = sum / num_valid_points;

    Scalar mean_inv = num_valid_points / sum;

    Eigen::Matrix<Scalar, 2, 3> Jw_se2;
    Jw_se2.template topLeftCorner<2, 2>().setIdentity();

    MatrixP3 J_se2;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (data[i] >= 0) {
        const Scalar data_i = data[i];
        const Vector2 grad_i = grad.row(i);
        grad.row(i) =
            num_valid_points * (grad_i * sum - grad_sum * data_i) / (sum * sum);

        data[i] *= mean_inv;
      } else {
        grad.row(i).setZero();
      }

      // Fill jacobians with respect to SE2 warp
      Jw_se2(0, 2) = -pattern2(1, i);
      Jw_se2(1, 2) = pattern2(0, i);
      J_se2.row(i) = grad.row(i) * Jw_se2;
    }

    // Derived from the function setFromImage in BASALT
    Matrix3 H_se2 = J_se2.transpose() * J_se2;
    Matrix3 H_se2_inv;
    H_se2_inv.setIdentity();
    H_se2.ldlt().solveInPlace(H_se2_inv);

    H_se2_inv_J_se2_T = H_se2_inv * J_se2.transpose();
  }

  inline bool residual(const visnav::Image<const uint8_t>& img,
                       const Matrix2P& transformed_pattern,
                       VectorP& residual) const {
    Scalar sum = 0;
    Vector2 grad_sum(0, 0);
    int num_valid_points = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      Vector2 p(transformed_pattern.col(i));

      if (img.InBounds(p.x(), p.y(), 3)) {
        residual[i] = img.interp<Scalar>(p.x(), p.y());
        sum += residual[i];
        num_valid_points++;
      } else {
        residual[i] = -1;
      }
    }

    int num_residuals = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (residual[i] >= 0 && data[i] >= 0) {
        const Scalar val = residual[i];
        residual[i] = (num_valid_points * val / sum) - data[i];
        num_residuals++;

      } else {
        residual[i] = 0;
      }
    }

    return num_residuals > PATTERN_SIZE / 2;
  }

  Vector2 pos = Vector2::Zero();
  VectorP data = VectorP::Zero();  // negative if the point is not valid

  // MatrixP3 J_se2;  // total jacobian with respect to se2 warp
  // Matrix3 H_se2_inv;
  Matrix3P H_se2_inv_J_se2_T = Matrix3P::Zero();

  Scalar mean = 0;

  bool valid = false;
};

template <typename Scalar, typename Pattern>
const typename OpticalFlowPatch<Scalar, Pattern>::Matrix2P
    OpticalFlowPatch<Scalar, Pattern>::pattern2 = Pattern::pattern2;

}  // namespace visnav
