#pragma once

#include <Eigen/Dense>
// #include <pangolin/image/managed_image.h>
#include <visnav/image/image.h>
#include <visnav/patterns.h>

namespace visnav {

template <typename Scalar, typename Pattern>
struct OpticalFlowPatch {
  static constexpr int PATTERN_SIZE = Pattern::PATTERN_SIZE;
  // static constexpr int PATTERN_SIZE =
  //     36;  // When HALF_PATCH_SIZE = 3. Otherwise needs calculation.
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

  // const int HALF_PATCH_SIZE = 3;

  static const Matrix2P pattern2;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OpticalFlowPatch() { mean = 0; }

  OpticalFlowPatch(const visnav::Image<const uint8_t>& img,
                   const Vector2& pos) {
    // std::cout << "BEFORE:\n" << std::endl;
    setFromImage(img, pos);
    // std::cout << "AFTER:\n" << std::endl;
  }

  void setFromImage(const visnav::Image<const uint8_t>& img,
                    const Vector2& pos) {
    // std::cout << "0 IS THIS ABORT?" << std::endl;
    this->pos = pos;

    int num_valid_points = 0;
    Scalar sum = 0;
    Vector2 grad_sum(0, 0);

    MatrixP2 grad;

    // int ind = 0;
    // for (int x = -HALF_PATCH_SIZE; x < HALF_PATCH_SIZE + 1; x++) {
    //   const int y_bound = sqrt(HALF_PATCH_SIZE * HALF_PATCH_SIZE - x * x);
    //   for (int y = -y_bound; y < y_bound + 1; y++) {
    //     pattern2.col(ind) = Vector2(x, y);
    //     ind++;
    //   }
    //   ind++;
    // }
    // std::cout << "1 IS THIS ABORT?" << std::endl;
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
    // std::cout << "2 IS THIS ABORT?" << std::endl;
    mean = sum / num_valid_points;

    Scalar mean_inv = num_valid_points / sum;

    Eigen::Matrix<Scalar, 2, 3> Jw_se2;
    Jw_se2.template topLeftCorner<2, 2>().setIdentity();

    MatrixP3 J_se2;

    // std::cout << "3 IS THIS ABORT?" << std::endl;
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
    // std::cout << "4 IS THIS ABORT?" << std::endl;
    Matrix3 H_se2 = J_se2.transpose() * J_se2;
    Matrix3 H_se2_inv;
    H_se2_inv.setIdentity();
    H_se2.ldlt().solveInPlace(H_se2_inv);

    H_se2_inv_J_se2_T = H_se2_inv * J_se2.transpose();
    // std::cout << "5 IS THIS ABORT?" << std::endl;
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

  // Eigen::Matrix<Scalar, 3, 1> interpGrad(const visnav::Image<uint8_t>& img,
  //                                        Scalar x, Scalar y) const {
  //   // static_assert(std::is_floating_point_v<Scalar>,
  //   //               "interpolation / gradient only makes sense "
  //   //               "for floating point result type");

  //   int ix = x;
  //   int iy = y;

  //   Scalar dx = x - ix;
  //   Scalar dy = y - iy;

  //   Scalar ddx = Scalar(1.0) - dx;
  //   Scalar ddy = Scalar(1.0) - dy;

  //   Eigen::Matrix<Scalar, 3, 1> res;

  //   const Scalar& px0y0 = img(ix, iy);
  //   const Scalar& px1y0 = img(ix + 1, iy);
  //   const Scalar& px0y1 = img(ix, iy + 1);
  //   const Scalar& px1y1 = img(ix + 1, iy + 1);

  //   res[0] = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 +
  //            dx * dy * px1y1;

  //   const Scalar& pxm1y0 = img(ix - 1, iy);
  //   const Scalar& pxm1y1 = img(ix - 1, iy + 1);

  //   Scalar res_mx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0
  //   +
  //                   dx * dy * px0y1;

  //   const Scalar& px2y0 = img(ix + 2, iy);
  //   const Scalar& px2y1 = img(ix + 2, iy + 1);

  //   Scalar res_px = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 +
  //                   dx * dy * px2y1;

  //   res[1] = Scalar(0.5) * (res_px - res_mx);

  //   const Scalar& px0ym1 = img(ix, iy - 1);
  //   const Scalar& px1ym1 = img(ix + 1, iy - 1);

  //   Scalar res_my = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1
  //   +
  //                   dx * dy * px1y0;

  //   const Scalar& px0y2 = img(ix, iy + 2);
  //   const Scalar& px1y2 = img(ix + 1, iy + 2);

  //   Scalar res_py = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 +
  //                   dx * dy * px1y2;

  //   res[2] = Scalar(0.5) * (res_py - res_my);

  //   return res;
  // }

  // Scalar interp(const visnav::Image<uint8_t>& img, Scalar x, Scalar y) const
  // {
  //   // static_assert(std::is_floating_point_v<S>,
  //   //               "interpolation / gradient only makes sense "
  //   //               "for floating point result type");

  //   int ix = x;
  //   int iy = y;

  //   Scalar dx = x - ix;
  //   Scalar dy = y - iy;

  //   Scalar ddx = Scalar(1.0) - dx;
  //   Scalar ddy = Scalar(1.0) - dy;

  //   return ddx * ddy * img(ix, iy) + ddx * dy * img(ix, iy + 1) +
  //          dx * ddy * img(ix + 1, iy) + dx * dy * img(ix + 1, iy + 1);
  // }

  Vector2 pos;
  VectorP data;  // negative if the point is not valid

  // MatrixP3 J_se2;  // total jacobian with respect to se2 warp
  // Matrix3 H_se2_inv;
  Matrix3P H_se2_inv_J_se2_T;

  Scalar mean;
};

template <typename Scalar, typename Pattern>
const typename OpticalFlowPatch<Scalar, Pattern>::Matrix2P
    OpticalFlowPatch<Scalar, Pattern>::pattern2 = Pattern::pattern2;

}  // namespace visnav
