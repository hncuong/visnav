/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <sophus/se3.hpp>

#include <cmath>
#include <iostream>
#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // TODO SHEET 1: implement
  // Compute theta
  auto theta = xi.norm();

  // w_hat
  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << T(0.), -xi(2, 0), xi(1, 0), xi(2, 0), T(0.), -xi(0, 0), -xi(1, 0),
      xi(0, 0), T(0.);

  Eigen::Matrix<T, 3, 3> exp_w_hat;
  exp_w_hat << T(1.), T(0.), T(0.), T(0.), T(1.), T(0.), T(0.), T(0.), T(1.);

  if (theta > T(std::numeric_limits<double>::epsilon())) {
    exp_w_hat += sin(theta) / theta * w_hat +
                 (T(1.0) - cos(theta)) / (theta * theta) * w_hat * w_hat;
  } else {
    exp_w_hat += w_hat;
  }

  return exp_w_hat;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  // Init w
  Eigen::Matrix<T, 3, 1> w;
  w << mat(2, 1) - mat(1, 2), mat(0, 2) - mat(2, 0), mat(1, 0) - mat(0, 1);

  // Calculate theta
  auto theta = acos((mat(0, 0) + mat(1, 1) + mat(2, 2) - T(1.)) / T(2.));

  // If theta small, approximate sin(theta) ~ theta
  auto factor = T(0.5);
  if (theta > T(std::numeric_limits<double>::epsilon())) {
    factor = theta / (2 * sin(theta));
  }
  // Calculate w
  w *= factor;
  return w;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 4, 4> transform;
  transform.setZero();
  transform(3, 3) = T(1.);

  // ROtation part
  Eigen::Matrix<T, 3, 1> w;
  Eigen::Matrix<T, 3, 1> v;
  w << xi(3, 0), xi(4, 0), xi(5, 0);
  v << xi(0, 0), xi(1, 0), xi(2, 0);

  Eigen::Matrix<T, 3, 3> exp_w_hat;  //= user_implemented_expmap()
  exp_w_hat = user_implemented_expmap(w);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) transform(i, j) = exp_w_hat(i, j);

  // Translation part
  // w_hat
  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << T(0.), -w(2, 0), w(1, 0), w(2, 0), T(0.), -w(0, 0), -w(1, 0),
      w(0, 0), T(0.);

  Eigen::Matrix<T, 3, 3> J;

  auto theta = w.norm();
  auto w_hat_sq = w_hat * w_hat;
  auto theta_sq = theta * theta;

  if (theta > T(std::numeric_limits<double>::epsilon())) {
    J = Eigen::Matrix<T, 3, 3>::Identity() +
        (T(1.0) - cos(theta)) / (theta_sq)*w_hat +
        (theta - sin(theta)) / (theta_sq * theta) * w_hat_sq;
  } else {
    J = exp_w_hat;
  }

  Eigen::Matrix<T, 3, 1> jv;
  jv = J * v;
  transform(0, 3) = jv(0, 0);
  transform(1, 3) = jv(1, 0);
  transform(2, 3) = jv(2, 0);

  return transform;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 3> R;
  R = mat.template topLeftCorner<3, 3>();
  Eigen::Matrix<T, 3, 1> translation;
  translation = mat.template topRightCorner<3, 1>();

  // Rotation
  Eigen::Matrix<T, 3, 1> w = user_implemented_logmap(R);
  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << T(0.), -w(2, 0), w(1, 0), w(2, 0), T(0.), -w(0, 0), -w(1, 0),
      w(0, 0), T(0.);
  auto theta = w.norm();

  // Translation
  Eigen::Matrix<T, 3, 3> J_inv;
  // APproximate
  if (theta < T(std::numeric_limits<double>::epsilon())) {
    J_inv = Eigen::Matrix<T, 3, 3>::Identity() - w_hat / T(2.);
  } else {
    J_inv = Eigen::Matrix<T, 3, 3>::Identity() - w_hat / T(2.) +
            (T(1.) / (theta * theta) -
             (1 + cos(theta)) / (2 * theta * sin(theta))) *
                w_hat * w_hat;
  }
  Eigen::Matrix<T, 3, 1> v = J_inv * translation;

  Eigen::Matrix<T, 6, 1> ret;
  ret.template topLeftCorner<3, 1>() = v;
  ret.template bottomLeftCorner<3, 1>() = w;

  return ret;
}

}  // namespace visnav
