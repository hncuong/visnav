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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/types.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 3: compute essential matrix
  // w_hat
  // T_0_1 = [R|t] is the transform from cam1 3d coord to cam0 3d coord
  // Then E = R transpose * t_cross
  // And x1 = RT (x0 - t) and x0 = R * x1 + t
  Eigen::Vector3d t_0_1_normalized = t_0_1 / t_0_1.norm();
  Eigen::Matrix3d t_cross;
  t_cross << 0., -t_0_1_normalized(2, 0), t_0_1_normalized(1, 0),
      t_0_1_normalized(2, 0), 0., -t_0_1_normalized(0, 0),
      -t_0_1_normalized(1, 0), t_0_1_normalized(0, 0), 0.;
  E = R_0_1.transpose() * t_cross;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers
    Eigen::Vector3d p0_3d = cam1->unproject(p0_2d);
    Eigen::Vector3d p1_3d = cam2->unproject(p1_2d);

    // Epipolar constraint.
    // x1T * E * x0 = 0
    auto c = p1_3d.transpose() * E * p0_3d;
    if (abs(c) < epipolar_error_threshold) md.inliers.push_back(md.matches[j]);
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();
  md.T_i_j = Sophus::SE3d();

  // TODO SHEET 3: Run RANSAC with using opengv's CentralRelativePose and store
  // the final inlier indices in md.inliers and the final relative pose in
  // md.T_i_j (normalize translation). If the number of inliers is smaller than
  // ransac_min_inliers, leave md.inliers empty. Note that if the initial RANSAC
  // was successful, you should do non-linear refinement of the model parameters
  // using all inliers, and then re-estimate the inlier set with the refined
  // model parameters.

  // create the central relative adapter
  // Create 3d vectors with unit norm as bearing vectors
  opengv::bearingVectors_t bearingVectors1;
  opengv::bearingVectors_t bearingVectors2;
  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers
    Eigen::Vector3d p0_3d = cam1->unproject(p0_2d);
    Eigen::Vector3d p1_3d = cam2->unproject(p1_2d);
    bearingVectors1.push_back(p0_3d);
    bearingVectors2.push_back(p1_3d);
  }

  // Create Ransac problem
  opengv::relative_pose::CentralRelativeAdapter adapter(bearingVectors1,
                                                        bearingVectors2);
  // Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  // Ransac problem
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::NISTER));

  // run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  //  ransac.max_iterations_ = 100;
  ransac.computeModel();
  // get the result
  opengv::transformation_t best_transformation =
      ransac.model_coefficients_;  // typedef Eigen::Matrix<double,3,4>

  // Refine the model use all inlier
  //  inliers = ransac.inliers_;
  std::vector<int> inliers = ransac.inliers_;
  // Create adaptor base on inliers
  opengv::bearingVectors_t bearingVectors1_inliers;
  opengv::bearingVectors_t bearingVectors2_inliers;
  for (auto i : inliers) {
    bearingVectors1_inliers.push_back(bearingVectors1[i]);
    bearingVectors2_inliers.push_back(bearingVectors2[i]);
  }

  // Recompute model coeffs using
  opengv::relative_pose::CentralRelativeAdapter adapter_inliers(
      bearingVectors1_inliers, bearingVectors2_inliers);
  adapter_inliers.sett12(best_transformation.block<3, 1>(0, 3));
  adapter_inliers.setR12(best_transformation.block<3, 3>(0, 0));
  opengv::transformation_t refined_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter_inliers);

  // Re-estimate inlier set
  ransac.sac_model_->selectWithinDistance(refined_transformation, ransac_thresh,
                                          inliers);

  // Check min_inliers and store inliers indices
  if (inliers.size() >= ransac_min_inliers) {
    for (auto i : inliers) {
      md.inliers.push_back(md.matches[i]);
    }
  }
  // Store final relative pose
  Eigen::Vector3d t_0_1 = refined_transformation.block<3, 1>(0, 3)
                              .normalized();  // Normalize translation
  Eigen::Matrix3d R_0_1 = refined_transformation.block<3, 3>(0, 0);
  md.T_i_j = Sophus::SE3d(R_0_1, t_0_1);
  // Check order
}
}  // namespace visnav
