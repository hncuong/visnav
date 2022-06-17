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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <pangolin/image/managed_image.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

namespace visnav {
void optical_flows(const pangolin::ManagedImage<uint8_t>& img_last,
                   const pangolin::ManagedImage<uint8_t>& img_current,
                   KeypointsData& kd_last, KeypointsData& current_frame_kd,
                   MatchData md) {}

void find_matches_landmarks_with_otpical_flow(
    const pangolin::ManagedImage<uint8_t>& img_last,
    const pangolin::ManagedImage<uint8_t>& img_current,
    const KeypointsData& kd_last, KeypointsData& current_frame_kd,
    const Landmarks& landmarks, const Corners& feature_corners,
    double reproject_threshold, LandmarkMatchData& md) {}

//    landmarks => KeypointsData kd
//        => optical_flows => current_frame_kd, MatchData
//    optical_flows()

// Remove the keypoints that die out in the

void localize_camera_optical_flow(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const KeypointsData& kdl, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  opengv::bearingVectors_t bearing_vectors;
  opengv::points_t points_t;
  size_t numberPoints = md.matches.size();
  // Reserve the size of bearing vectors and points = # matches
  bearing_vectors.reserve(numberPoints);
  points_t.reserve(numberPoints);

  // Affine2f:  Represents an homogeneous transformation in a 2 dimensional
  // space

  /*
   * - Adding unprojected points from the camera into bearing_vectors
   * - Adding landmarks at specific tracks into points
   * - pass bearing_vectors and points into CentralAbsoluteAdapter
   */

  for (auto& kv : md.matches) {
    FeatureId feature_id = kv.first;
    const auto& trackId = kv.second;
    points_t.push_back(landmarks.at(trackId).p);

    Eigen::Vector3d unprojected_point =
        cam->unproject(kdl.corners.at(feature_id));
    bearing_vectors.push_back(unprojected_point);
  }

  if (points_t.size() == 0 || bearing_vectors.size() == 0) return;

  /*
   * Use CentralAbsoluteAdapter & corresponding RANSAC implementation
   * AbsolutePoseSacProblem
   * - AbsolutePoseSacProblem that uses a minimal variant of PnP taking exactly
   * 3 points: KNEIP
   */
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors,
                                                        points_t);

  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  // create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)

  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));

  ransac.sac_model_ = absposeproblem_ptr;

  /*
   * Specifying reprojection error threshold
   * - Focal length of 500.0
   * - Reprojection error in pixels -->
   * reprojection_error_pnp_inlier_threshold_pixel
   */

  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  if (ransac.computeModel()) {
    // Set rotation block in Adapter
    adapter.setR(ransac.model_coefficients_.block<3, 3>(0, 0));
    // Set translation block in Adapter
    adapter.sett(ransac.model_coefficients_.block<3, 1>(0, 3));
    // Return the result from RANSAC
    opengv::transformation_t optimized =
        opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);
    ransac.sac_model_->selectWithinDistance(optimized, ransac.threshold_,
                                            ransac.inliers_);

    // Return the new transformation matrix in world coordinates (refined pose)
    Eigen::Matrix4d res;
    res.block<3, 3>(0, 0) = optimized.block<3, 3>(0, 0);
    res.block<3, 1>(0, 3) = optimized.block<3, 1>(0, 3);
    res.block<1, 4>(3, 0) = Eigen::Vector4d(0, 0, 0, 1);
    md.T_w_c = Sophus::SE3d(res);

    // Return set of track ids for all inliers
    for (auto i : ransac.inliers_) {
      md.inliers.push_back(md.matches[i]);
    }
  }
}

void add_new_landmarks_optical_flow(
    const FrameCamId fcidl, const FrameCamId fcidr, const KeypointsData& kdl,
    const KeypointsData& kdr, const Calibration& calib_cam,
    const MatchData& md_stereo, const LandmarkMatchData& md,
    Landmarks& landmarks, TrackId& next_landmark_id) {
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);
  // Pose of camera 1 (right) w.r.t camera 0 (left)
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  /*
   * Add new landmarks and observations
   */

  std::set<std::pair<FeatureId, FeatureId>> existing_landmark;
  for (auto& landmark_inlier : md.inliers) {
    FeatureId feature_id = landmark_inlier.first;
    TrackId track_id = landmark_inlier.second;
    // Add new landmark into landmarks here
    std::pair<FrameCamId, FeatureId> landmark_pair =
        std::make_pair(fcidl, feature_id);

    landmarks.at(track_id).obs.insert(landmark_pair);

    // Check if the left point is in md_stereo.inliers then add both
    // observations
    for (auto& md_stereo_inlier : md_stereo.inliers) {
      FeatureId stereo_feature_id = md_stereo_inlier.first;
      TrackId stereo_track_id = md_stereo_inlier.second;
      if (feature_id == stereo_feature_id) {
        existing_landmark.insert(md_stereo_inlier);
        landmark_pair = std::make_pair(fcidr, stereo_track_id);
        landmarks.at(track_id).obs.insert(landmark_pair);
      }
    }
  }

  /*
   * For all inlier stereo observations that were not added to the existing
   * landmarks
   * - Triangulate
   * - Add new landmarks
   *
   */

  opengv::bearingVectors_t bearing_vectors_0, bearing_vectors_1;

  for (auto& md_stereo_inlier : md_stereo.inliers) {
    // TODO: Adding new points into bearing_vectors
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearing_vectors_0, bearing_vectors_1, t_0_1, R_0_1);
  int idx = 0;

  for (auto stereo_inlier : md_stereo.inliers) {
    // Triangulate observations that were not added to existing landmarks
    if (existing_landmark.find(stereo_inlier) == existing_landmark.end()) {
      Landmark landmark;
      opengv::point_t point = opengv::triangulation::triangulate(adapter, idx);
      // md.T_w_c: camera pose estimated from landmarks in world frame
      point = md.T_w_c * point;

      landmark.p = point;
      std::pair<FrameCamId, FeatureId> left_cam_inlier =
          std::make_pair(fcidl, stereo_inlier.first);
      std::pair<FrameCamId, FeatureId> right_cam_inlier =
          std::make_pair(fcidr, stereo_inlier.second);

      landmark.obs.insert(left_cam_inlier);
      landmark.obs.insert(right_cam_inlier);

      /* Here next_landmark_id is a running index of the landmarks, so after
       * adding a new landmark you should always increase next_landmark_id by 1.
       */
      landmarks.insert(std::make_pair(next_landmark_id, landmark));
      next_landmark_id++;
    }
    idx++;
  }
}

void remove_old_keyframes_optical_flow(const FrameCamId fcidl,
                                       const int max_num_kfs, Cameras& cameras,
                                       Landmarks& landmarks,
                                       Landmarks& old_landmarks,
                                       std::set<FrameId>& kf_frames) {}

}  // namespace visnav
