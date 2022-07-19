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

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.
  for (const auto& kv : landmarks) {
    TrackId trackId = kv.first;
    const auto& p_3w = kv.second.p;

    // Project to camera
    auto p_3c = current_pose.inverse() * p_3w;
    // Only take landmark infront of camera
    if (p_3c.z() >= cam_z_threshold) {
      auto p_2c = cam->project(p_3c);
      // CHeck if it inside the image
      if (p_2c.x() >= 0. && p_2c.x() <= cam->width() - 1. && p_2c.y() >= 0. &&
          p_2c.y() <= cam->height() - 1.) {
        projected_points.emplace_back(p_2c);
        projected_track_ids.emplace_back(trackId);
      }
    }
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_threshold and feature_match_dist_2_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.

  // For each keypoints find landmark inside match_max_dist_2d
  for (size_t i = 0; i < kdl.corners.size(); i++) {
    const auto& kp_2c = kdl.corners.at(i);
    const auto& kp_descriptor = kdl.corner_descriptors.at(i);

    // Save best and second best distance
    size_t kp_smallest_dist = 256;
    size_t kp_second_best_dist = kp_smallest_dist;
    TrackId match_trackId = -1;

    // Find the landmarks with smallest distance
    for (size_t j = 0; j < projected_points.size(); j++) {
      const auto& lm_p2c = projected_points.at(j);

      // Check if landmark inside the radius
      auto l2_distance = (lm_p2c - kp_2c).norm();
      if (l2_distance > match_max_dist_2d) continue;

      // Find distance as smallest distance between observer
      const auto& lm_trackId = projected_track_ids.at(j);
      size_t kp2lm_dist = 256;
      const FeatureTrack& observers = landmarks.at(lm_trackId).obs;
      for (const auto& kv : observers) {
        const FrameCamId& fcid = kv.first;
        const FeatureId& featureId = kv.second;

        const auto& obs_descriptor =
            feature_corners.at(fcid).corner_descriptors.at(featureId);
        auto descriptor_dist = (obs_descriptor ^ kp_descriptor).count();
        if (descriptor_dist < kp2lm_dist) kp2lm_dist = descriptor_dist;
      }

      if (kp2lm_dist < kp_smallest_dist) {
        // Update two best distance
        kp_second_best_dist = kp_smallest_dist;
        kp_smallest_dist = kp2lm_dist;

        // Filter match threshold
        if (kp2lm_dist < feature_match_threshold) match_trackId = lm_trackId;

      } else if (kp2lm_dist < kp_second_best_dist)
        kp_second_best_dist = kp2lm_dist;
    }

    // Filter by feature match dist 2 best
    if (kp_second_best_dist >= feature_match_dist_2_best * kp_smallest_dist &&
        match_trackId >= 0) {
      // Add i (featureId) and match_trackId
      md.matches.emplace_back(i, match_trackId);
    }
  }
}

void localize_camera(const Sophus::SE3d& current_pose,
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

  // TODO SHEET 5: Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this exercise we don't explicitly have
  // tracks.

  // Create bearingVectors and points
  // bearingVectors is Camera Frame and points is World Frame
  size_t numberPoints = md.matches.size();
  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;
  bearingVectors.reserve(numberPoints);
  points.reserve(numberPoints);

  // Fill with correspondences in matches
  for (const auto& kv : md.matches) {
    // 3D point in world frame
    const auto& trackId = kv.second;
    points.emplace_back(landmarks.at(trackId).p);

    // 2D -> 3D point in cam frame
    const auto& feature_id = kv.first;
    const auto& p_2c = kdl.corners.at(feature_id);
    bearingVectors.emplace_back(cam->unproject(p_2c));
  }

  //
  // RANSAC
  //
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));

  ransac.sac_model_ = absposeproblem_ptr;
  const double default_focal_length = 500.;
  const double ransac_thresh =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel /
                     default_focal_length));
  ransac.threshold_ = ransac_thresh;

  //  ransac.max_iterations_ = maxIterations;
  ransac.computeModel();
  opengv::transformation_t best_transformation = ransac.model_coefficients_;

  // Get the inliers and refine pose to T_w_c
  std::vector<int> inliers = ransac.inliers_;
  opengv::bearingVectors_t bearingVectors_inliers;
  opengv::points_t points_inliers;
  bearingVectors.reserve(inliers.size());
  points.reserve(inliers.size());

  for (auto i : inliers) {
    bearingVectors_inliers.push_back(bearingVectors[i]);
    points_inliers.push_back(points[i]);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter_inliers(
      bearingVectors_inliers, points_inliers);
  adapter_inliers.sett(best_transformation.block<3, 1>(0, 3));
  adapter_inliers.setR(best_transformation.block<3, 3>(0, 0));
  opengv::transformation_t refined_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter_inliers);

  md.T_w_c = Sophus::SE3d(refined_transformation.block<3, 3>(0, 0),
                          refined_transformation.block<3, 1>(0, 3));

  // Refine inliers and add to inlier_track_ids
  // Re-estimate inlier set
  ransac.sac_model_->selectWithinDistance(refined_transformation, ransac_thresh,
                                          inliers);

  md.inliers.reserve(inliers.size());
  for (auto i : inliers) {
    md.inliers.emplace_back(md.matches.at(i));
  }
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to landmark
  // matches add the observations to the existing landmarks. If the left
  // camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to the
  // existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a new
  // landmark you should always increase next_landmark_id by 1.

  const auto& T_w_c = md.T_w_c;
  const auto& stereo_inliers = md_stereo.inliers;  // FeatureId, FeatureId
  const auto& landmark_inliers = md.inliers;       // FeatureId, TrackId
  std::unordered_map<FeatureId, FeatureId> stereo_inliers_map(
      stereo_inliers.begin(), stereo_inliers.end());
  const FeatureId LANDMARK_EXIST = -1;
  std::set<FeatureId> featureExistInflows;

  // For each landmark inliers add observer
  for (const auto& feature_n_Track : landmark_inliers) {
    const auto& featureId_left = feature_n_Track.first;
    const auto& trackId = feature_n_Track.second;

    // Add observation
    landmarks.at(trackId).obs.emplace(fcidl, featureId_left);
    if (stereo_inliers_map.count(featureId_left) > 0) {
      landmarks.at(trackId).obs.emplace(fcidr,
                                        stereo_inliers_map.at(featureId_left));
      // Set this in stereo_inliers_map to -1 to mark that it already exist in
      // landmark
      //      stereo_inliers_map.at(featureId_left) = LANDMARK_EXIST;
      featureExistInflows.emplace(featureId_left);
    }
  }

  // Add new landamrks by Triangulate for inlier stereo obs that were not in
  // landmark
  size_t numberPoints = stereo_inliers.size();
  opengv::bearingVectors_t bearingVectors0;
  opengv::bearingVectors_t bearingVectors1;
  bearingVectors0.reserve(numberPoints);
  bearingVectors1.reserve(numberPoints);

  for (const auto& featureId_lr : stereo_inliers) {
    bearingVectors0.emplace_back(
        calib_cam.intrinsics.at(fcidl.cam_id)
            ->unproject(kdl.corners.at(featureId_lr.first)));
    bearingVectors1.emplace_back(
        calib_cam.intrinsics.at(fcidr.cam_id)
            ->unproject(kdr.corners.at(featureId_lr.second)));
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearingVectors0, bearingVectors1, t_0_1, R_0_1);

  for (size_t i = 0; i < stereo_inliers.size(); i++) {
    const auto& featureId_lr = stereo_inliers.at(i);
    // If observer is not in landmark
    if (featureExistInflows.count(featureId_lr.first) == 0) {
      opengv::point_t p =
          T_w_c * opengv::triangulation::triangulate(adapter, i);

      Landmark lm;
      lm.p = p;
      lm.obs.emplace(fcidl, featureId_lr.first);
      lm.obs.emplace(fcidr, featureId_lr.second);
      lm.first_frame_obs = fcidl.frame_id;
      lm.last_frame_obs = fcidl.frame_id;

      // Add new landmark
      landmarks.emplace(next_landmark_id, lm);
      next_landmark_id++;
    }
  }
}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Cameras& old_cameras,
                          Landmarks& landmarks, Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // TODO SHEET 5: Remove old cameras and observations if the number of keyframe
  // pairs (left and right image is a pair) is larger than max_num_kfs. The ids
  // of all the keyframes that are currently in the optimization should be
  // stored in kf_frames. Removed keyframes should be removed from cameras and
  // landmarks with no left observations should be moved to old_landmarks.

  // Find old keyframes to be removed
  if (kf_frames.size() <= max_num_kfs) return;

  std::vector<TrackId> to_remove_landmarks;

  while (kf_frames.size() > max_num_kfs) {
    auto min_frameId = *kf_frames.begin();
    FrameCamId min_fcidl = FrameCamId(min_frameId, 0);
    FrameCamId min_fcidr = FrameCamId(min_frameId, 1);
    old_cameras[min_fcidl] = cameras.at(min_fcidl);

    // Remove from cameras
    if (cameras.count(min_fcidl) > 0) cameras.erase(min_fcidl);
    if (cameras.count(min_fcidr) > 0) cameras.erase(min_fcidr);

    // Remove the observation
    for (auto& track_lm : landmarks) {
      const auto& trackId = track_lm.first;
      auto& landmark = track_lm.second;

      if (landmark.obs.count(min_fcidl) > 0) landmark.obs.erase(min_fcidl);
      if (landmark.obs.count(min_fcidr) > 0) landmark.obs.erase(min_fcidr);

      // If no observer left
      if (landmark.obs.size() == 0) {
        old_landmarks.emplace(trackId, landmark);
        to_remove_landmarks.emplace_back(trackId);
      }
    }

    // Update kf_frames
    kf_frames.erase(min_frameId);
  }

  for (const auto& trackId : to_remove_landmarks) landmarks.erase(trackId);
}
}  // namespace visnav
