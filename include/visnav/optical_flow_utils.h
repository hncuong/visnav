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

#ifndef OPTICAL_FLOW_UTILS_H
#define OPTICAL_FLOW_UTILS_H

#include <set>
#include <vector>

#include <visnav/common_types.h>
#include <visnav/calibration.h>
#include <visnav/keypoints.h>
#include <visnav/map_utils.h>

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <pangolin/image/managed_image.h>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

namespace visnav {

// Use optical flow from a last frame to a new frame to establish
//  1. keypoints in new frame
//  2. match data between last frame and new frame
void optical_flows_opencv(const pangolin::ManagedImage<uint8_t>& img_src,
                          const pangolin::ManagedImage<uint8_t>& img_dest,
                          const KeypointsData& kd_src, KeypointsData& kd_dest,
                          MatchData& md, int pyramid_level,
                          double distance_threshold) {
  // If last frame have empty corner
  if (kd_src.corners.size() == 0) return;

  // Create two frame and convert to Gray
  cv::Mat old_frame(img_src.h, img_src.w, CV_8U, img_src.ptr);
  cv::Mat frame(img_dest.h, img_dest.w, CV_8U, img_dest.ptr);

  // Keypoints data
  std::vector<cv::Point2f> p0, p1, ptmp;
  p0.reserve(kd_src.corners.size());
  p1.reserve(kd_src.corners.size());
  for (size_t i = 0; i < kd_src.corners.size(); i++) {
    p0.emplace_back(kd_src.corners.at(i).x(), kd_src.corners.at(i).y());
  }

  // Optical Flows
  std::vector<uchar> status;
  std::vector<float> err;
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
  cv::Size windowSize(15, 15);
  //  int pyramid_level = 2;
  //  double distance_threshold = 1.0;

  std::vector<cv::Mat> old_frame_pyr, frame_pyr;

  cv::buildOpticalFlowPyramid(old_frame, old_frame_pyr, windowSize,
                              pyramid_level);
  cv::buildOpticalFlowPyramid(frame, frame_pyr, windowSize, pyramid_level);

  cv::calcOpticalFlowPyrLK(old_frame_pyr, frame_pyr, p0, p1, status, err,
                           windowSize, pyramid_level, criteria);

  cv::calcOpticalFlowPyrLK(frame_pyr, old_frame_pyr, p1, ptmp, status, err,
                           windowSize, pyramid_level, criteria);

  // TODO Inverse optical flows

  // Convert results to current_frame_kd and Match data
  int match_id = 0;
  //  for (size_t i = 0; i < p1.size(); i++) {
  //    if (status[i] == 1) {
  //      kd_dest.corners.emplace_back(p1[i].x, p1[i].y);
  //      md.matches.emplace_back(i, match_id);
  //      match_id++;
  //    }
  //  }

  for (int i = 0; i < ptmp.size(); i++) {
    if (status[i] == 1) {
      Eigen::Vector2d diff(std::abs((ptmp[i] - p0[i]).x),
                           std::abs((ptmp[i] - p0[i]).y));
      Eigen::Vector2d right_keypoint(p1[i].x, p1[i].y);

      kd_dest.corners.push_back(right_keypoint);
      if ((diff.allFinite()) && (diff.squaredNorm() < distance_threshold)) {
        md.matches.push_back(std::make_pair(i, match_id));
      }
      match_id++;
    }
  }
  std::cout << "Do Bi-directional optical flow" << std::endl;
  std::cout << "kd_dest corner size: " << kd_dest.corners.size()
            << " md matches size: " << md.matches.size() << std::endl;
}

/// Compute matches of current frame -> flows
/// Based on matches of current frame -> last frame
/// Remove flows that has no match in current frame
void find_matches_landmarks_with_otpical_flow(const FrameCamId& fcid_last,
                                              const MatchData& md_last,
                                              Flows& flows,
                                              LandmarkMatchData& md) {
  // If no match to last frame available
  if (md_last.matches.size() == 0) return;

  // For each flows, find featureId of last frame
  // Map to featureId in new frame with md_last
  // Map featureId last -> current frame
  std::map<FeatureId, FeatureId> matches_map;
  for (const auto& kv : md_last.matches) {
    matches_map.insert(std::make_pair(kv.first, kv.second));
  }

  std::vector<TrackId> flow_to_discard;
  for (auto& kv_lm : flows) {
    const TrackId& trackId = kv_lm.first;
    FeatureTrack& flow = kv_lm.second.flow;

    // Actually not expected to have flow of last = 0
    // Cause we gonna kill dieout flows later
    if (flow.count(fcid_last) > 0) {
      // trackId -> last feature Id
      const FeatureId& last_featureId = flow.at(fcid_last);

      // If there is a match: last featureId -> current featureId
      if (matches_map.count(last_featureId) > 0) {
        const FeatureId& current_featureId = matches_map.at(last_featureId);
        md.matches.emplace_back(current_featureId, trackId);

        // Add observer for the flow
        FrameCamId fcid_current =
            FrameCamId(fcid_last.frame_id + 1, fcid_last.cam_id);
        if (current_featureId < 0)
          std::cout << "Invalid: Fcurrent_featureId = " << current_featureId
                    << "\n";
        flow.emplace(fcid_current, current_featureId);
        // Increase length of flow
        kv_lm.second.length++;
      } else {
        // If there is not match
        // Discard the flows
        flow_to_discard.emplace_back(trackId);
      }
    }
  }

  // Now discard dieout flows
  for (const auto& trackId : flow_to_discard) {
    flows.at(trackId).alive = false;
  }
}

void localize_camera_optical_flow(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const KeypointsData& kdl, const Flows& flows,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

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
    points.emplace_back(flows.at(trackId).p);

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
  bearingVectors_inliers.reserve(inliers.size());
  points_inliers.reserve(inliers.size());

  // Fill with correspondences in matches inliers
  for (const auto& i : inliers) {
    const auto& kv = md.matches.at(i);

    bearingVectors_inliers.push_back(cam->unproject(kdl.corners.at(kv.first)));
    points_inliers.emplace_back(flows.at(kv.second).p);
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

/*
 * Add new flows and flows for Optical flows
 */
void add_new_landmarks_optical_flow(
    const FrameCamId fcidl, const FrameCamId fcidr, const KeypointsData& kdl,
    const KeypointsData& kdr, const Calibration& calib_cam,
    const MatchData& md_stereo, const LandmarkMatchData& md, Flows& flows,
    TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  const auto& T_w_c = md.T_w_c;
  const auto& stereo_inliers = md_stereo.inliers;  // FeatureId, FeatureId
  const auto& landmark_inliers = md.inliers;       // FeatureId, TrackId

  // Create inlier map for quick search
  std::unordered_map<FeatureId, FeatureId> stereo_inliers_map;
  for (const auto& kv : stereo_inliers) {
    stereo_inliers_map.emplace(kv.first, kv.second);
  }

  //  const FeatureId LANDMARK_EXIST = -2;
  std::set<FeatureId> featureExistInflows;

  // For each landmark inliers add observer
  for (const auto& feature_n_Track : landmark_inliers) {
    const auto& featureId_left = feature_n_Track.first;
    const auto& trackId = feature_n_Track.second;

    // Add observation
    flows.at(trackId).obs.emplace(fcidl, featureId_left);
    if (stereo_inliers_map.count(featureId_left) > 0) {
      flows.at(trackId).obs.emplace(fcidr,
                                    stereo_inliers_map.at(featureId_left));
      // Set this to mark that it already exist in
      // landmark
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
    // TODO Check if stereo_inliers not Filter -1 yet
    const auto& featureId_lr = stereo_inliers.at(i);
    // If observer is not in landmark
    if (featureExistInflows.count(featureId_lr.first) == 0) {
      opengv::point_t p =
          T_w_c * opengv::triangulation::triangulate(adapter, i);

      Flow flow;
      flow.p = p;
      flow.alive = true;
      flow.is_landmark = true;

      // Add observation
      flow.obs.emplace(fcidl, featureId_lr.first);
      flow.obs.emplace(fcidr, featureId_lr.second);

      // Add flow obs
      flow.flow.emplace(fcidl, featureId_lr.first);
      flow.length = 1;

      // Add new landmark
      // and flow
      flows.emplace(next_landmark_id, flow);

      next_landmark_id++;
    }
  }
}

// TODO Update this function
void remove_old_keyframes_optical_flow(const FrameCamId fcidl,
                                       const int max_num_kfs, Cameras& cameras,
                                       Cameras& old_cameras, Flows& flows,
                                       Flows& old_flows,
                                       std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // Find old keyframes to be removed
  if (kf_frames.size() <= max_num_kfs) return;

  std::vector<TrackId> to_remove_flows;

  while (kf_frames.size() > max_num_kfs) {
    auto min_frameId = *kf_frames.begin();
    FrameCamId min_fcidl = FrameCamId(min_frameId, 0);
    FrameCamId min_fcidr = FrameCamId(min_frameId, 1);
    old_cameras[min_fcidl] = cameras.at(min_fcidl);

    // Remove from cameras
    if (cameras.count(min_fcidl) > 0) cameras.erase(min_fcidl);
    if (cameras.count(min_fcidr) > 0) cameras.erase(min_fcidr);

    // Remove the observation
    for (auto& track_lm : flows) {
      const auto& trackId = track_lm.first;
      auto& landmark = track_lm.second;

      if (landmark.obs.count(min_fcidl) > 0) landmark.obs.erase(min_fcidl);
      if (landmark.obs.count(min_fcidr) > 0) landmark.obs.erase(min_fcidr);

      // If no observer left
      if (landmark.obs.size() == 0) {
        old_flows.emplace(trackId, landmark);
        to_remove_flows.emplace_back(trackId);
      }
    }

    // Update kf_frames
    kf_frames.erase(min_frameId);
  }

  for (const auto& trackId : to_remove_flows) flows.erase(trackId);
}

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment_for_flows(const Corners& feature_corners,
                                 const BundleAdjustmentOptions& options,
                                 const std::set<FrameCamId>& fixed_cameras,
                                 Calibration& calib_cam, Cameras& cameras,
                                 Flows& flows) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem
  // Add Params Block and Set Constant for Intrinsics
  // Then Add Param Block for all Camera poses + 3D flows
  // Set Camera in fixed_cameras to Constant
  problem.AddParameterBlock(calib_cam.intrinsics.at(0)->data(), 8);
  problem.AddParameterBlock(calib_cam.intrinsics.at(1)->data(), 8);
  if (!options.optimize_intrinsics) {
    problem.SetParameterBlockConstant(calib_cam.intrinsics.at(0)->data());
    problem.SetParameterBlockConstant(calib_cam.intrinsics.at(1)->data());
  }

  // For each Camera pose
  for (auto& kv : cameras) {
    problem.AddParameterBlock(kv.second.T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
  }
  for (const auto& fcid : fixed_cameras) {
    problem.SetParameterBlockConstant(cameras.at(fcid).T_w_c.data());
  }

  // For each 3D landmark
  for (auto& kv : flows) {
    problem.AddParameterBlock(kv.second.p.data(), 3);
  }

  // Add residual blocks
  // For each landmark: trackid ->
  // Look for it feature track: fcid ->
  for (auto& kv : flows) {
    auto& p_3d = kv.second.p;
    const auto& lm = kv.second.obs;

    for (const auto& fcid_featureId : lm) {
      // Get respected Pose of Cam and 2d point
      // And intrinsic
      const auto& fcid = fcid_featureId.first;
      const auto& featureId = fcid_featureId.second;

      auto& T_w_c = cameras.at(fcid).T_w_c;
      // FIXME featureId -1 case; fcid 140 1;
      const auto& p_2d = feature_corners.at(fcid).corners.at(featureId);
      auto& intrinsic = calib_cam.intrinsics.at(fcid.cam_id);
      const auto& cam_model = intrinsic->name();

      // TODO Check How to Get Cam mode
      BundleAdjustmentReprojectionCostFunctor* c =
          new BundleAdjustmentReprojectionCostFunctor(p_2d, cam_model);
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2,
          Sophus::SE3d::num_parameters, 3, 8>(c);
      // TODO Huber loss for regularize
      //      ceres::LossFunction* lost_function = nullptr;
      if (options.use_huber) {
        problem.AddResidualBlock(cost_function,
                                 new ceres::HuberLoss(options.huber_parameter),
                                 T_w_c.data(), p_3d.data(), intrinsic->data());
      } else {
        problem.AddResidualBlock(cost_function, NULL, T_w_c.data(), p_3d.data(),
                                 intrinsic->data());
      }
    }
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

/**
 * Update and add flows to all flows
 * For each match from last frame to current frame as a flow observation if flow
 * exist Else add a new flow
 */
void update_and_add_flows(const FrameCamId& fcid_last, const MatchData& md_last,
                          Flows& all_flows, TrackId& current_flow_id) {
  // Add a matches map here
  // If no match to last frame available
  if (md_last.matches.size() == 0) return;

  // For each flows, find featureId of last frame
  // Map to featureId in new frame with md_last
  // Map featureId last -> current frame
  std::map<FeatureId, FeatureId> matches_map;
  std::set<FeatureId> featuresExistInFlows;
  for (const auto& kv : md_last.matches) {
    matches_map.insert(std::make_pair(kv.first, kv.second));
  }
  FrameCamId fcid_current =
      FrameCamId(fcid_last.frame_id + 1, fcid_last.cam_id);

  // Add observation for existing flows
  for (auto& kv_fl : all_flows) {
    auto& fl = kv_fl.second;

    if (fl.flow.count(fcid_last) > 0) {
      // trackId -> last feature Id
      const FeatureId& last_featureId = fl.flow.at(fcid_last);

      // If there is a match: last featureId -> current featureId
      if (matches_map.count(last_featureId) > 0) {
        featuresExistInFlows.emplace(last_featureId);
        const FeatureId& current_featureId = matches_map.at(last_featureId);

        // Add observer for the flow
        fl.flow.emplace(fcid_current, current_featureId);
        // Increase length of flow
        fl.length++;
      } else {
        // If there is not match
        // Flow is dead
        fl.alive = false;
      }
    }
  }

  // Add new flows for non exist
  for (const auto& kv : md_last.matches) {
    if (featuresExistInFlows.count(kv.first) == 0) {
      Flow fl;
      fl.alive = true;
      fl.flow.emplace(fcid_current, kv.second);
      fl.flow.emplace(fcid_last, kv.first);
      fl.length = 2;
      all_flows.emplace(current_flow_id, fl);
      current_flow_id++;
    }
  }
}

void project_landmarks_of(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam, const Flows& landmarks,
    const double cam_z_threshold,
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

}  // namespace visnav
#endif  // OPTICAL_FLOW_UTILS_H
