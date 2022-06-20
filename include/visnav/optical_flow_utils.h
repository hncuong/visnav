#ifndef OPTICAL_FLOW_UTILS_H
#define OPTICAL_FLOW_UTILS_H

#include <set>
#include <vector>

#include <visnav/common_types.h>
#include <visnav/calibration.h>
#include <visnav/keypoints.h>

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

const int PYRAMID_LEVEL = 3;

namespace visnav {

// Add new keypoints to current exist keypoints in a frame
// with new detected corners
void add_keypoints(const pangolin::ManagedImage<uint8_t>& img_raw,
                   KeypointsData& kd, int num_features) {
  KeypointsData new_kd;
  detectKeypoints(img_raw, new_kd, num_features);

  // TODO Update to check overlap keypoints later
  for (const auto& kp : new_kd.corners) {
    kd.corners.emplace_back(kp);
  }
}

// TODO Update this function
// Use optical flow from a last frame to a new frame to establish
//  1. keypoints in new frame
//  2. match data between last frame and new frame
void optical_flows(const pangolin::ManagedImage<uint8_t>& img_last,
                   const pangolin::ManagedImage<uint8_t>& img_current,
                   const KeypointsData& kd_last, KeypointsData& kd_current,
                   MatchData& md) {
  // If last frame have empty corner
  if (kd_last.corners.size() == 0) return;

  // Create two frame and convert to Gray
  cv::Mat old_frame(img_last.h, img_last.w, CV_8U, img_last.ptr);
  cv::Mat frame(img_current.h, img_current.w, CV_8U, img_current.ptr);

  // Keypoints data
  std::vector<cv::Point2f> p0, p1;
  p0.reserve(kd_last.corners.size());
  p1.reserve(kd_last.corners.size());
  for (size_t i = 0; i < kd_last.corners.size(); i++) {
    p0.emplace_back(kd_last.corners.at(i).x(), kd_last.corners.at(i).y());
  }

  // Optical Flows
  std::vector<uchar> status;
  std::vector<float> err;
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);

  cv::calcOpticalFlowPyrLK(old_frame, frame, p0, p1, status, err,
                           cv::Size(15, 15), 2, criteria);

  // TODO Inverse optical flows

  // Convert results to current_frame_kd and Match data
  int match_id = 0;
  for (size_t i = 0; i < p1.size(); i++) {
    if (status[i] == 1) {
      kd_current.corners.emplace_back(p1[i].x, p1[i].y);
      md.matches.emplace_back(i, match_id);
      match_id++;
    }
  }
}

// Compute matches of current frame -> flows
// Based on matches of current frame -> last frame
// Remove flows that has no match in current frame
void find_matches_landmarks_with_otpical_flow(const FrameCamId& fcid_last,
                                              const MatchData& md_last,
                                              Landmarks& flows,
                                              LandmarkMatchData& md) {
  // If no match to last frame available
  if (md_last.matches.size() == 0) return;

  // For each landmarks, find featureId of last frame
  // Map to featureId in new frame with md_last
  // Map featureId last -> current frame
  std::unordered_map<FeatureId, FeatureId> matches_map(md_last.matches.begin(),
                                                       md_last.matches.end());

  std::vector<TrackId> flow_to_discard;
  for (auto& kv_lm : flows) {
    const TrackId& trackId = kv_lm.first;
    FeatureTrack& obs = kv_lm.second.obs;

    // Actually not expected to have obs of last = 0
    // Cause we gonna kill dieout flows later
    if (obs.count(fcid_last) > 0) {
      // trackId -> last feature Id
      const FeatureId& last_featureId = obs.at(fcid_last);

      // If there is a match: last featureId -> current featureId
      if (matches_map.count(last_featureId) > 0) {
        const FeatureId& current_featureId = matches_map.at(last_featureId);
        md.matches.emplace_back(current_featureId, trackId);

        // Add observer for the flow
        FrameCamId fcid_current =
            FrameCamId(fcid_last.frame_id + 1, fcid_last.cam_id);
        obs.emplace(fcid_current, current_featureId);
      } else {
        // If there is not match
        // Discard the flows
        flow_to_discard.emplace_back(trackId);
      }
    }
  }

  // Now discard dieout flows
  for (const auto& trackId : flow_to_discard) {
    flows.erase(trackId);
  }
}

// Remove the keypoints that die out in the
// TODO Update this function
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

/*
 * Add new landmarks and flows for Optical flows
 */
void add_new_landmarks_optical_flow(
    const FrameCamId fcidl, const FrameCamId fcidr, const KeypointsData& kdl,
    const KeypointsData& kdr, const Calibration& calib_cam,
    const MatchData& md_stereo, const LandmarkMatchData& md,
    Landmarks& landmarks, TrackId& next_landmark_id, Landmarks& flows) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  const auto& T_w_c = md.T_w_c;
  const auto& stereo_inliers = md_stereo.inliers;  // FeatureId, FeatureId
  const auto& landmark_inliers = md.inliers;       // FeatureId, TrackId
  std::unordered_map<FeatureId, FeatureId> stereo_inliers_map(
      stereo_inliers.begin(), stereo_inliers.end());
  const FeatureId LANDMARK_EXIST = -1;

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
      stereo_inliers_map.at(featureId_left) = LANDMARK_EXIST;
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
    if (stereo_inliers_map.at(featureId_lr.first) != LANDMARK_EXIST) {
      opengv::point_t p =
          T_w_c * opengv::triangulation::triangulate(adapter, i);

      Landmark lm, flow;
      lm.p = p;
      lm.obs.emplace(fcidl, featureId_lr.first);
      lm.obs.emplace(fcidr, featureId_lr.second);
      flow.obs.emplace(fcidl, featureId_lr.first);
      flow.obs.emplace(fcidr, featureId_lr.second);

      // Add new landmark
      // and flow
      landmarks.emplace(next_landmark_id, lm);
      flows.emplace(next_landmark_id, flow);

      next_landmark_id++;
    }
  }
}

// TODO Update this function
void remove_old_keyframes_optical_flow(const FrameCamId fcidl,
                                       const int max_num_kfs, Cameras& cameras,
                                       Landmarks& landmarks,
                                       Landmarks& old_landmarks,
                                       std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // Find old keyframes to be removed
  if (kf_frames.size() <= max_num_kfs) return;

  std::vector<TrackId> to_remove_landmarks;

  while (kf_frames.size() > max_num_kfs) {
    auto min_frameId = *kf_frames.begin();
    FrameCamId min_fcidl = FrameCamId(min_frameId, 0);
    FrameCamId min_fcidr = FrameCamId(min_frameId, 1);

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
#endif  // OPTICAL_FLOW_UTILS_H
