#ifndef OPTICAL_FLOW_UTILS_H
#define OPTICAL_FLOW_UTILS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <visnav/common_types.h>
#include <visnav/calibration.h>

#include <set>
#include <vector>

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
  std::vector<cv::uchar> status;
  std::vector<float> err;
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);

  cv::calcOpticalFlowPyrLK(old_frame, frame, p0, p1, status, err, Size(15, 15),
                           2, criteria);

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

// TODO Update this function
// Compute matches of current frame -> landmarks
// Based on matches of current frame -> last frame
void find_matches_landmarks_with_otpical_flow(const FrameCamId& fcid_last,
                                              const MatchData& md_last,
                                              const Landmarks& landmarks,
                                              LandmarkMatchData& md) {
  // If no match to last frame available
  if (md_last.matches.size() == 0) return;

  // For each landmarks, find featureId of last frame
  // Map to featureId in new frame with md_last
  // Map featureId last -> current frame
  std::unordered_map<FeatureId, FeatureId> matches_map(md_last.matches.begin(),
                                                       md_last.matches.end());

  for (const auto& kv_lm : landmarks) {
    const TrackId& trackId = kv_lm.first;
    const FeatureTrack& obs = kv_lm.second.obs;

    if (obs.count(fcid_last) > 0) {
      // trackId -> last feature Id
      const FeatureId& last_featureId = obs.at(fcid_last);
      // last featureId -> current featureId
      if (matches_map.count(last_featureId) > 0) {
        const FeatureId& current_featureId = matches_map.at(last_featureId);
        md.matches.emplace_back(current_featureId, trackId);
      }
    }
  }
}

// Remove the keypoints that die out in the
// TODO Update this function
void localize_camera_optical_flow(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const KeypointsData& kdl, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    LandmarkMatchData& md) {}

// TODO Update this function
void add_new_landmarks_optical_flow(
    const FrameCamId fcidl, const FrameCamId fcidr, const KeypointsData& kdl,
    const KeypointsData& kdr, const Calibration& calib_cam,
    const MatchData& md_stereo, const LandmarkMatchData& md,
    Landmarks& landmarks, TrackId& next_landmark_id) {}
// TODO Update this function
void remove_old_keyframes_optical_flow(const FrameCamId fcidl,
                                       const int max_num_kfs, Cameras& cameras,
                                       Landmarks& landmarks,
                                       Landmarks& old_landmarks,
                                       std::set<FrameId>& kf_frames) {}

}  // namespace visnav
#endif  // OPTICAL_FLOW_UTILS_H
