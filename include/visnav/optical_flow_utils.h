#ifndef OPTICAL_FLOW_UTILS_H
#define OPTICAL_FLOW_UTILS_H

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
    LandmarkMatchData& md) {}

void add_new_landmarks_optical_flow(
    const FrameCamId fcidl, const FrameCamId fcidr, const KeypointsData& kdl,
    const KeypointsData& kdr, const Calibration& calib_cam,
    const MatchData& md_stereo, const LandmarkMatchData& md,
    Landmarks& landmarks, TrackId& next_landmark_id) {}

void remove_old_keyframes_optical_flow(const FrameCamId fcidl,
                                       const int max_num_kfs, Cameras& cameras,
                                       Landmarks& landmarks,
                                       Landmarks& old_landmarks,
                                       std::set<FrameId>& kf_frames) {}

}  // namespace visnav
#endif  // OPTICAL_FLOW_UTILS_H
