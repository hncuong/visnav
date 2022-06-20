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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <sophus/se2.hpp>

namespace visnav {

void matchOptFlow(const pangolin::ManagedImage<uint8_t>& currImgL,
                  const pangolin::ManagedImage<uint8_t>& currImgR,
                  cv::Mat prevImageL, cv::Mat prevImageR, KeypointsData prevKDL,
                  // for now no references consider referencing
                  KeypointsData& currKDL, double threshold) {
  std::cout << "matchOptFlow" << std::endl;
  cv::Mat imageL(currImgL.h, currImgL.w, CV_8U, currImgL.ptr);
  cv::Mat imageR(currImgR.h, currImgR.w, CV_8U, currImgR.ptr);

  std::vector<float> err;
  std::vector<uchar> status;
  currKDL.corners.clear();
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.003);
  std::vector<cv::Point2f> prevPointsL, currPointsL, testL;

  for (auto& c : prevKDL.corners) {
    prevPointsL.push_back(cv::Point2f(c.x(), c.y()));
  }

  std::vector<cv::Mat> prevImageLPyramid, imageLPyramid;
  cv::Size windowSize(5, 5);
  const int& PYRAMID_LEVEL = 3;

  std::cout << "Creating OpticalFlowPyramid" << std::endl;
  cv::buildOpticalFlowPyramid(prevImageL, prevImageLPyramid, windowSize,
                              PYRAMID_LEVEL);
  cv::buildOpticalFlowPyramid(imageL, imageLPyramid, windowSize, PYRAMID_LEVEL);

  std::cout << "prevImageLPyramid: " << prevImageLPyramid.size()
            << " imageLPyramid: " << imageLPyramid.size() << std::endl;

  std::cout << "Start Calculating Optical Flow" << std::endl;
  cv::calcOpticalFlowPyrLK(prevImageLPyramid, imageLPyramid, prevPointsL,
                           currPointsL, status, err, windowSize, 2, criteria);
  cv::calcOpticalFlowPyrLK(imageLPyramid, prevImageLPyramid, currPointsL, testL,
                           status, err, windowSize, 2, criteria);

  std::cout << "currPointsL size: " << currPointsL.size() << std::endl;
  std::cout << "testL size: " << testL.size() << std::endl;
  std::cout << "End Calculating Optical Flow" << std::endl;

  for (int i = 0; i < testL.size(); i++) {
    Eigen::Vector2f diff(abs((testL[i] - prevPointsL[i]).x),
                         abs((testL[i] - prevPointsL[i]).y));
    Eigen::Vector2d tempy(round(currPointsL[i].x), round(currPointsL[i].y));

    if ((diff.allFinite()) && (tempy.x() >= 0) && (tempy.y() >= 0) &&
        ((tempy.y() < imageL.rows && tempy.x() < imageL.cols)) &&
        (diff.norm() < threshold)) {
      currKDL.corners.push_back(tempy);
    }
  }
  std::cout << "NUM KEYPOINTS T-1 to T " << currKDL.corners.size() << " TOTAL "
            << testL.size() << std::endl;
}

// TODO: Make this KD reference
void matchLeftRightOptFlow(const pangolin::ManagedImage<uint8_t>& currImgL,
                           const pangolin::ManagedImage<uint8_t>& currImgR,
                           KeypointsData& currKDL, KeypointsData& currKDR,
                           std::vector<std::pair<int, int>>& matches,
                           double threshold) {
  std::cout << "matchLeftRightOptFlow" << std::endl;
  std::vector<cv::Point2f> currPointsL, currPointsR, currPointsTemp;
  for (auto& c : currKDL.corners) {
    currPointsL.push_back(cv::Point2f(c.x(), c.y()));
  }

  std::vector<float> err;
  std::vector<uchar> status;
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.003);
  cv::Mat imageL(currImgL.h, currImgL.w, CV_8U, currImgL.ptr);
  cv::Mat imageR(currImgR.h, currImgR.w, CV_8U, currImgR.ptr);
  currKDR.corners.clear();
  matches.clear();

  std::vector<cv::Mat> currImgLPyramid, currImgRPyramid;
  cv::Size windowSize(5, 5);
  const int& PYRAMID_LEVEL = 3;

  std::cout << "Creating OpticalFlow Pyramid" << std::endl;
  cv::buildOpticalFlowPyramid(imageL, currImgLPyramid, windowSize,
                              PYRAMID_LEVEL);
  cv::buildOpticalFlowPyramid(imageR, currImgRPyramid, windowSize,
                              PYRAMID_LEVEL);

  std::cout << "currImgLPyramid: " << currImgLPyramid.size()
            << " currImgRPyramid: " << currImgRPyramid.size() << std::endl;

  std::cout << "Do Optical Flow on Image Pyramid L -> R" << std::endl;
  cv::calcOpticalFlowPyrLK(currImgLPyramid, currImgRPyramid, currPointsL,
                           currPointsR, status, err, windowSize, 2, criteria);

  std::cout << "Do Optical Flow on Image Pyramid R -> L" << std::endl;
  cv::calcOpticalFlowPyrLK(currImgRPyramid, currImgLPyramid, currPointsR,
                           currPointsTemp, status, err, windowSize, 2,
                           criteria);

  std::cout << "currPointsL: " << currPointsL.size()
            << " currPointsR: " << currPointsR.size()
            << " currPointsTemp: " << currPointsTemp.size() << std::endl;

  std::cout << "Optical Flow on Image Pyramid is done" << std::endl;

  std::cout << "currPointsTemp: " << currPointsTemp.size() << std::endl;

  for (int i = 0; i < currPointsTemp.size(); i++) {
    Eigen::Vector2f diff(abs((currPointsTemp[i] - currPointsL[i]).x),
                         abs((currPointsTemp[i] - currPointsL[i]).y));
    Eigen::Vector2d tempy(round(currPointsR[i].x), round(currPointsR[i].y));

    currKDR.corners.push_back(tempy);
    if ((diff.allFinite()) && (tempy.x() >= 0) && (tempy.y() >= 0) &&
        ((tempy.y() < imageL.rows && tempy.x() < imageL.cols)) &&
        (diff.norm() < threshold)) {
      matches.push_back(std::make_pair(i, i));
    }
  }

  std::cout << "Matches after optical flow: " << matches.size() << std::endl;
}

// Add new keypoints to current exist keypoints in a frame
// with new detected corners
void add_keypoints(const pangolin::ManagedImage<uint8_t>& img_raw,
                   KeypointsData& kd, int num_features) {
  KeypointsData new_kd;
  // detectKeypoints(img_raw, new_kd, num_features);

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
                                       std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  std::vector<FrameCamId> camera_frame_to_remove;
  std::vector<TrackId> landmark_to_remove;
  while (kf_frames.size() > max_num_kfs) {
    // Get the current frameId to be removed --> set preserve the order (i.e.
    // order by which item is added first)
    FrameId cur_frame = *kf_frames.begin();

    // Removed keyframes should be removed from cameras
    /*
     * Cameras: collection {imageId => Camera} for all cameras in the map
     * - Cameras: std::map<FrameCamId, Camera, std::less<FrameCamId>,
             Eigen::aligned_allocator<std::pair<const FrameCamId, Camera>>>
    */
    for (auto camera : cameras) {
      if (camera.first.frame_id == cur_frame) {
        // Remove the camera
        for (auto& landmark : landmarks) {
          landmark.second.obs.erase(camera.first);
          // landmarks with no left observations should be moved to
          // old_landmarks
          if (landmark.second.obs.empty()) {
            old_landmarks.insert(
                std::make_pair(landmark.first, landmark.second));
            // Add landmark with no lefft observation into vector storing
            // TrackId landmark to be removed
            landmark_to_remove.push_back(landmark.first);
          }
        }
        // Add keyframes to be removed from cameras into vector storing
        // FrameCamId keyframes to be removed
        camera_frame_to_remove.push_back(camera.first);
      }
    }
    // Remove the current keyframe from kf_frames
    kf_frames.erase(cur_frame);

    // Remove each of camera to be removed
    for (auto removed_camera : camera_frame_to_remove) {
      cameras.erase(removed_camera);
    }
    // Remove each of landmark to be removed in landmarks
    // - It was added into old_landmarks instead
    for (auto removed_landmark : landmark_to_remove) {
      landmarks.erase(removed_landmark);
    }
    camera_frame_to_remove.clear();
    landmark_to_remove.clear();
  }
}

}  // namespace visnav
