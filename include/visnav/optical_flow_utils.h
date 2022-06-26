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
#include <opencv2/features2d.hpp>

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <sophus/se2.hpp>

namespace visnav {

// keep track of next available unique id for detected keypoints
int last_keypoint_id = 0;
// Finding the matches between projected_landmarks and detected keypoints in the
// current frame
void findMatchesLandmarksOpticalFlow(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids, MatchData& md) {
  md.matches.clear();
}

void MatchOpticalFlow(const pangolin::ManagedImage<uint8_t>& imgl,
                      const pangolin::ManagedImage<uint8_t>& imgr,
                      KeypointsData& kdl, KeypointsData& kdr,
                      std::vector<std::pair<FeatureId, FeatureId>>& matches,
                      cv::TermCriteria criteria, cv::Size windowSize,
                      int pyramid_lv, double threshold) {
  std::vector<cv::Point2f> currPointsL, currPointsR, currPointsTemp;
  for (auto& c : kdl.corners) {
    currPointsL.push_back(cv::Point2f(c.x(), c.y()));
  }
  std::vector<float> err;
  std::vector<uchar> status;

  cv::Mat imageL(imgl.h, imgl.w, CV_8U, imgl.ptr);
  cv::Mat imageR(imgr.h, imgr.w, CV_8U, imgr.ptr);
  kdr.corners.clear();

  std::vector<cv::Mat> imageL_pyramid, imageR_pyramid;

  cv::buildOpticalFlowPyramid(imageL, imageL_pyramid, windowSize, pyramid_lv);
  cv::buildOpticalFlowPyramid(imageR, imageR_pyramid, windowSize, pyramid_lv);

  cv::calcOpticalFlowPyrLK(imageL_pyramid, imageR_pyramid, currPointsL,
                           currPointsR, status, err, windowSize, pyramid_lv,
                           criteria);
  cv::calcOpticalFlowPyrLK(imageR_pyramid, imageL_pyramid, currPointsR,
                           currPointsTemp, status, err, windowSize, pyramid_lv,
                           criteria);

  for (int i = 0; i < currPointsTemp.size(); i++) {
    Eigen::Vector2d diff(std::abs((currPointsTemp[i] - currPointsL[i]).x),
                         std::abs((currPointsTemp[i] - currPointsL[i]).y));
    Eigen::Vector2d right_keypoint(currPointsL[i].x, currPointsL[i].y);

    kdr.corners.push_back(right_keypoint);
    if ((diff.allFinite()) && (diff.norm() < threshold)) {
      matches.push_back(std::make_pair(i, i));
    }
  }
  std::cout << kdr.corners.size() << " " << matches.size() << std::endl;
}

/* Detect keypoints in the image frame
 * TODO: Use Grid-based keypoint detection
 */
void detectKeypoints_optical_flow(pangolin::ManagedImage<u_int8_t>& img,
                                  KeypointsData& kd, double fast_t) {
  cv::Mat img_cv(img.h, img.w, CV_8U, img.ptr);
  std::vector<cv::KeyPoint> keypoints, selected_keypoints;
  cv::FAST(img_cv, keypoints, fast_t);

  for (size_t i = 0; i < keypoints.size(); i++) {
    kd.corners.emplace_back(keypoints[i].pt.x, keypoints[i].pt.y);
  }
}

void detectKeyPointsOpticalFlowSingleCam(
    const pangolin::ManagedImage<uint8_t>& img_last,
    const pangolin::ManagedImage<uint8_t>& img_current, KeypointsData& last_kd,
    KeypointsData& current_kd, cv::TermCriteria criteria, cv::Size windowSize,
    int pyramid_lv, double threshold) {
  current_kd.corners.clear();
  current_kd.corner_angles.clear();
  current_kd.corner_descriptors.clear();

  cv::Mat img_last_cv(img_last.h, img_last.w, CV_8U, img_last.ptr);
  cv::Mat img_current_cv(img_current.h, img_current.w, CV_8U, img_current.ptr);

  std::vector<cv::Point2f> img_last_points, img_current_points, img_tmp_points;
  std::vector<cv::Mat> img_last_pyr, img_current_pyr;
  std::vector<uchar> status;
  std::vector<float> err;

  // Build image pyramid corresponding to each cv image
  cv::buildOpticalFlowPyramid(img_last_cv, img_last_pyr, windowSize,
                              pyramid_lv);
  cv::buildOpticalFlowPyramid(img_current_cv, img_current_pyr, windowSize,
                              pyramid_lv);

  std::vector<FeatureId> img_last_fids;

  // Assign all keypoints & its corresponding FeatureId to img_last_points &
  // img_last_fids
  for (size_t i = 0; i < last_kd.corners.size(); i++) {
    cv::Point2f last_kd_corner_cv(last_kd.corners.at(i)[0],
                                  last_kd.corners.at(i)[1]);
    img_last_points.push_back(last_kd_corner_cv);
    img_last_fids.emplace_back(i);
  }

  // Optical flow to track points from img_last_pyr to img_current_pyr
  cv::calcOpticalFlowPyrLK(img_last_pyr, img_current_pyr, img_last_points,
                           img_current_points, status, err, windowSize,
                           pyramid_lv, criteria);
  std::vector<cv::Point2f> good_img_current_points;
  std::vector<int> good_img_current_point_ids;
  // Select only points which are visible in img_current_pyr
  for (size_t i = 0; i < img_last_points.size(); i++) {
    if (status[i] == 1) {
      good_img_current_points.push_back(img_current_points[i]);
      good_img_current_point_ids.push_back(i);
    }
  }

  status.clear();
  err.clear();
  // Optical flow to track points from img_cuurent_pyr to img_last_pyr
  cv::calcOpticalFlowPyrLK(img_current_pyr, img_last_pyr,
                           good_img_current_points, img_tmp_points, status, err,
                           windowSize, pyramid_lv, criteria);

  for (size_t i = 0; i < good_img_current_points.size(); i++) {
    if (status[i] == 1) {
      Eigen::Vector2d origin(img_last_points[good_img_current_point_ids[i]].x,
                             img_last_points[good_img_current_point_ids[i]].y);
      Eigen::Vector2d backward_point(img_tmp_points[i].x, img_tmp_points[i].y);
      double distance = (origin - backward_point).squaredNorm();
      if (distance < threshold) {
        Sophus::SE2d transformation;
        Eigen::Matrix2d rotation = Eigen::Matrix2d::Identity();
        transformation.setRotationMatrix(rotation);
        Eigen::Vector2d current_point(
            img_current_points[good_img_current_point_ids[i]].x,
            img_current_points[good_img_current_point_ids[i]].y);

        transformation.translation() = current_point;

        current_kd.corners.push_back(current_point);
      }
    }
  }
}

// void matchOpticalFlow(const pangolin::ManagedImage<uint8_t>& img_last,
//                      const pangolin::ManagedImage<uint8_t>& img_current,
//                      KeypointsData& current_kdl, KeypointsData& current_kdr,
//                      std::vector<std::pair<int, int>>& matches,
//                      cv::TermCriteria criteria, cv::Size windowSize,
//                      int pyramid_lv, double threshold) {
//  cv::Mat img_last_cv(img_last.h, img_last.w, CV_8U, img_last.ptr);
//  cv::Mat img_current_cv(img_current.h, img_current.w, CV_8U,
//  img_current.ptr);

//  std::vector<cv::Point2f> img_last_points, img_current_points,
//  img_tmp_points; std::vector<cv::Mat> img_last_pyr, img_current_pyr;
//  std::vector<uchar> status;
//  std::vector<float> err;

//  // Build image pyramid corresponding to each cv image
//  cv::buildOpticalFlowPyramid(img_last_cv, img_last_pyr, windowSize,
//                              pyramid_lv);
//  cv::buildOpticalFlowPyramid(img_current_cv, img_current_pyr, windowSize,
//                              pyramid_lv);

//  std::vector<FeatureId> img_last_fids;

//  // Assign all keypoints & its corresponding FeatureId to img_last_points &
//  // img_last_fids
//  for (size_t i = 0; i < current_kdl.corners.size(); i++) {
//    cv::Point2f last_kd_corner_cv(current_kdl.corners.at(i)[0],
//                                  current_kdl.corners.at(i)[1]);
//    img_last_points.push_back(last_kd_corner_cv);
//    img_last_fids.emplace_back(i);
//  }

//  // Optical flow to track points from img_last_pyr to img_current_pyr
//  cv::calcOpticalFlowPyrLK(img_last_pyr, img_current_pyr, img_last_points,
//                           img_current_points, status, err, windowSize,
//                           pyramid_lv, criteria);
//  std::vector<cv::Point2f> good_img_current_points;
//  std::vector<int> good_img_current_point_ids;
//  // Select only points which are visible in img_current_pyr
//  for (size_t i = 0; i < img_last_points.size(); i++) {
//    if (status[i] == 1) {
//      good_img_current_points.push_back(img_current_points[i]);
//      good_img_current_point_ids.push_back(i);
//    }
//  }

//  status.clear();
//  err.clear();
//  // Optical flow to track points from img_cuurent_pyr to img_last_pyr
//  cv::calcOpticalFlowPyrLK(img_current_pyr, img_last_pyr,
//                           good_img_current_points, img_tmp_points, status,
//                           err, windowSize, pyramid_lv, criteria);

//  for (size_t i = 0; i < good_img_current_points.size(); i++) {
//    if (status[i] == 1) {
//      Eigen::Vector2d origin(img_last_points[good_img_current_point_ids[i]].x,
//                             img_last_points[good_img_current_point_ids[i]].y);
//      Eigen::Vector2d backward_point(img_tmp_points[i].x,
//      img_tmp_points[i].y); double distance = (origin -
//      backward_point).squaredNorm();

//      Eigen::Vector2d right_point(good_img_current_points[i].x,
//                                  good_img_current_points[i].y);

//      current_kdr.corners.push_back(right_point);
//      if (distance < threshold) {
//        matches.push_back(std::make_pair(i, i));
//      }
//    }
//  }
//}

void findInlierEssential_optical_flow(
    std::vector<std::map<FeatureId, Sophus::SE2d>> flows,
    const std::shared_ptr<AbstractCamera<double>>& cam1,
    const std::shared_ptr<AbstractCamera<double>>& cam2, Eigen::Matrix3d& E,
    double epipolar_error_threshold) {
  std::vector<int> keypoints_to_filter;

  for (auto& left_image_flow : flows.at(1)) {
    // keypoint is detected in both left and right images
    FeatureId left_image_fid = left_image_flow.first;
    if (flows.at(0).find(left_image_fid) != flows.at(0).end()) {
      Eigen::Vector2d p_l =
          flows.at(0).find(left_image_fid)->second.translation();
      Eigen::Vector2d p_r = left_image_flow.second.translation();

      Eigen::Vector3d proj_l = cam1->unproject(p_l);
      Eigen::Vector3d proj_r = cam2->unproject(p_r);

      // check epipolar constraint
      if (abs(proj_l.transpose() * E * proj_r) > epipolar_error_threshold) {
        keypoints_to_filter.emplace_back(left_image_fid);
      }
    }
  }

  // remove keypoints from right camera if epipolar constraint is failed
  for (FeatureId removed_fid : keypoints_to_filter) {
    flows.at(1).erase(removed_fid);
  }
}

void match_optical_flow(const pangolin::ManagedImage<uint8_t>& img_last,
                        const pangolin::ManagedImage<uint8_t>& img_current,
                        cv::TermCriteria criteria, cv::Size windowSize,
                        int pyramid_lv, double threshold,
                        std::map<int, Sophus::SE2d>& flow_img_last,
                        std::map<int, Sophus::SE2d>& flow_img_current) {
  if (flow_img_last.size() == 0) return;

  cv::Mat img_last_cv(img_last.h, img_last.w, CV_8U, img_last.ptr);
  cv::Mat img_current_cv(img_current.h, img_current.w, CV_8U, img_current.ptr);

  std::vector<cv::Point2f> img_last_points, img_current_points, img_tmp_points;
  std::vector<cv::Mat> img_last_pyr, img_current_pyr;
  std::vector<uchar> status;
  std::vector<float> err;

  // Build image pyramid corresponding to each cv image
  cv::buildOpticalFlowPyramid(img_last_cv, img_last_pyr, windowSize,
                              pyramid_lv);
  cv::buildOpticalFlowPyramid(img_current_cv, img_current_pyr, windowSize,
                              pyramid_lv);

  std::vector<FeatureId> img_last_fids;

  // Assign all keypoints & its corresponding FeatureId to img_last_points &
  // img_last_fids
  for (auto last_flow : flow_img_last) {
    Eigen::Vector2d p = last_flow.second.translation();
    cv::Point2f p_cv(p[0], p[1]);
    img_last_points.emplace_back(p_cv);
    img_last_fids.emplace_back(last_flow.first);
  }

  // Optical flow to track points from img_last_pyr to img_current_pyr
  cv::calcOpticalFlowPyrLK(img_last_pyr, img_current_pyr, img_last_points,
                           img_current_points, status, err, windowSize,
                           pyramid_lv, criteria);
  std::vector<cv::Point2f> good_img_current_points;
  std::vector<int> good_img_current_point_ids;
  // Select only points which are visible in img_current_pyr
  for (size_t i = 0; i < img_last_points.size(); i++) {
    if (status[i] == 1) {
      good_img_current_points.push_back(img_current_points[i]);
      good_img_current_point_ids.push_back(i);
    }
  }

  status.clear();
  err.clear();
  // Optical flow to track points from img_cuurent_pyr to img_last_pyr
  cv::calcOpticalFlowPyrLK(img_current_pyr, img_last_pyr,
                           good_img_current_points, img_tmp_points, status, err,
                           windowSize, pyramid_lv, criteria);

  for (size_t i = 0; i < good_img_current_points.size(); i++) {
    if (status[i] == 1) {
      Eigen::Vector2d origin(img_last_points[good_img_current_point_ids[i]].x,
                             img_last_points[good_img_current_point_ids[i]].y);
      Eigen::Vector2d backward_point(img_tmp_points[i].x, img_tmp_points[i].y);
      double distance = (origin - backward_point).squaredNorm();
      if (distance < threshold) {
        Sophus::SE2d transformation;
        Eigen::Matrix2d rotation = Eigen::Matrix2d::Identity();
        transformation.setRotationMatrix(rotation);
        Eigen::Vector2d current_point(
            img_current_points[good_img_current_point_ids[i]].x,
            img_current_points[good_img_current_point_ids[i]].y);

        transformation.translation() = current_point;

        flow_img_current[img_last_fids[good_img_current_point_ids[i]]] =
            transformation;
      }
    }
  }
}

void add_points(std::vector<std::map<int, Sophus::SE2d>>& transforms,
                pangolin::ManagedImage<uint8_t>& left_image,
                pangolin::ManagedImage<uint8_t>& right_image, int fast_t,
                cv::TermCriteria criteria, cv::Size windowSize,
                bool track_new_points_to_right, int pyramid_level,
                double distance_threshold) {
  cv::Mat left_image_cv(left_image.h, left_image.w, CV_8U, left_image.ptr);
  cv::Mat right_image_cv(right_image.h, right_image.w, CV_8U, right_image.ptr);
  std::vector<Eigen::Vector2d> points_l;

  // keep tracked points for left image
  for (const auto& kv : transforms.at(0)) {
    points_l.emplace_back(kv.second.translation());
  }

  KeypointsData new_points;

  detectKeypoints_optical_flow(left_image, new_points, fast_t);

  std::map<FeatureId, Sophus::SE2d> new_poses_l, new_poses_r;

  for (size_t i = 0; i < new_points.corners.size(); ++i) {
    // it is newly found, there is no extra transformation than its position
    Eigen::Matrix2d rot = Eigen::Matrix2d::Identity();

    transforms.at(0)[last_keypoint_id].setRotationMatrix(rot);
    transforms.at(0)[last_keypoint_id].translation() = new_points.corners[i];

    new_poses_l[last_keypoint_id].setRotationMatrix(rot);
    new_poses_l[last_keypoint_id].translation() = new_points.corners[i];

    last_keypoint_id++;
  }

  if (new_poses_l.size() > 0 && track_new_points_to_right) {
    // track points from left camera to right camera
    match_optical_flow(left_image, right_image, criteria, windowSize,
                       pyramid_level, distance_threshold, new_poses_l,
                       new_poses_r);
    for (const auto& kv : new_poses_r) {
      transforms.at(1).emplace(kv);
    }
  }
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

void find_matches_landmarks_optical_flow(
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
  //  UNUSED(kdl);
  //  UNUSED(landmarks);
  //  UNUSED(feature_corners);
  //  UNUSED(projected_points);
  //  UNUSED(projected_track_ids);
  //  UNUSED(match_max_dist_2d);
  //  UNUSED(feature_match_threshold);
  //  UNUSED(feature_match_dist_2_best);

  /*
   * Do matching similarly to brute force matching of ORB Features in Exercise 3
   */
  for (size_t kp_idx = 0; kp_idx < kdl.corners.size(); kp_idx++) {
    int min_distance = 257;
    int second_min_distance = 257;
    int best_match = -1;
    Eigen::Vector2d cur_corner = kdl.corners.at(kp_idx);
    std::bitset<256> cur_corner_descriptor = kdl.corner_descriptors.at(kp_idx);

    for (size_t pt_idx = 0; pt_idx < projected_points.size(); pt_idx++) {
      Eigen::Vector2d projected_point = projected_points.at(pt_idx);
      // Check if detected corners is in a circle with radius match_max_dist_2d
      if ((cur_corner - projected_point).squaredNorm() <=
          match_max_dist_2d * match_max_dist_2d) {
        TrackId pt_track_id = projected_track_ids.at(pt_idx);
        Landmark landmark = landmarks.at(pt_track_id);
        int min_dist_landmark = 257;
        /*
         * Iterate through landmark.obs --> Inlier observations in the current
        // map.
         * FeatureTrack --> Feature tracks are collections of {ImageId =>
        FeatureId}.
         * - I.e. a collection of all images that observed this feature and the
        corresponding feature index in that image.
        */
        for (std::pair<FrameCamId, FeatureId> inlier_obs : landmark.obs) {
          FrameCamId fcid = inlier_obs.first;
          FrameId fid = inlier_obs.second;
          std::bitset<256> feature_corner_descriptor =
              feature_corners.at(fcid).corner_descriptors.at(fid);
          int hamming_distance =
              (cur_corner_descriptor ^ feature_corner_descriptor).count();
          min_dist_landmark = std::min(min_dist_landmark, hamming_distance);
        }

        if (min_dist_landmark < min_distance) {
          second_min_distance = min_distance;
          min_distance = min_dist_landmark;
          best_match = projected_track_ids.at(pt_idx);
        } else if (min_dist_landmark < second_min_distance) {
          second_min_distance = min_dist_landmark;
        }
      }
    }

    /*
     * Discard matches in 2 scenarios
     * 1) Discard matches when the min_distance is greater than
    feature_match_threshold
     * 2) Discard matches if the distance to the second last best match is
    smaller than the smallest distance multiplied by feature_match_dist_2_best
    */
    if ((min_distance >= feature_match_threshold) ||
        (second_min_distance < feature_match_dist_2_best * min_distance)) {
      continue;
    }
    // Appending matches into LandmarkMatchData storing image to landmark
    // matches
    md.matches.push_back(std::make_pair(kp_idx, best_match));
  }
}  // namespace visnav

// Remove the keypoints that die out in the

void localize_camera_optical_flow(
    const std::shared_ptr<AbstractCamera<double>>& cam,
    std::map<FeatureId, Sophus::SE2d>& flows_per_frame,
    const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    LandmarkMatchData& md, Sophus::SE3d& T_w_c, std::vector<int>& inliers) {
  inliers.clear();

  T_w_c = Sophus::SE3d();

  if (md.matches.size() < 4) {
    return;
  }

  opengv::bearingVectors_t bearing_vectors;
  opengv::points_t points_t;

  //  size_t numberPoints = md.matches.size();
  //  // Reserve the size of bearing vectors and points = # matches
  //  bearing_vectors.reserve(numberPoints);
  //  points_t.reserve(numberPoints);

  /*
   * - Adding unprojected points from the camera into bearing_vectors
   * - Adding landmarks at specific tracks into points
   * - pass bearing_vectors and points into CentralAbsoluteAdapter
   */

  // Find keypoints that belong to landmarks -> Fill with correspondences in
  // matches
  for (auto l : landmarks) {
    std::cout << "Landmark trackId: " << l.first << std::endl;
  }

  std::cout << "Loop through each item in flows_per_item" << std::endl;

  for (auto& f : flows_per_frame) {
    std::cout << "flows_per_frame item:  (" << f.first << ", "
              << f.second.translation() << ")" << std::endl;

    // In case that we found the landmark with specific FeatureId
    if (landmarks.find(f.first) != landmarks.end()) {
      // 3D point in world frame
      points_t.emplace_back(landmarks.at(f.first).p);

      // 2D -> 3D point in cam frame
      const auto& p_2c = f.second.translation();
      bearing_vectors.emplace_back(cam->unproject(p_2c));

      // populate md to use later in add_new_landmarks.
      // `inliers` will contain indices from md.matches
      md.matches.emplace_back(std::make_pair(f.first, f.first));
    }
  }

  std::cout << "md.matches size: " << md.matches.size() << std::endl;

  std::cout << "points_t size: " << points_t.size()
            << " bearing_vectors size: " << bearing_vectors.size() << std::endl;

  if (points_t.size() == 0 || bearing_vectors.size() == 0) return;

  /*
   * Use CentralAbsoluteAdapter & corresponding RANSAC implementation
   * AbsolutePoseSacProblem
   * - AbsolutePoseSacProblem that uses a minimal variant of PnP taking
   * exactly 3 points: KNEIP
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
                                            inliers);

    // Return the new transformation matrix in world coordinates (refined
    // pose)
    Eigen::Matrix4d res;
    res.block<3, 3>(0, 0) = optimized.block<3, 3>(0, 0);
    res.block<3, 1>(0, 3) = optimized.block<3, 1>(0, 3);
    res.block<1, 4>(3, 0) = Eigen::Vector4d(0, 0, 0, 1);

    T_w_c = Sophus::SE3d(res);
  }
}

void add_new_landmarks_optical_flow(
    const FrameCamId fcidl, const FrameCamId fcidr, const KeypointsData& kdl,
    const KeypointsData& kdr, const Calibration& calib_cam,
    Sophus::SE3d& T_w_c0, Landmarks& landmarks, std::vector<int>& inliers,
    const MatchData& md_stereo, const LandmarkMatchData& md,
    TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);  // Left camera should have camera_id = 0
  assert(fcidr.cam_id == 1);  // Right camera should have camera_id = 1
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

  std::set<std::pair<FeatureId, FeatureId>> existing_landmark;

  for (auto landmark_inlier : md.inliers) {
    FeatureId feature_id = landmark_inlier.first;
    TrackId track_id = landmark_inlier.second;
    // Add new Landmarks
    std::pair<FrameCamId, FeatureId> landmark_pair =
        std::make_pair(fcidl, feature_id);

    landmarks.at(track_id).obs.insert(landmark_pair);
    /*
     * Check if keypoints also appears in md_stereo.inliers
     */
    for (auto stereo_inlier : md_stereo.inliers) {
      if (feature_id == stereo_inlier.first) {
        // Add keypoints that also appears in md_stereo.inliers into the
        // existing_landmark to check the existence of the keypoint
        existing_landmark.insert(stereo_inlier);
        TrackId stereo_track_id = stereo_inlier.second;
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
  opengv::bearingVectors_t bearing_vec_1;
  opengv::bearingVectors_t bearing_vec_2;

  // i = second image coordinate system j = first image coordinate
  // MatchData.inlier: collection of {featureId_i, featureId_j} pairs of all
  // matches
  for (auto stereo_inlier : md_stereo.inliers) {
    bearing_vec_1.push_back(calib_cam.intrinsics[fcidl.cam_id]->unproject(
        kdl.corners.at(stereo_inlier.first)));
    bearing_vec_2.push_back(calib_cam.intrinsics[fcidr.cam_id]->unproject(
        kdr.corners.at(stereo_inlier.second)));
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearing_vec_1, bearing_vec_2, t_0_1, R_0_1);
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
