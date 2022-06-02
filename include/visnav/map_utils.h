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

#include <fstream>
#include <thread>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/local_parameterization_se3.hpp>
#include <visnav/reprojection.h>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string &map_path, const Corners &feature_corners,
                   const Matches &feature_matches,
                   const FeatureTracks &feature_tracks,
                   const FeatureTracks &outlier_tracks, const Cameras &cameras,
                   const Landmarks &landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto &kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string &map_path, Corners &feature_corners,
                   Matches &feature_matches, FeatureTracks &feature_tracks,
                   FeatureTracks &outlier_tracks, Cameras &cameras,
                   Landmarks &landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto &kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const FrameCamId &fcid0,
                                   const FrameCamId &fcid1,
                                   const Calibration &calib_cam,
                                   const Corners &feature_corners,
                                   const FeatureTracks &feature_tracks,
                                   const Cameras &cameras,
                                   Landmarks &landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<FrameCamId> fcids = {fcid0, fcid1};
  if (!GetTracksInImages(fcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // TODO SHEET 4: Triangulate all new features and add to the map

  // Create to bearing vectors vector<Vector3d>
  // Using Calibcam
  size_t numberPoints = shared_track_ids.size();
  opengv::bearingVectors_t bearingVectors0;
  opengv::bearingVectors_t bearingVectors1;
  bearingVectors0.reserve(numberPoints);
  bearingVectors1.reserve(numberPoints);

  //  auto cam0 = calib_cam.intrinsics.at(fcid0.cam_id);
  //  auto cam1 = calib_cam.intrinsics.at(fcid1.cam_id);

  const auto &corners0 = feature_corners.at(fcid0).corners;
  const auto &corners1 = feature_corners.at(fcid1).corners;

  for (const auto &track_id : shared_track_ids) {
    //    if (landmarks.count(track_id) == 0) {
    const auto &feature_track = feature_tracks.at(track_id);
    const auto &feature_id0 = feature_track.at(fcid0);
    const auto &feature_id1 = feature_track.at(fcid1);

    const auto &point_0 = corners0.at(feature_id0);
    const auto &point_1 = corners1.at(feature_id1);

    bearingVectors0.emplace_back(
        calib_cam.intrinsics.at(fcid0.cam_id)->unproject(point_0));
    bearingVectors1.emplace_back(
        calib_cam.intrinsics.at(fcid1.cam_id)->unproject(point_1));
    //    }
  }

  // TODO Unproject 2D points from Corners of FrameCamId and FeatureId

  // position t of the second camera seen from the first one and
  // the rotation R from the second camera back to the first camera frame.
  // [R | t] is the transformation of position p1 from first camera coordinate
  // to position p2 in second camera coordinate
  // i.e. p2 = R * p1 + t
  // While position of cameras look opposite: c1 = c2 + t
  auto T_w_c0 = cameras.at(fcid0).T_w_c;
  auto T_w_c1 = cameras.at(fcid1).T_w_c;
  auto T_0_1 = T_w_c0.inverse() * T_w_c1;
  opengv::translation_t translation = T_0_1.translation();
  opengv::rotation_t rotation = T_0_1.rotationMatrix();

  // create a central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearingVectors0, bearingVectors1, translation, rotation);

  // Find overlapping tracks that are not yet landmarks and add to scene.
  // We go through existing cams one by one to triangulate landmarks
  // pair-wise. If there are additional cameras in the existing map also
  // sharing the same track, we add observations after triangulation.
  for (size_t j = 0; j < numberPoints; j++) {
    const auto &track_id = shared_track_ids[j];
    if (landmarks.count(track_id) == 0) {
      opengv::point_t p =
          T_w_c0 * opengv::triangulation::triangulate(adapter, j);
      // TODO Rotate to world cam

      Landmark lm;
      lm.p = p;
      const FeatureTrack &feature_track = feature_tracks.at(track_id);
      for (const auto &kv : feature_track) {
        // Check of camera already in map
        if (cameras.count(kv.first) > 0) {
          lm.obs[kv.first] = kv.second;
        }
      }

      // Add new landmark
      landmarks.emplace(track_id, lm);
      new_track_ids.emplace_back(track_id);
    }
  }

  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const FrameCamId &fcid0,
                                       const FrameCamId &fcid1,
                                       const Calibration &calib_cam,
                                       const Corners &feature_corners,
                                       const FeatureTracks &feature_tracks,
                                       Cameras &cameras, Landmarks &landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(fcid0.frame_id == fcid1.frame_id && fcid0.cam_id != fcid1.cam_id)) {
    std::cerr << "Images " << fcid0 << " and " << fcid1
              << " don't form a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)
  Camera cam0{calib_cam.T_i_c[fcid0.cam_id]};
  Camera cam1{calib_cam.T_i_c[fcid1.cam_id]};
  //  cam0.T_w_c =  calib_cam.T_i_c[fcid0.cam_id];
  //  Camera cam1;
  //  cam1.T_w_c = calib_cam.T_i_c[fcid1.cam_id];
  cameras.emplace(fcid0, cam0);
  cameras.emplace(fcid1, cam1);
  //  cameras.insert(std::make_pair(fcid0, cam0));
  //  cameras.insert(std::make_pair(fcid1, cam1));

  // Add new landmarks
  add_new_landmarks_between_cams(fcid0, fcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const FrameCamId &fcid, const std::vector<TrackId> &shared_track_ids,
    const Calibration &calib_cam, const Corners &feature_corners,
    const FeatureTracks &feature_tracks, const Landmarks &landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d &T_w_c, std::vector<TrackId> &inlier_track_ids) {
  inlier_track_ids.clear();

  // TODO SHEET 4: Localize a new image in a given map
  UNUSED(fcid);
  UNUSED(shared_track_ids);
  UNUSED(calib_cam);
  UNUSED(feature_corners);
  UNUSED(feature_tracks);
  UNUSED(landmarks);
  UNUSED(T_w_c);
  UNUSED(reprojection_error_pnp_inlier_threshold_pixel);
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(const Corners &feature_corners,
                       const BundleAdjustmentOptions &options,
                       const std::set<FrameCamId> &fixed_cameras,
                       Calibration &calib_cam, Cameras &cameras,
                       Landmarks &landmarks) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem
  UNUSED(feature_corners);
  UNUSED(options);
  UNUSED(fixed_cameras);
  UNUSED(calib_cam);
  UNUSED(cameras);
  UNUSED(landmarks);

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

} // namespace visnav
