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
#include <vector>

#include <thread>

#include <sophus/se2.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <visnav/common_types.h>
#include <visnav/calibration.h>
#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/optical_flow_utils.h>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

/// Additional packages for Optical Flow
#include <visnav/image/image.h>
#include <visnav/image/image_pyr.h>
#include <visnav/patch.h>
#include <sophus/se2.hpp>

const int OF_TRACK_MAX_ITERATIONS = 20;

namespace visnav {
typedef OpticalFlowPatch<float, Pattern50<float>> PatchT;

/**
 * @brief initialize_transforms
 * @param kdl
 * @param transforms
 */
void initialize_transforms(
    const KeypointsData& kdl,
    std::unordered_map<FeatureId, Eigen::AffineCompact2f>& transforms) {
  // Init transformation to track
  for (size_t i = 0; i < kdl.corners.size(); i++) {
    Eigen::AffineCompact2f tf;
    tf.setIdentity();
    tf.translation() = kdl.corners[i].cast<float>();
    transforms.emplace(i, tf);
  }
}

/**
 * @brief trackPointAtLevel
 * @param img_2
 * @param dp
 * @param transform
 * @return
 */
inline bool trackPointAtLevel(const visnav::Image<const uint8_t>& img_2,
                              const PatchT& dp,
                              Eigen::AffineCompact2f& transform) {
  bool patch_valid = true;
  int max_iterations = OF_TRACK_MAX_ITERATIONS;

  for (int iteration = 0; patch_valid && iteration < max_iterations;
       iteration++) {
    typename PatchT::VectorP res;

    typename PatchT::Matrix2P transformed_pat =
        transform.linear().matrix() * dp.pattern2;
    transformed_pat.colwise() += transform.translation();

    bool valid = dp.residual(img_2, transformed_pat, res);

    if (valid) {
      typename PatchT::Vector3 inc = -dp.H_se2_inv_J_se2_T * res;

      transform *= Sophus::SE2f::exp(inc).matrix();

      const int filter_margin = 3;

      if (!img_2.InBounds(transform.translation(), filter_margin))
        patch_valid = false;
    } else {
      patch_valid = false;
    }
    // std::cout << "PATCH VALID: " << patch_valid << std::endl;
  }

  return patch_valid;
}

/**
 * @brief trackPoint
 * @param old_pyr
 * @param pyr
 * @param num_levels
 * @param old_transform
 * @param transform
 * @return
 */
inline bool trackPoint(const visnav::ManagedImagePyr<uint8_t>& old_pyr,
                       const visnav::ManagedImagePyr<uint8_t>& pyr,
                       const size_t& num_levels,
                       const Eigen::AffineCompact2f& old_transform,
                       Eigen::AffineCompact2f& transform) {
  bool patch_valid = true;

  transform.linear().setIdentity();

  for (int level = num_levels; level >= 0 && patch_valid; level--) {
    const float scale = 1 << level;  // bit shift

    transform.translation() /= scale;

    PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

    // Perform tracking on current level
    patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);

    transform.translation() *= scale;
  }

  transform.linear() = old_transform.linear() * transform.linear();

  return patch_valid;
}

/**
 * @brief trackPoints
 * @param kd
 * @param old_pyr
 * @param pyr
 * @param num_levels
 * @param distance_threshold
 * @param transforms
 */
void trackPoints(
    const KeypointsData& kd, const visnav::ManagedImagePyr<uint8_t>& old_pyr,
    const visnav::ManagedImagePyr<uint8_t>& pyr, const size_t& num_levels,
    const double& distance_threshold,
    std::unordered_map<FeatureId, Eigen::AffineCompact2f>& transforms) {
  int num_points = kd.corners.size();

  auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      Eigen::AffineCompact2f transform_1 = transforms[i];

      Eigen::AffineCompact2f transform_2 = transform_1;

      transform_2.linear().setIdentity();
      bool valid =
          trackPoint(old_pyr, pyr, num_levels, transform_1, transform_2);
      bool flag = false;
      transform_2.linear() = transform_1.linear() * transform_2.linear();

      if (valid) {
        Eigen::AffineCompact2f transform_1_recovered = transform_2;

        transform_1_recovered.linear().setIdentity();
        valid = trackPoint(pyr, old_pyr, num_levels, transform_2,
                           transform_1_recovered);
        transform_1_recovered.linear() =
            transform_2.linear() * transform_1_recovered.linear();

        if (valid) {
          float dist2 =
              (transform_1.translation() - transform_1_recovered.translation())
                  .squaredNorm();

          if (dist2 < distance_threshold) {
            transforms[i] = transform_2;
            flag = true;
          }
        }
      }

      // Delete transform not under threshold
      if (!flag) {
        transforms.erase(i);
      }
    }
  };

  tbb::blocked_range<size_t> range(0, num_points);

  tbb::parallel_for(range, compute_func);
}

/**
 * @brief match_optical: Fill keypoints and matches with transform
 * @param kdr keypoints in target frame to be filled
 * @param transforms 2D affine transforms
 * @param matches matches from source to target frame
 */
void match_optical(
    KeypointsData& kdr,
    const std::unordered_map<FeatureId, Eigen::AffineCompact2f>& transforms,
    std::vector<std::pair<int, int>>& matches) {
  kdr.corners.clear();
  matches.clear();
  int i = 0;
  int j = 0;
  std::unordered_map<FeatureId, TrackId> updated_tracks;
  for (const auto& t : transforms) {
    kdr.corners.push_back(t.second.translation().cast<double>());
    matches.emplace_back(t.first, i);
    i++;
  }
}

/**
 * @brief matchOpticalFlow: Use Optical flows to find matches and keypoints in
 * target frame from a source frame
 * @param img_src
 * @param img_dest
 * @param kd_src
 * @param kd_dest
 * @param md
 * @param pyramid_level
 * @param distance_threshold
 * @param transforms
 */
void matchOpticalFlow(const visnav::ManagedImage<uint8_t>& img_src,
                      const visnav::ManagedImage<uint8_t>& img_dest,
                      const KeypointsData& kd_src, KeypointsData& kd_dest,
                      MatchData& md, int pyramid_level,
                      double distance_threshold) {
  // Init transform
  std::unordered_map<FeatureId, Eigen::AffineCompact2f> transforms;
  initialize_transforms(kd_src, transforms);

  // Create Image pyramid for two image pairs
  visnav::ManagedImagePyr<uint8_t> img_src_pyr, img_dest_pyr;
  img_src_pyr.setFromImage(img_src, pyramid_level);
  img_dest_pyr.setFromImage(img_dest, pyramid_level);

  // Calculate transformation
  trackPoints(kd_src, img_src_pyr, img_dest_pyr, pyramid_level,
              distance_threshold, transforms);

  // Fill keypoints in target frame and maches
  match_optical(kd_dest, transforms, md.matches);
}

void optical_flows(const pangolin::ManagedImage<uint8_t>& img_src,
                   const pangolin::ManagedImage<uint8_t>& img_dest,
                   const KeypointsData& kd_src, KeypointsData& kd_dest,
                   MatchData& md, const int& pyramid_level,
                   const double& distance_threshold, const bool& run_basalt) {
  if (!run_basalt) {
    // Run Lucas-Kanade method
    optical_flows_opencv(img_src, img_dest, kd_src, kd_dest, md, pyramid_level,
                         distance_threshold);
  } else {
    // Run Basalt method of Usenko
    // Covert to Basalt Image
    visnav::ManagedImage<uint8_t> img_src_v, img_dest_v;
    img_src_v.CopyFrom(visnav::Image<uint8_t>(img_src.ptr, img_src.w, img_src.h,
                                              img_src.pitch));
    img_dest_v.CopyFrom(visnav::Image<uint8_t>(img_dest.ptr, img_dest.w,
                                               img_dest.h, img_dest.pitch));

    // Run Optical flow
    matchOpticalFlow(img_src_v, img_dest_v, kd_src, kd_dest, md, pyramid_level,
                     distance_threshold);
  }
}

}  // namespace visnav
