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

#include <pangolin/image/managed_image.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <visnav/common_types.h>
#include <visnav/keypoints.h>

namespace visnav {

/*
 * Add new keypoints in a grids. If any cell flows dieout; Try to create new
 * flow there
 */
bool add_flows_on_grids(const pangolin::ManagedImage<uint8_t>& img_raw,
                        KeypointsData& kd, int num_features,
                        const int& num_bin_x, const int& num_bin_y,
                        double filled_cells_thresh) {
  const auto& width = img_raw.w;
  const auto& height = img_raw.h;
  double cell_width = (double)(width + 1) / num_bin_x;
  double cell_height = (double)(height + 1) / num_bin_y;

  auto total_cells = num_bin_x * num_bin_y;
  std::set<std::pair<int, int>> flow_cell_exist;

  //  // Examine flow count for each cell
  for (const auto& kp : kd.corners) {
    /*
     * Becareful when converting double to int here; under flow could happen.
     * Check close to 0 convert here
     */
    double kpx = kp[0];
    double kpy = kp[1];
    if (kpx < 1.0) kpx = 1.0;
    if (kpy < 1.0) kpy = 1.0;
    if (kpx > width - 1.0) kpx = width - 1.0;
    if (kpy > height - 1.0) kpy = height - 1.0;

    int bin_x = (int)(kpx / cell_width);
    int bin_y = (int)(kpy / cell_height);
    if (flow_cell_exist.count(std::make_pair(bin_x, bin_y)) == 0)
      flow_cell_exist.insert(std::make_pair(bin_x, bin_y));
  }

  const auto& n_fill_cells = flow_cell_exist.size();
  std::cout << n_fill_cells << " cells filled over " << total_cells << "!\n";

  if (n_fill_cells >= total_cells * filled_cells_thresh) {
    return false;
  } else {
    size_t num_keypoints_added = 0;
    KeypointsData new_kd;
    detectKeypoints(img_raw, new_kd, num_features);
    std::cout << "Detected " << new_kd.corners.size() << " kp!\n";

    // Add only for empty regions
    for (const auto& kp : new_kd.corners) {
      double kpx = kp[0];
      double kpy = kp[1];
      if (kpx < 1.0) kpx = 1.0;
      if (kpy < 1.0) kpy = 1.0;
      if (kpx > width - 1.0) kpx = width - 1.0;
      if (kpy > height - 1.0) kpy = height - 1.0;

      int bin_x = (int)(kpx / cell_width);
      int bin_y = (int)(kpy / cell_height);

      if (flow_cell_exist.count(std::make_pair(bin_x, bin_y)) == 0) {
        kd.corners.emplace_back(kp.x(), kp.y());
        num_keypoints_added++;
      }
    }

    //     Check num addes
    if (num_keypoints_added == 0) {
      std::cout << "ADD FLOWS: Failed to add new keypoints\n";
      return false;
    } else {
      std::cout << "ADD FLOWS: Added " << num_keypoints_added
                << " new keypoints\n";
      return true;
    }
  }
}

}  // namespace visnav
