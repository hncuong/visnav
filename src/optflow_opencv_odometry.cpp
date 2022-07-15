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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <random>
#include <string>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <opencv2/core.hpp>

#include <CLI/CLI.hpp>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/matching_utils.h>
#include <visnav/vo_utils.h>
#include <visnav/optical_flow_utils.h>
#include <visnav/of_grid.h>

#include <visnav/gui_helper.h>
#include <visnav/tracks.h>

#include <visnav/serialization.h>

// Header files for custom image and image pyramid
#include <visnav/image/image_pyr.h>

using namespace visnav;

///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////

void draw_image_overlay(pangolin::View& v, size_t view_id);
void change_display_to_image(const FrameCamId& fcid);
void draw_scene();
void load_data(const std::string& path, const std::string& calib_path);
bool next_step();
void optimize();
void compute_projections();
void save_trajectory();
void save_all_trajectory();

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 2;

///////////////////////////////////////////////////////////////////////////////
/// Variables
///////////////////////////////////////////////////////////////////////////////

int current_frame = 0;
Sophus::SE3d current_pose;
bool take_keyframe = true;
TrackId next_landmark_id = 0;

std::atomic<bool> opt_running{false};
std::atomic<bool> opt_finished{false};

std::set<FrameId> kf_frames;

// std::shared_ptr<std::thread> opt_thread;

/// For Optical flows
int num_consecutive_regular_frames = 0;
int max_consecutive_regular_frames = 20;

/// intrinsic calibration
Calibration calib_cam;
Calibration calib_cam_opt;

/// loaded images
tbb::concurrent_unordered_map<FrameCamId, std::string> images;

/// timestamps for all stereo pairs
std::vector<Timestamp> timestamps;

/// detected feature locations and descriptors
Corners feature_corners;

/// pairwise feature matches
Matches feature_matches;

/// camera poses in the current map
Cameras cameras;
// Camera poses
Cameras all_cameras;

/// Save all poses for all timestamp
Cameras all_poses;

/// copy of cameras for optimization in parallel thread
Cameras cameras_opt;

/// landmark positions and feature observations in current map
// Flows landmarks;

/// copy of landmarks for optimization in parallel thread
Flows landmarks_opt;

/// landmark positions that were removed from the current map
Flows old_landmarks;

/// cashed info on reprojected landmarks; recomputed every time time from
/// cameras, landmarks, and feature_tracks; used for visualization and
/// determining outliers; indexed by images
ImageProjections image_projections;

//////////////////////////////////////////////
/// Optical Flows specific
/// Keypoints data of last frame
KeypointsData kd_last;
/// Last image
pangolin::ManagedImage<uint8_t> imgl_last;

/// Flows
Flows flows;
Flows all_flows;

/// Next flow id to add to all flows
TrackId next_flow_id = 0;

/// Flows to show
std::unordered_map<TrackId, Eigen::Vector3f> flows_to_show;

/// For generating random colors
cv::RNG rng;
/// Forward-backward back project distance threshold for optical flows
/// Square norm
double backproject_distance_threshold2 = 0.04;

/// Image Pyramid for leftframe, rightframe, nextframe
visnav::ManagedImagePyr<uint8_t> left_pyr, right_pyr, next_pyr, imgl_last_pyr;

/// Map storing FeatureID and its corresponding TrackID
std::unordered_map<FeatureId, TrackId> propagate_tracks;

/// Last match feature to Track
LandmarkMatchData last_md_featureToTrack;

/// Pyramid level
int pyramid_level = 3;

/// Cells to store keypoints
int num_bin_x = 5;
int num_bin_y = 5;
size_t last_frame_num_filled_cells = 25;

/// Num keypoints added to get new keyframes
int new_kf_num_new_keypoints = 80;
int new_kps = 0;

/// TIME Measurements
double total_t1 = 0.;
double total_t2 = 0.;
double total_t3 = 0.;
double total_t4 = 0.;
double total_t5 = 0.;
double total_t6 = 0.;

///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

// The following GUI elements can be enabled / disabled from the main panel by
// switching the prefix from "ui" to "hidden" or vice verca. This way you can
// show only the elements you need / want for development.

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, true);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, true);
pangolin::Var<bool> show_detected("ui.show_detected", true, true);
pangolin::Var<bool> show_matches("ui.show_matches", true, true);
pangolin::Var<bool> show_inliers("ui.show_inliers", true, true);
pangolin::Var<bool> show_reprojections("ui.show_reprojections", true, true);
pangolin::Var<bool> show_outlier_observations("ui.show_outlier_obs", false,
                                              true);
pangolin::Var<bool> show_ids("ui.show_ids", false, true);
pangolin::Var<bool> show_epipolar("hidden.show_epipolar", false, true);
pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, true);
pangolin::Var<bool> show_old_points3d("hidden.show_old_points3d", true, true);

//////////////////////////////////////////////
/// For Optical flows options
pangolin::Var<bool> show_flows("ui.show_flows", true, true);
pangolin::Var<int> num_flows_to_draw("hidden.num_flows_to_draw", 500, 1, 1000);
pangolin::Var<int> max_track_length("hidden.max_track_length", 20, 10, 200);
pangolin::Var<double> line_width("hidden.line_width", 2.0, 1.0, 3.0);

//////////////////////////////////////////////
/// Feature extraction and matching options

pangolin::Var<int> num_features_per_image("hidden.num_features", 1500, 10,
                                          5000);
pangolin::Var<bool> rotate_features("hidden.rotate_features", true, true);
pangolin::Var<int> feature_match_max_dist("hidden.match_max_dist", 70, 1, 255);
pangolin::Var<double> feature_match_test_next_best("hidden.match_next_best",
                                                   1.2, 1, 4);

pangolin::Var<double> match_max_dist_2d("hidden.match_max_dist_2d", 20.0, 1.0,
                                        50);

pangolin::Var<int> new_kf_min_inliers("hidden.new_kf_min_inliers", 80, 1, 200);

pangolin::Var<int> max_num_kfs("hidden.max_num_kfs", 10, 5, 200);

pangolin::Var<double> cam_z_threshold("hidden.cam_z_threshold", 0.1, 1.0, 0.0);

//////////////////////////////////////////////
/// Adding cameras and landmarks options

pangolin::Var<double> reprojection_error_pnp_inlier_threshold_pixel(
    "hidden.pnp_inlier_thresh", 3.0, 0.1, 10);

//////////////////////////////////////////////
/// Bundle Adjustment Options

pangolin::Var<bool> ba_optimize_intrinsics("hidden.ba_opt_intrinsics", false,
                                           true);
pangolin::Var<int> ba_verbose("hidden.ba_verbose", 1, 0, 2);

pangolin::Var<double> reprojection_error_huber_pixel("hidden.ba_huber_width",
                                                     1.0, 0.1, 10);

///////////////////////////////////////////////////////////////////////////////
/// GUI buttons
///////////////////////////////////////////////////////////////////////////////

// if you enable this, next_step is called repeatedly until completion
pangolin::Var<bool> continue_next("ui.continue_next", false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

Button save_trajectory_btn("ui.save_trajectory", &save_trajectory);

Button save_all_trajectory_btn("ui.save_all_trajectory", &save_all_trajectory);

///////////////////////////////////////////////////////////////////////////////
/// GUI and Boilerplate Implementation
///////////////////////////////////////////////////////////////////////////////

// Parse parameters, load data, and create GUI window and event loop (or
// process everything in non-gui mode).
int main(int argc, char** argv) {
  bool show_gui = true;
  std::string dataset_path = "data/V1_01_easy/mav0";
  std::string cam_calib = "opt_calib.json";

  /// OF Start frame for debugging
  int start_frame_idx = 0;
  double backproject_distance_threshold_in_pixels = 0.2;
  int kf_frequency = 20;
  int df_pyramid_lv = 3;

  CLI::App app{"Visual odometry."};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path,
                 "Dataset path. Default: " + dataset_path);
  app.add_option("--cam-calib", cam_calib,
                 "Path to camera calibration. Default: " + cam_calib);

  /// OF Odometry options
  app.add_option(
      "--start-frame", start_frame_idx,
      "Start frame index. Default: " + std::to_string(start_frame_idx));
  app.add_option("--dist-thresh", backproject_distance_threshold_in_pixels,
                 "Forward-backward project distance threshold for Optical "
                 "Flow. Default: " +
                     std::to_string(backproject_distance_threshold_in_pixels));
  app.add_option("--kf_frequency", kf_frequency,
                 "Number of max consecutive regular frames: " +
                     std::to_string(kf_frequency));
  app.add_option("--pyramid-lv", df_pyramid_lv,
                 "Number of pyramid lv " + std::to_string(df_pyramid_lv));
  app.add_option("--nbin-x", num_bin_x,
                 "Number bins on x axis " + std::to_string(num_bin_x));
  app.add_option("--nbin-y", num_bin_y,
                 "Number bins on x axis " + std::to_string(num_bin_y));
  app.add_option("--new-kps-kf", new_kf_num_new_keypoints,
                 "Number of accum new kps to take new kf " +
                     std::to_string(new_kf_num_new_keypoints));

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  load_data(dataset_path, cam_calib);
  current_frame = start_frame_idx;
  backproject_distance_threshold2 = backproject_distance_threshold_in_pixels *
                                    backproject_distance_threshold_in_pixels;
  max_consecutive_regular_frames = kf_frequency;
  pyramid_level = df_pyramid_lv;
  last_frame_num_filled_cells = num_bin_x * num_bin_y;
  std::cout << "START with frame " << current_frame
            << " and backproject threshold " << backproject_distance_threshold2
            << " and KF frequency " << max_consecutive_regular_frames
            << " and pyramid lv " << pyramid_level << " and num bins x, y "
            << num_bin_x << " " << num_bin_y << " last frame filled "
            << last_frame_num_filled_cells << " - new kf accumulate new kps "
            << new_kf_num_new_keypoints << "\n";

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View& main_view =
        pangolin::Display("main")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View& img_view_display =
        pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // extra options panel
    pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
        pangolin::Attach::Pix(2 * UI_WIDTH));
    ui_show_hidden.Meta().gui_changed = true;

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < NUM_CAMS) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::Display("scene")
            .SetAspect(-640 / 480.0)
            .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (ui_show_hidden.GuiChanged()) {
        hidden_panel.Show(ui_show_hidden);
        const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
        main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
      }

      display3D.Activate(camera);
      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // light gray background

      draw_scene();

      img_view_display.Activate();

      if (lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (show_frame1.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame1, 0));
          change_display_to_image(FrameCamId(show_frame1, 1));
        } else if (show_frame2.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame2, 0));
          change_display_to_image(FrameCamId(show_frame2, 1));
        }
      }

      if (show_frame1.GuiChanged() || show_cam1.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame1);
        auto cam_id = static_cast<CamId>(show_cam1);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[0]->SetImage(img);
        } else {
          img_view[0]->Clear();
        }
      }

      if (show_frame2.GuiChanged() || show_cam2.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame2);
        auto cam_id = static_cast<CamId>(show_cam2);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_BYTE;
          fmt.scalable_internal_format = GL_LUMINANCE8;

          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[1]->SetImage(img);
        } else {
          img_view[1]->Clear();
        }
      }

      pangolin::FinishFrame();

      if (continue_next) {
        // stop if there is nothing left to do
        continue_next = next_step();
      } else {
        // if the gui is just idling, make sure we don't burn too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
  } else {
    // non-gui mode: Process all frames, then exit
    while (next_step()) {
      // nop
    }
  }

  /// Print time measure
  std::cout << "\nTIME MEASUREMENTS: " << total_t1 << " " << total_t2 << " "
            << total_t3 << " " << total_t4 << " " << total_t5 << " " << total_t6
            << " \n";

  /// Save cemeras trajectory for evaluation
  save_trajectory();
  save_all_trajectory();

  return 0;
}

// Visualize features and related info on top of the image views
void draw_image_overlay(pangolin::View& v, size_t view_id) {
  UNUSED(v);

  auto frame_id =
      static_cast<FrameId>(view_id == 0 ? show_frame1 : show_frame2);
  auto cam_id = static_cast<CamId>(view_id == 0 ? show_cam1 : show_cam2);

  FrameCamId fcid(frame_id, cam_id);

  float text_row = 20;

  if (show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);  // red
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (feature_corners.find(fcid) != feature_corners.end()) {
      const KeypointsData& cr = feature_corners.at(fcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        //        double angle = cr.corner_angles[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(5, text_row);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  /*
   * Draw some flows from start to current frame
   * TODO Draw consistent set of flows
   */
  //  show_flows = true;
  if (show_flows) {
    glColor3f(0.0, 0.0, 0.5);  // navy
    glLineWidth(line_width);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    /// TODO Change to all flows for visualization
    /// Maybe no 3d correspondent yet
    /// Merge later if needed
    size_t num_flows = all_flows.size();
    /// Check alive flows and draw circle around it
    std::vector<TrackId> aliveFlows;
    int total_flows_length = 0;

    for (const auto& kv : all_flows) {
      if (kv.second.alive) {
        aliveFlows.emplace_back(kv.first);
        total_flows_length += kv.second.flow.size();

        // Draw circle
        if (kv.second.flow.count(fcid) > 0 && feature_corners.count(fcid) > 0) {
          const auto& featureId = kv.second.flow.at(fcid);
          const KeypointsData& cr = feature_corners.at(fcid);
          Eigen::Vector2d c = cr.corners[featureId];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);
        }
      }
    }

    int mean_flow_length = 0;
    if (aliveFlows.size() > 0)
      mean_flow_length = total_flows_length / aliveFlows.size();

    pangolin::GlFont::I()
        .Text("Current frame has %d flows, avg length %d, %d landmarks",
              aliveFlows.size(), mean_flow_length, flows.size())
        .Draw(5, text_row);
    text_row += 20;

    /*
     * Ensure number of flows to draw.
     * Discard die out flows
     * Add new flows from alive flows if possible
     */
    // Check for liveness
    std::vector<TrackId> flowsToDiscard;
    for (const auto& flow_id : flows_to_show) {
      if (all_flows.count(flow_id.first) == 0) {
        // If flow disappear
        flowsToDiscard.emplace_back(flow_id.first);
      } else {
        // Or not alive anymore
        if (!all_flows.at(flow_id.first).alive)
          //// if (all_flows.at(flow_id.first).flow.count(fcid) == 0) // Wrong
          // since there are left and right frame
          flowsToDiscard.emplace_back(flow_id.first);
      }
    }
    // Discard
    for (const auto& flow_id : flowsToDiscard) {
      flows_to_show.erase(flow_id);
    }

    // Fill it until it reach the numbers needed
    if (flows_to_show.size() < num_flows_to_draw) {
      auto num_flows_to_add = num_flows_to_draw - flows_to_show.size();

      // Try to add flows then pick random color associated
      std::vector<TrackId> flowsToAdd;

      // Pick random flows
      // First choose alive flows that not show yet
      std::vector<TrackId> flowsToPick;
      for (const auto& kv : all_flows) {
        if (flows_to_show.count(kv.first) == 0 && kv.second.alive) {
          flowsToPick.emplace_back(kv.first);
        }
      }

      // Now try to pick new flows
      if (num_flows_to_add >= flowsToPick.size()) {
        // Pick all
        for (const auto& flow_id : flowsToPick)
          flowsToAdd.emplace_back(flow_id);
      } else {
        // Random selection.
        // NOTE: this function is C++17
        // FIXME Notice in case it doesn't work here
        std::sample(flowsToPick.begin(), flowsToPick.end(),
                    std::back_inserter(flowsToAdd), num_flows_to_add,
                    std::mt19937{std::random_device{}()});
      }

      // Add to show along with a random color
      for (const auto& flow_id : flowsToAdd) {
        // Pick random color
        float r = rng.uniform(0, 256) / 255.f;
        float g = rng.uniform(0, 256) / 255.f;
        float b = rng.uniform(0, 256) / 255.f;
        Eigen::Vector3f color_code(r, g, b);

        flows_to_show.emplace(flow_id, color_code);
      }
    }

    /*
     * Draw selected flows's trajectory.
     */
    /// Draw flows of current fcid to
    size_t flow_draw_cnt = 0;
    std::vector<TrackId> flows_to_discard;

    for (const auto& kv : flows_to_show) {
      const auto& trackId = kv.first;
      const auto& color_code = kv.second;
      // TODO Change gl color here
      glColor3f(color_code[0], color_code[1], color_code[2]);

      // Only select some flows
      if (all_flows.count(trackId) == 0 || !all_flows.at(trackId).alive) {
        flows_to_discard.emplace_back(trackId);
      } else {
        const auto& flow = all_flows.at(trackId);

        // If flow exist in the frame
        // Draw the point in the current frame
        // And draw flow back to the start
        //        int max_track_length = 20;
        int track_length = 0;

        if (flow.flow.count(fcid) > 0 && feature_corners.count(fcid) > 0) {
          const auto& featureId = flow.flow.at(fcid);

          const KeypointsData& cr = feature_corners.at(fcid);
          Eigen::Vector2d c = cr.corners[featureId];
          //          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", trackId).Draw(c[0], c[1]);
          }

          // Draw line to previous if exists
          FrameCamId previous_fcid(fcid.frame_id - 1, fcid.cam_id);
          while (flow.flow.count(previous_fcid) > 0 &&
                 feature_corners.count(previous_fcid) > 0) {
            // Get previous frame featureId of the flow
            const auto& prev_featureId = flow.flow.at(previous_fcid);
            Eigen::Vector2d prev_c =
                feature_corners.at(previous_fcid).corners[prev_featureId];

            // Draw to previous
            pangolin::glDrawLine(c, prev_c);

            // Check maximum length reach
            track_length++;
            if (track_length >= max_track_length) break;

            // Update previous frame keypoints and fcid
            c = prev_c;
            previous_fcid =
                FrameCamId(previous_fcid.frame_id - 1, previous_fcid.cam_id);
          }

          flow_draw_cnt++;
        }
      }
    }

    // Discard die out flows
    for (const auto& trackId : flows_to_discard) flows_to_show.erase(trackId);
  }

  if (show_matches || show_inliers) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);  // blue
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && show_matches) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.matches.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.matches[i].first
                                  : it->second.matches[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          //          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d matches", it->second.matches.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }

    glColor3f(0.0, 1.0, 0.0);  // green

    if (idx >= 0 && show_inliers) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.inliers.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.inliers[i].first
                                  : it->second.inliers[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          //          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d inliers", it->second.inliers.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }
  }

  if (show_reprojections) {
    if (image_projections.count(fcid) > 0) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);  // red
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      const size_t num_points = image_projections.at(fcid).obs.size();
      double error_sum = 0;
      size_t num_outliers = 0;

      // count up and draw all inlier projections
      for (const auto& lm_proj : image_projections.at(fcid).obs) {
        error_sum += lm_proj->reprojection_error;

        if (lm_proj->outlier_flags != OutlierNone) {
          // outlier point
          glColor3f(1.0, 0.0, 0.0);  // red
          ++num_outliers;
        } else if (lm_proj->reprojection_error >
                   reprojection_error_huber_pixel) {
          // close to outlier point
          glColor3f(1.0, 0.5, 0.0);  // orange
        } else {
          // clear inlier point
          glColor3f(1.0, 1.0, 0.0);  // yellow
        }
        pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
        pangolin::glDrawLine(lm_proj->point_measured,
                             lm_proj->point_reprojected);
      }

      // only draw outlier projections
      if (show_outlier_observations) {
        glColor3f(1.0, 0.0, 0.0);  // red
        for (const auto& lm_proj : image_projections.at(fcid).outlier_obs) {
          pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
          pangolin::glDrawLine(lm_proj->point_measured,
                               lm_proj->point_reprojected);
        }
      }

      glColor3f(1.0, 0.0, 0.0);  // red
      pangolin::GlFont::I()
          .Text("Average repr. error (%u points, %u new outliers): %.2f",
                num_points, num_outliers, error_sum / num_points)
          .Draw(5, text_row);
      text_row += 20;
    }
  }

  if (show_epipolar) {
    glLineWidth(1.0);
    glColor3f(0.0, 1.0, 1.0);  // bright teal
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && it->second.inliers.size() > 20) {
      Sophus::SE3d T_this_other =
          idx == 0 ? it->second.T_i_j : it->second.T_i_j.inverse();

      Eigen::Vector3d p0 = T_this_other.translation().normalized();

      int line_id = 0;
      for (double i = -M_PI_2 / 2; i <= M_PI_2 / 2; i += 0.05) {
        Eigen::Vector3d p1(0, sin(i), cos(i));

        if (idx == 0) p1 = it->second.T_i_j * p1;

        p1.normalize();

        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            line;
        for (double j = -1; j <= 1; j += 0.001) {
          line.emplace_back(calib_cam.intrinsics[cam_id]->project(
              p0 * j + (1 - std::abs(j)) * p1));
        }

        Eigen::Vector2d c = calib_cam.intrinsics[cam_id]->project(p1);
        pangolin::GlFont::I().Text("%d", line_id).Draw(c[0], c[1]);
        line_id++;

        pangolin::glDrawLineStrip(line);
      }
    }
  }
}

// Update the image views to a given image id
void change_display_to_image(const FrameCamId& fcid) {
  if (0 == fcid.cam_id) {
    // left view
    show_cam1 = 0;
    show_frame1 = fcid.frame_id;
    show_cam1.Meta().gui_changed = true;
    show_frame1.Meta().gui_changed = true;
  } else {
    // right view
    show_cam2 = fcid.cam_id;
    show_frame2 = fcid.frame_id;
    show_cam2.Meta().gui_changed = true;
    show_frame2.Meta().gui_changed = true;
  }
}

// Render the 3D viewer scene of cameras and points
void draw_scene() {
  const FrameCamId fcid1(show_frame1, show_cam1);
  const FrameCamId fcid2(show_frame2, show_cam2);

  const u_int8_t color_camera_current[3]{255, 0, 0};         // red
  const u_int8_t color_camera_left[3]{0, 125, 0};            // dark green
  const u_int8_t color_camera_right[3]{0, 0, 125};           // dark blue
  const u_int8_t color_points[3]{0, 0, 0};                   // black
  const u_int8_t color_old_points[3]{170, 170, 170};         // gray
  const u_int8_t color_selected_left[3]{0, 250, 0};          // green
  const u_int8_t color_selected_right[3]{0, 0, 250};         // blue
  const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
  const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple

  // render cameras
  if (show_cameras3d) {
    for (const auto& cam : cameras) {
      if (cam.first == fcid1) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_left,
                      0.1f);
      } else if (cam.first == fcid2) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_right,
                      0.1f);
      } else if (cam.first.cam_id == 0) {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_left, 0.1f);
      } else {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_right,
                      0.1f);
      }
    }
    render_camera(current_pose.matrix(), 2.0f, color_camera_current, 0.1f);
  }

  // render trajectory
  // Draw the trajectory of left cameras
  bool show_trajectory = true;
  FrameCamId fcid_cur(show_frame1, 0);
  FrameCamId fcid_prev(fcid_cur.frame_id - 1, 0);
  glLineWidth(2.0);
  glColor3f(1.0, 0.0, 0.0);  // red
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (show_trajectory) {
    // Draw line to the last frame
    if (all_poses.count(fcid_cur) > 0) {
      while (all_poses.count(fcid_prev) > 0) {
        // T_w_c to location
        const auto& cur_Twc = all_poses.at(fcid_cur).T_w_c;
        const auto& prev_Twc = all_poses.at(fcid_prev).T_w_c;

        Eigen::Vector3d camera_p3d = Eigen::Vector3d::Zero();
        Eigen::Vector3d cur_p3d = cur_Twc * camera_p3d;
        Eigen::Vector3d prev_p3d = prev_Twc * camera_p3d;

        std::vector<Eigen::Vector3d> vertices;
        vertices.emplace_back(cur_p3d);
        vertices.emplace_back(prev_p3d);
        pangolin::glDrawLines(vertices);

        // Update cur and last
        fcid_cur = FrameCamId(fcid_prev.frame_id, 0);
        fcid_prev = FrameCamId(fcid_cur.frame_id - 1, 0);
      }
    }
  }

  // render points
  if (show_points3d && flows.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (const auto& kv_lm : flows) {
      const bool in_cam_1 = kv_lm.second.obs.count(fcid1) > 0;
      const bool in_cam_2 = kv_lm.second.obs.count(fcid2) > 0;

      const bool outlier_in_cam_1 = kv_lm.second.outlier_obs.count(fcid1) > 0;
      const bool outlier_in_cam_2 = kv_lm.second.outlier_obs.count(fcid2) > 0;

      if (in_cam_1 && in_cam_2) {
        glColor3ubv(color_selected_both);
      } else if (in_cam_1) {
        glColor3ubv(color_selected_left);
      } else if (in_cam_2) {
        glColor3ubv(color_selected_right);
      } else if (outlier_in_cam_1 || outlier_in_cam_2) {
        glColor3ubv(color_outlier_observation);
      } else {
        glColor3ubv(color_points);
      }

      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }

  // render points
  if (show_old_points3d && old_landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);

    for (const auto& kv_lm : old_landmarks) {
      glColor3ubv(color_old_points);
      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }
}

// Load images, calibration, and features / matches if available
void load_data(const std::string& dataset_path, const std::string& calib_path) {
  const std::string timestams_path = dataset_path + "/cam0/data.csv";

  {
    std::ifstream times(timestams_path);

    int id = 0;

    while (times) {
      std::string line;
      std::getline(times, line);

      if (line.size() < 20 || line[0] == '#' || id > 2700) continue;

      {
        std::string timestamp_str = line.substr(0, 19);
        std::istringstream ss(timestamp_str);
        Timestamp timestamp;
        ss >> timestamp;
        timestamps.push_back(timestamp);
      }

      std::string img_name = line.substr(20, line.size() - 21);

      for (int i = 0; i < NUM_CAMS; i++) {
        FrameCamId fcid(id, i);

        std::stringstream ss;
        ss << dataset_path << "/cam" << i << "/data/" << img_name;

        images[fcid] = ss.str();
      }

      id++;
    }

    std::cerr << "Loaded " << id << " image pairs" << std::endl;
  }

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_cam);
      std::cout << "Loaded camera from " << calib_path << " with models ";
      for (const auto& cam : calib_cam.intrinsics) {
        std::cout << cam->name() << " ";
      }
      std::cout << std::endl;
    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  show_frame1.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame1.Meta().gui_changed = true;
  show_frame2.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame2.Meta().gui_changed = true;
}

///////////////////////////////////////////////////////////////////////////////
/// Here the algorithmically interesting implementation begins
///////////////////////////////////////////////////////////////////////////////

// Execute next step in the overall odometry pipeline. Call this repeatedly
// until it returns false for automatic execution.
bool next_step() {
  auto start = std::chrono::high_resolution_clock::now();
  if (current_frame >= int(images.size()) / NUM_CAMS) return false;

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  FrameCamId fcidl(current_frame, 0), fcidr(current_frame, 1);
  FrameCamId fcid_last(current_frame - 1, 0);
  KeypointsData kdl;

  pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[fcidl]);
  std::cout << "\nPROCESSING " << fcidl << "\n";

  MatchData md_stereo;
  LandmarkMatchData md;
  KeypointsData kdr;

  // 1st: Flow from last frame to current frame
  MatchData md_last;
  auto ckp1 = std::chrono::high_resolution_clock::now();
  double t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(ckp1 - start)
                  .count();
  total_t1 += t1 / 1e9;

  /// Do Optical Flow here
  if (current_frame > 0) {
    optical_flows_opencv(imgl_last, imgl, kd_last, kdl, md_last,
                         backproject_distance_threshold2, pyramid_level);
  }

  auto ckp2 = std::chrono::high_resolution_clock::now();
  double t2 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(ckp2 - ckp1).count();
  total_t2 += t2 / 1e9;

  // Add to all flows for visualization
  update_and_add_flows(fcid_last, md_last, all_flows, next_flow_id);

  /// Examine to add new flows by grids
  /// If number of empty cells cross a threshold * total_cells
  /// Then try to create new flows in empty cells
  double empty_cells_thresh = 0.99;
  size_t new_kps_added = add_flows_on_grids(
      imgl, kdl, num_features_per_image, num_bin_x, num_bin_y,
      empty_cells_thresh, last_frame_num_filled_cells);

  new_kps += new_kps_added;
  //  if (new_kps >= new_kf_num_new_keypoints) {
  //    std::cout << "\tAccumulate " << new_kps
  //              << " new keypoints. Take keyframe.\n";
  //    take_keyframe = true;
  //  }

  auto ckp3 = std::chrono::high_resolution_clock::now();
  double t3 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(ckp3 - ckp2).count();
  total_t3 += t3 / 1e9;

  if (take_keyframe) {
    // Reset some counter
    num_consecutive_regular_frames = 0;
    take_keyframe = false;
    new_kps = 0;

    pangolin::ManagedImage<uint8_t> imgr = pangolin::LoadImage(images[fcidr]);

    // Optical Flow to find keypoints and stereo matches to right image
    optical_flows_opencv(imgl, imgr, kdl, kdr, md_stereo,
                         backproject_distance_threshold2, pyramid_level);

    md_stereo.T_i_j = T_0_1;

    auto ckp4 = std::chrono::high_resolution_clock::now();
    double t4 =
        std::chrono::duration_cast<std::chrono::nanoseconds>(ckp4 - ckp3)
            .count();
    total_t4 += t4 / 1e9;

    Eigen::Matrix3d E;
    computeEssential(T_0_1, E);
    findInliersEssential(kdl, kdr, calib_cam.intrinsics[0],
                         calib_cam.intrinsics[1], E, 1e-3, md_stereo);

    std::cout << "KF Found " << md_stereo.inliers.size() << " stereo-matches."
              << std::endl;

    feature_corners[fcidl] = kdl;
    feature_corners[fcidr] = kdr;
    feature_matches.insert(
        std::make_pair(std::make_pair(fcidl, fcidr), md_stereo));

    // Update kd last and img last for next frame
    kd_last = kdl;
    imgl_last.CopyFrom(imgl);

    // CHange to optical flow version
    // Update find match landmark for Optical flow; landmark that does not
    // have obs last frame
    find_matches_landmarks_with_otpical_flow(fcid_last, md_last, flows, md);

    std::cout << "KF Found " << md.matches.size() << " matches." << std::endl;
    // TODO CHange to optical flow version
    localize_camera_optical_flow(
        current_pose, calib_cam.intrinsics[0], kdl, flows,
        reprojection_error_pnp_inlier_threshold_pixel, md);

    current_pose = md.T_w_c;

    cameras[fcidl].T_w_c = current_pose;
    cameras[fcidr].T_w_c = current_pose * T_0_1;
    // TODO CHange to optical flow version
    add_new_landmarks_optical_flow(fcidl, fcidr, kdl, kdr, calib_cam, md_stereo,
                                   md, flows, next_landmark_id);
    // Update match for next round
    last_md_featureToTrack = md;

    // TODO CHange to optical flow version
    remove_old_keyframes_optical_flow(fcidl, max_num_kfs, cameras, all_cameras,
                                      flows, old_landmarks, kf_frames);
    optimize();

    if (!opt_running && opt_finished) {
      //      opt_thread->join();
      // TODO If move to thread; for flows update only T_w_c
      flows = landmarks_opt;
      cameras = cameras_opt;
      calib_cam = calib_cam_opt;

      opt_finished = false;
    }

    current_pose = cameras[fcidl].T_w_c;
    // TODO Save all poses for left cam
    for (const auto& kv : cameras) {
      if (kv.first.cam_id == 0) {
        all_poses.emplace(kv.first, kv.second);
      }
    }

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    compute_projections();

    auto ckp5 = std::chrono::high_resolution_clock::now();
    double t5 =
        std::chrono::duration_cast<std::chrono::nanoseconds>(ckp5 - ckp4)
            .count();
    total_t5 += t5 / 1e9;

    current_frame++;
    return true;
  } else {
    //
    num_consecutive_regular_frames++;

    // Change kd_last to kdl for next frame
    kd_last = kdl;
    imgl_last.CopyFrom(imgl);

    feature_corners[fcidl] = kdl;

    // TODO Use Optical Flows version
    LandmarkMatchData md;
    find_matches_landmarks_with_otpical_flow(fcid_last, md_last, flows, md);

    std::cout << "Found " << md.matches.size() << " matches." << std::endl;

    localize_camera_optical_flow(
        current_pose, calib_cam.intrinsics[0], kdl, flows,
        reprojection_error_pnp_inlier_threshold_pixel, md);

    // Update match for next round
    last_md_featureToTrack = md;

    current_pose = md.T_w_c;
    Camera current_cam;
    current_cam.T_w_c = md.T_w_c;
    all_poses.emplace(fcidl, current_cam);

    if (int(md.inliers.size()) < new_kf_min_inliers
        //        || num_consecutive_regular_frames >=
        //        max_consecutive_regular_frames
        //        && !opt_running && !opt_finished
    ) {
      std::cout << "Found " << md.inliers.size() << " inliers matches."
                << "Take new keyframe next step!\n";
      take_keyframe = true;
    }

    //    if (!opt_running && opt_finished) {
    //      //      opt_thread->join();
    //      flows = landmarks_opt;
    //      cameras = cameras_opt;
    //      calib_cam = calib_cam_opt;

    //      opt_finished = false;
    //    }

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    auto ckp6 = std::chrono::high_resolution_clock::now();
    double t6 =
        std::chrono::duration_cast<std::chrono::nanoseconds>(ckp6 - ckp3)
            .count();
    total_t6 += t6 / 1e9;

    current_frame++;
    return true;
  }
}

// Compute reprojections for all landmark observations for visualization and
// outlier removal.
void compute_projections() {
  image_projections.clear();

  for (const auto& kv_lm : flows) {
    const TrackId track_id = kv_lm.first;

    for (const auto& kv_obs : kv_lm.second.obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].obs.push_back(proj_lm);
    }

    for (const auto& kv_obs : kv_lm.second.outlier_obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].outlier_obs.push_back(proj_lm);
    }
  }
}

// Optimize the active map with bundle adjustment
void optimize() {
  size_t num_obs = 0;
  for (const auto& kv : flows) {
    num_obs += kv.second.obs.size();
  }

  std::cerr << "Optimizing map with " << cameras.size() << " cameras, "
            << flows.size() << " points and " << num_obs << " observations."
            << std::endl;

  // Fix oldest two cameras to fix SE3 and scale gauge. Making the whole
  // second camera constant is a bit suboptimal, since we only need 1 DoF, but
  // it's simple and the initial poses should be good from calibration.
  FrameId fid = *(kf_frames.begin());
  // std::cout << "fid " << fid << std::endl;

  // Prepare bundle adjustment
  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = ba_optimize_intrinsics;
  ba_options.use_huber = true;
  ba_options.huber_parameter = reprojection_error_huber_pixel;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = ba_verbose;

  calib_cam_opt = calib_cam;
  cameras_opt = cameras;
  landmarks_opt = flows;

  opt_running = true;

  /*
   * Try bundle adjustment in foregroun here
   */
  std::set<FrameCamId> fixed_cameras = {{fid, 0}, {fid, 1}};

  bundle_adjustment_for_flows(feature_corners, ba_options, fixed_cameras,
                              calib_cam_opt, cameras_opt, landmarks_opt);

  opt_finished = true;
  opt_running = false;

  //  opt_thread.reset(new std::thread([fid, ba_options] {
  //    std::set<FrameCamId> fixed_cameras = {{fid, 0}, {fid, 1}};

  //    bundle_adjustment(feature_corners, ba_options, fixed_cameras,
  //    calib_cam_opt,
  //                      cameras_opt, landmarks_opt);

  //    opt_finished = true;
  //    opt_running = false;
  //  }));

  // Update project info cache
  compute_projections();
}

void save_trajectory() {
  all_cameras.insert(cameras.begin(), cameras.end());

  // add last frame as well
  FrameCamId last_fcidl(current_frame - 1, 0);
  Camera current_cam;
  current_cam.T_w_c = current_pose;
  all_cameras[last_fcidl] = current_cam;

  // Store the trajectory over
  std::ofstream trajectory_file("stamped_of_opencv_odometry_trajectory.txt");
  trajectory_file << std::fixed;

  if (trajectory_file.is_open()) {
    for (auto& camera : all_cameras) {
      FrameCamId fcid = camera.first;
      Camera current_camera = camera.second;
      if (fcid.cam_id == 1) {
        continue;
      }
      const auto& translation = current_camera.T_w_c.translation().data();
      const auto& quaternion_coefficients =
          current_camera.T_w_c.so3().unit_quaternion().coeffs();
      double ts = timestamps[fcid.frame_id] / 1e9;

      trajectory_file << ts << " " << translation[0] << " " << translation[1]
                      << " " << translation[2] << " "
                      << quaternion_coefficients.x() << " "
                      << quaternion_coefficients.y() << " "
                      << quaternion_coefficients.z() << " "
                      << quaternion_coefficients.w() << "\n";
    }
    trajectory_file.close();
    std::cout
        << "Trajectory is saved to stamped_of_opencv_odometry_trajectory.txt"
        << std::endl;
  } else {
    std::cout << "Fail to open the file" << std::endl;
  }
}

void save_all_trajectory() {
  // Store the trajectory over
  std::ofstream trajectory_file(
      "stamped_of_opencv_odometry_trajectory_all.txt");
  trajectory_file << std::fixed;

  if (trajectory_file.is_open()) {
    for (auto& camera : all_poses) {
      FrameCamId fcid = camera.first;
      Camera current_camera = camera.second;
      if (fcid.cam_id == 1) {
        continue;
      }
      const auto& translation = current_camera.T_w_c.translation().data();
      const auto& quaternion_coefficients =
          current_camera.T_w_c.so3().unit_quaternion().coeffs();
      double ts = timestamps[fcid.frame_id] / 1e9;

      trajectory_file << ts << " " << translation[0] << " " << translation[1]
                      << " " << translation[2] << " "
                      << quaternion_coefficients.x() << " "
                      << quaternion_coefficients.y() << " "
                      << quaternion_coefficients.z() << " "
                      << quaternion_coefficients.w() << "\n";
    }
    trajectory_file.close();
    std::cout << "Trajectory is saved to "
                 "stamped_of_opencv_odometry_trajectory_all.txt"
              << std::endl;
  } else {
    std::cout << "Fail to open the file" << std::endl;
  }
}
