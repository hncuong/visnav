## Indirect Visual Odometry with Optical Flow Project for Vision-based Navigation course
Team member:
- Cuong Ha
- Warakorn Jetlohasiri

We adapted Optical flow to track and match keypoints in Visual Odometry (replace feature matching) by extending Visual Odometry pipeline.

Two Sparse Optical Flow methods were used:
- Lucas-Kanade method: OpenCV implementations.
- Usenko's KLT-based optical flow.

Optical flow were use to track and match keypoints for both frame to frame and stereo image pairs.

The extending code are mainly included in (`include/visnav/optical_flow.h`, `include/visnav/optical_flow_utils.h`, `include/visnav/of_grid.h`, `src/optical_flow_odometry.cpp`).
- `optical_flow.h`: contains Usenko's optical flow method implementations.
- `optical_flow_utils.h`: contains find matches landmarks, localization, add landmarks function for optical flow odometry.
- `of_grid`: grid-based keypoints adding to create new flows.
- `optical_flow_odometry`: optical flow visual odometry executor.

Evaluation were processed using ATE, RPE metrics with [TUM RGB-D benchmark tools](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools). 
Trajectory were evalutated with [RPG trajectory evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation). 

This code is a part of the practical course "Vision-based Navigation" (IN2106) taught at the Technical University of Munich.

It was originally developed for the winter term 2018. The latest iteration is winter term 2021/2022.

The authors are Vladyslav Usenko, Nikolaus Demmel, David Schubert and Zhakshylyk Nurlanov.

### License

The code for this practical course is provided under a BSD 3-clause license. See the LICENSE file for details.

Parts of the code (`include/tracks.h`, `include/union_find.h`) are adapted from OpenMVG and distributed under an MPL 2.0 licence.

Parts of the code (`include/local_parameterization_se3.hpp`, `src/test_ceres_se3.cpp`) are adapted from Sophus and distributed under an MIT license.

Note also the different licenses of thirdparty submodules.


You can find [setup instructions here.](wiki/Setup.md)
