for runid in `seq 1 5`
do
    python -W ignore rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_rpe.py stamped_groundtruth.txt results/stamped_odometry_trajectory_all_$runid.txt --delta 0.05 --delta_unit s --fixed_delta
done
