for runid in `seq 0 39`
do
    python -W ignore rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_rpe.py stamped_groundtruth.txt results/stamped_trajectory_optical_flow_all_$runid.txt --delta 0.05 --delta_unit s --fixed_delta
done
