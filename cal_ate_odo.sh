runid=0
for runid in `seq 1 5`
do
    python rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_ate.py stamped_groundtruth.txt results/stamped_odometry_trajectory_all_$runid.txt
done
