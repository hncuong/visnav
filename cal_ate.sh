runid=0
for runid in `seq 0 39`
do
    python rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_ate.py stamped_groundtruth.txt results/stamped_trajectory_optical_flow_all_$runid.txt
done
