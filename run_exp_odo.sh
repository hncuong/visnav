runid=0
for i in `seq 5 10`
do
    echo $i $runid
    ./build/release/odometry --dataset-path data/V1_01_easy/mav0/ --show-gui false --run-id $runid > logs/run_odo_$runid.log
    runid=$((runid+1))
done
