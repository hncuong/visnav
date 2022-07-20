
runid=0
for i in `seq 1 5`
do
    for pyramid in 2 3
    do 
        for cell in 5 8 
        do
            echo $i $pyramid $cell $runid
            ./build/release/optical_flow_odometry --dataset-path data/V1_01_easy/mav0/ --dist-thresh 0.2  --show-gui false --nbin-x $cell --nbin-y $cell  --pyramid-lv $pyramid  --stereo-pyramid-lv $pyramid  --use-bs-fw false --use-bs-st false --run-id $runid > logs/run$runid.log
            runid=$((runid+1))

            echo $i $pyramid $cell $runid
            ./build/release/optical_flow_odometry --dataset-path data/V1_01_easy/mav0/ --dist-thresh 0.2  --show-gui false --nbin-x $cell --nbin-y $cell  --pyramid-lv $pyramid  --stereo-pyramid-lv $pyramid  --use-bs-fw true --use-bs-st true --run-id $runid > logs/run$runid.log
            runid=$((runid+1))
        done
    done
done
