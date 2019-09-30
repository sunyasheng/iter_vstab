#!/usr/bin/env bash
#nohup python3 train.py --mode train > ./tmp.out &
#python3 train.py --mode evaluate --img_dir ./test_in/examples_5 --out_dir ./examples_5_out --stab_iter 5 #--debug True
#python3 train.py --mode evaluate --img_dir ./test_in/1566207094610139.mp4 --out_dir ./test_out --stab_iter 5
#python3 train.py --mode evaluate --img_dir ./test_in/1566207099687387.mp4 --out_dir ./test_out --stab_iter 5
#python3 train.py --mode evaluate --img_dir ./test_in/1566207105070438.mp4 --out_dir ./test_out --stab_iter 5
#python3 train.py --mode evaluate --img_dir ./test_in/1566207097138077.mp4 --out_dir ./test_out --stab_iter 5
#python3 train.py --mode evaluate --img_dir ./test_in/1566207102508579.mp4 --out_dir ./test_out --stab_iter 5
for file in `ls ./test_in/Regular`
do
    if [ "${file:0-4}" == "_dir" ]
    then
        python3 train.py --mode evaluate --img_dir ./test_in/Regular/$file --out_dir ./test_out/Regular/$file --stab_iter 5
    fi
done