#!/usr/bin/env bash
#nohup python3 train.py --mode train > ./tmp.out &
python3 train.py --mode evaluate --img_dir ./test_in/1566207094610139.mp4 --out_dir ./test_out
python3 train.py --mode evaluate --img_dir ./test_in/1566207099687387.mp4 --out_dir ./test_out
python3 train.py --mode evaluate --img_dir ./test_in/1566207105070438.mp4 --out_dir ./test_out
python3 train.py --mode evaluate --img_dir ./test_in/1566207097138077.mp4 --out_dir ./test_out
python3 train.py --mode evaluate --img_dir ./test_in/1566207102508579.mp4 --out_dir ./test_out
