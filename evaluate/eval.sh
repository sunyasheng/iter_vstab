#!/usr/bin/env bash

out_dir=/Users/yashengsun/Pictures/480p/Regular

for file in `ls $out_dir`
do
    if [ "${file:0-4}" == "_dir" ]
    then
        echo $file
        python eval_utils.py --output_video $out_dir/$file
    fi
done