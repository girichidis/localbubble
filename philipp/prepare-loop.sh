#!/bin/bash

mkdir data pics
for file in SILCC_hdf5_plt_cnt_????
do
    for coldens in 5e-5 1e-4 1.67e-4 5e-4 1e-3 1.67e-3
    do
        for radius in 150
        do
            echo python ../localbubble/philipp/Mollweide.py $file -LB -radius $radius -Sigma_crit $coldens -odir_data data -odir pics
        done
    done
done
