#!/bin/bash
PRETRAINED_MODEL=../training/model/ad08-fkeras/model_ToyCar.h5
X_NPY_DIR=./processed_data/64input_test_data.npy
Y_NPY_DIR=./processed_data/64input_test_data_ground_truths.npy
OUTPUT_DIR=./fault-analysis

# Sanity check
#bits=10600
bits=248832
VMs=8
system=0
lbi=$((bits/VMs * system))
hbi=$((bits/VMs * system + bits/VMs))
for (( i=$lbi; i<$hbi ; i++ )); do 
echo "Sanity check ber = 0"
CUDA_VISIBLE_DEVICES="" python3 sampling_faulty_eval_experiment.py \
        --config ad08-fkeras.yml \
        --output_dir ./fault-analysis \
        --x_npy_dir $X_NPY_DIR \
        --y_npy_dir $Y_NPY_DIR \
        --pretrained_model $PRETRAINED_MODEL \
        --efd_fp "./efd_emd_hesstrace_v0-799999_b${lbi}-${hbi}-forloop.log" \
        --efr_fp "./efr_emd_hesstrace_v0-799999_b${lbi}-${hbi}-forloop.log" \
        --efx_overwrite 0 \
        --use_custom_bfr 1 \
        --bfr_start $i \
        --bfr_end   $((i+1)) \
        --bfr_step  1 \
        --batch_size 512 \
        --num_val_inputs 4096
exit
done