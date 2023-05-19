#!/bin/bash
PRETRAINED_MODEL=../training/model/ad08-fkeras/model_ToyCar.h5
X_NPY_DIR=./processed_data/64input_test_data.npy
Y_NPY_DIR=./processed_data/64input_test_data_ground_truths.npy
OUTPUT_DIR=./fault-analysis

NUM_VAL_INPUTS=(512 1024 2048 4096 8192 16384 32768 65536 131072 262144 481964)

for i in ${NUM_VAL_INPUTS[@]}; do
python3 sampling_faulty_eval_experiment.py \
        --config ad08-fkeras.yml \
        --output_dir ./fault-analysis \
        --x_npy_dir $X_NPY_DIR \
        --y_npy_dir $Y_NPY_DIR \
        --pretrained_model $PRETRAINED_MODEL \
        --batch_size 512 \
        --num_val_inputs $i
done