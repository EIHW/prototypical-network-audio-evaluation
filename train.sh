#!/bin/sh

# variables adjusted for quicker testing
# from paper epoches 10, episodes train 100, episdoes val 40

MODEL=${1?Error: no name given}

python3 train.py \
--arch default_convnet \
--workers 1 \
--epochs 1 \
--print-freq 10 \
--optimizer adam \
--step_size 20 \
--gamma 0.5 \
--lr 0.001 \
-j 4 \
--model_name $MODEL \
--n_query_train 20 \
--n_query_val 10 \
--n_support 5 \
--n_way_train 4 \
--n_way_val 4 \
--n_episodes_train 5 \
--n_episodes_val 4 \
--train_csv 'train_ori_specs_meta'\

