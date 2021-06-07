#!/bin/sh

# variables adjusted for quicker testing
# from paper episodes train 50

MODEL=${1?Error: no name given}

python3 test_generated.py \
--gpu 0 \
--cpu \
--arch default_convnet \
--workers 1 \
--n_episodes 4 \
--n_way 4 \
--n_support 5 \
--n_query 5 \
--checkpoint 'models_trained/'$MODEL'/model_best_acc.pth.tar' \
--evaluation_name $MODEL \
--test_gen 'train_gan_aug_meta' \
--support_ori 'train_ori_specs_meta'



