#!/bin/bash

case="190930_jma_flat_train_ndf8"

CUDA_VISIBLE_DEVICES=0 python ../scripts/train.py --input_dir ../data/jma_flat --dataset jma \
		    --model savp --model_hparams_dict ../hparams/jma/ours_savp/model_hparams_BS8.json \
            --output_dir ../logs/jma/$case
