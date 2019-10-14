#!/bin/bash

case="191007_jma_savp_train_basecase.bash"

CUDA_VISIBLE_DEVICES=0 python ../scripts/train.py --input_dir ../data/jma_flat --dataset jma \
		            --model savp --model_hparams_dict ../hparams/jma/ours_savp/model_hparams_BS8.json \
                    --summary_freq 500 --image_summary_freq 500 --eval_summary_freq 500 \
                    --accum_eval_summary_freq 10000 --save_freq 1000 \
                    --output_dir ../logs/jma/$case

