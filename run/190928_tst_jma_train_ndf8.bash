#!/bin/bash

case="ous_savp_190928_ndf8"

CUDA_VISIBLE_DEVICES=0 python ../scripts/train.py --input_dir ../data/jma --dataset jma \
		    --model savp --model_hparams_dict ../hparams/jma/ours_savp/model_hparams_BS8_ls5.json \
            --output_dir ../logs/jma/$case
