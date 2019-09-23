#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../scripts/train.py --input_dir ../data/jma --dataset jma \
		    --model savp --model_hparams_dict ../hparams/jma/ours_savp/model_hparams_BS4.json \
            --output_dir ../logs/jma/ours_savp
