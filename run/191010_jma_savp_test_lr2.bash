#!/bin/bash

case="191008_jma_savp_train_lr2"

CUDA_VISIBLE_DEVICES=0 python ../scripts/generate.py --input_dir ../data/jma_flat/train --dataset jma \
		            --dataset_hparams sequence_length=24 \
                    --checkpoint ../logs/jma/$case \
                    --mode test \
                    --results_dir ../logs/jma/$case

