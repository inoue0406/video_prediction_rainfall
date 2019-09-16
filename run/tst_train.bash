#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../scripts/train.py --input_dir ../data/bair --dataset bair \
		    --model savp --model_hparams_dict ../hparams/bair_action_free/ours_savp/model_hparams_smallBS.json \
                    --output_dir ../logs/bair_action_free/ours_savp
