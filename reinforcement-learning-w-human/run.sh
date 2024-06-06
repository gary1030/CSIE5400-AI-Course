#!/bin/bash

python main.py \
    --exp_name "${1}" \
    --model_name "${2}" \
    --train \
    --wandb_token "${3}" \
    --num_epochs "${5}" \
    --beta "${6}" \
    --optimizer "${4}" \
    --train_batch_size 1 \
    --eval_batch_size 1 \

