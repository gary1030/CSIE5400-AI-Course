#!/bin/bash

# python main.py \
#     --exp_name "${1}" \
#     --model_name "${2}" \
#     --train \
#     --wandb_token "${3}" \
#     --num_epochs 1 \
#     --max_prompt_length 512 \
#     --train_batch_size 1 \
#     --eval_batch_size 1 \

python main.py \
    --exp_name "DPO" \
    --model_name "unsloth/llama-3-8b-bnb-4bit" \
    --train \
    --wandb_token "798d552535c4742bf29c5d005e9a6fe2f3addaac" \
    --num_epochs 1 \
    --optimizer "paged_adamw_32bit" \
    --train_batch_size 1 \
    --eval_batch_size 1 \

python main.py \
    --exp_name "ORPO" \
    --model_name "unsloth/llama-3-8b-bnb-4bit" \
    --train \
    --wandb_token "798d552535c4742bf29c5d005e9a6fe2f3addaac" \
    --num_epochs 1 \
    --optimizer "paged_adamw_32bit" \
    --train_batch_size 1 \
    --eval_batch_size 1 \
