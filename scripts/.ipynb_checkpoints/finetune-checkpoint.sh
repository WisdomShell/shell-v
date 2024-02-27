#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

################## CodeShell ##################
PROMPT_VERSION="codeshell"
MODEL_VERSION="codeshell-chat"
################## CodeShell ##################


deepspeed  --master_port 29502 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./ckpt/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./data/llava_data/sft/llava665k_and_hal_instruct.json \
    --image_folder ./data/ \
    --vision_tower ./ckpt/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./ckpt/shell-v-Pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./output/shell-v-SFT  \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
