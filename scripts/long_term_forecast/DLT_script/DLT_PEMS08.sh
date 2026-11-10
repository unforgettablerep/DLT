#!/bin/bash
# DLT on PEMS08. Run from the repository root:  bash scripts/long_term_forecast/DLT_script/DLT_PEMS08.sh
#   VARIANT=dlt bash ...            choose the kernel variant (default: dltdyn)
#   EXTRA_ARGS="--no_use_gpu --num_workers 0" bash ...   force CPU
#   EXTRA_ARGS="--use_multi_gpu --devices 0,1" bash ...  multi-GPU (DataParallel)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
model_name=DLT
variant=${VARIANT:-dltdyn}

for pred_len in 12 24 48 96; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_96_$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --freq t \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --dlt_variant $variant \
    --dlt_k 8 --dlt_L 48 --dlt_R 4 \
    --learning_rate 0.003 \
    --batch_size 32 \
    --train_epochs 10 \
    --patience 3 \
    --des Exp \
    --itr 1 $EXTRA_ARGS
done
