#!/bin/bash
# S-Mamba on Traffic, look-back 96. Run from the repository root.
# Hyper-parameters of the official S-Mamba script (github.com/wzhwzhwzh0921/S-D-Mamba,
# scripts/multivariate_forecasting/Traffic/S_Mamba.sh); everything else is the default of run.py, which equals the
# default of the official code (d_state 32, batch 32, 10 epochs, patience 3, learning rate 1e-4).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
model_name=SMamba

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$(python scripts/paths.py root traffic.csv)/" \
  --data_path traffic.csv \
  --model_id Traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 862 --dec_in 862 --c_out 862 \
  --e_layers 4 --d_model 512 --d_ff 512 --learning_rate 0.001 --batch_size 16 \
  --des Exp --itr 1 $EXTRA_ARGS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$(python scripts/paths.py root traffic.csv)/" \
  --data_path traffic.csv \
  --model_id Traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 862 --dec_in 862 --c_out 862 \
  --e_layers 4 --d_model 512 --d_ff 512 --learning_rate 0.001 --batch_size 16 \
  --des Exp --itr 1 $EXTRA_ARGS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$(python scripts/paths.py root traffic.csv)/" \
  --data_path traffic.csv \
  --model_id Traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 862 --dec_in 862 --c_out 862 \
  --e_layers 4 --d_model 512 --d_ff 512 --learning_rate 0.002 --batch_size 16 \
  --des Exp --itr 1 $EXTRA_ARGS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$(python scripts/paths.py root traffic.csv)/" \
  --data_path traffic.csv \
  --model_id Traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 862 --dec_in 862 --c_out 862 \
  --e_layers 4 --d_model 512 --d_ff 512 --learning_rate 0.0008 --batch_size 16 \
  --des Exp --itr 1 $EXTRA_ARGS
