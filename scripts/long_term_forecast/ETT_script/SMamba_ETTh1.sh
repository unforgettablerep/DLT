#!/bin/bash
# S-Mamba on ETTh1, look-back 96. Run from the repository root.
# Hyper-parameters of the official S-Mamba script (github.com/wzhwzhwzh0921/S-D-Mamba,
# scripts/multivariate_forecasting/ETT/S_Mamba_ETTh1.sh); everything else is the default of run.py, which equals the
# default of the official code (d_state 32, batch 32, 10 epochs, patience 3, learning rate 1e-4).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
model_name=SMamba

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$(python scripts/paths.py root ETTh1.csv)/" \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 256 --d_ff 256 --d_state 2 --learning_rate 0.00007 \
  --des Exp --itr 1 $EXTRA_ARGS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$(python scripts/paths.py root ETTh1.csv)/" \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 256 --d_ff 256 --d_state 2 --learning_rate 0.00007 \
  --des Exp --itr 1 $EXTRA_ARGS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$(python scripts/paths.py root ETTh1.csv)/" \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 256 --d_ff 256 --d_state 2 --learning_rate 0.00005 \
  --des Exp --itr 1 $EXTRA_ARGS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$(python scripts/paths.py root ETTh1.csv)/" \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 256 --d_ff 256 --d_state 2 --learning_rate 0.00005 \
  --des Exp --itr 1 $EXTRA_ARGS
