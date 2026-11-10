#!/bin/bash
# One training run of the stand-alone DLT-dyn model: one GPU, one dataset.
#
#   bash scripts/train_dltdyn.sh                               # Weather, horizon 96, GPU 0, seed 2021
#   DATASET=PEMS08 PRED_LEN=12 bash scripts/train_dltdyn.sh    # another dataset / horizon
#   GPU=0 SEED=2022 DATA_DIR=/data/tslib bash scripts/train_dltdyn.sh
#   EXTRA_ARGS="--dlt_k 16 --dlt_R 8" bash scripts/train_dltdyn.sh      # any further run.py argument
#
# DATASET   Weather | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Exchange | ECL | Traffic | PEMS04 | PEMS08
#           The file is looked up in the Time-Series-Library layout ($DATA_DIR/weather/weather.csv,
#           $DATA_DIR/ETT-small/ETTh1.csv, $DATA_DIR/PEMS/PEMS08.npz, ...) and then directly in $DATA_DIR.
#           A missing CSV is downloaded by Time-Series-Library from Hugging Face (not the PEMS files).
# Settings  those of the reported runs: input 96, k = 8 CCF-selected neighbours, L = 48 lags, R = 4 basis
#           kernels, no gate; Adam 3e-3 halved every epoch, batch 32, 10 epochs, early stopping (patience 3).
#           PEMS at horizon 12 uses the TimeMixer protocol (L1 loss; MAE / MAPE / RMSE on the original scale).
# Output    ./checkpoints/<setting>/checkpoint.pth and one JSON line per run in ./results/quickstart.jsonl
#           (metrics, epochs, seconds per epoch). Without CUDA the run falls back to the CPU.
set -e
cd "$(dirname "$0")/.."                      # run.py finds models/ relative to the repository root

DATASET=${DATASET:-Weather}
PRED_LEN=${PRED_LEN:-96}
GPU=${GPU:-0}
SEED=${SEED:-2021}
DATA_DIR=${DATA_DIR:-./dataset}

case $DATASET in        # sub-directory, file, --data, variables, --freq
  Weather)  SUB=weather;       FILE=weather.csv;       DATA=custom; C=21;  FREQ=t ;;
  ETTh1)    SUB=ETT-small;     FILE=ETTh1.csv;         DATA=ETTh1;  C=7;   FREQ=h ;;
  ETTh2)    SUB=ETT-small;     FILE=ETTh2.csv;         DATA=ETTh2;  C=7;   FREQ=h ;;
  ETTm1)    SUB=ETT-small;     FILE=ETTm1.csv;         DATA=ETTm1;  C=7;   FREQ=t ;;
  ETTm2)    SUB=ETT-small;     FILE=ETTm2.csv;         DATA=ETTm2;  C=7;   FREQ=t ;;
  Exchange) SUB=exchange_rate; FILE=exchange_rate.csv; DATA=custom; C=8;   FREQ=d ;;
  ECL)      SUB=electricity;   FILE=electricity.csv;   DATA=custom; C=321; FREQ=h ;;
  Traffic)  SUB=traffic;       FILE=traffic.csv;       DATA=custom; C=862; FREQ=h ;;
  PEMS04)   SUB=PEMS;          FILE=PEMS04.npz;        DATA=PEMS;   C=307; FREQ=t ;;
  PEMS08)   SUB=PEMS;          FILE=PEMS08.npz;        DATA=PEMS;   C=170; FREQ=t ;;
  *) echo "unknown DATASET '$DATASET'"; exit 1 ;;
esac
ROOT=$DATA_DIR/$SUB
[ -f "$ROOT/$FILE" ] || { [ -f "$DATA_DIR/$FILE" ] && ROOT=$DATA_DIR; }

LABEL_LEN=48; PROTOCOL=""
if [ $DATA = PEMS ]; then
  LABEL_LEN=0
  [ "$PRED_LEN" = 12 ] && PROTOCOL="--loss MAE --inverse --vali_inverse"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}     # single GPU; inside the process it is cuda:0
mkdir -p results
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model DLT --model_id ${DATASET}_96_${PRED_LEN} --des quickstart \
  --root_path "$ROOT/" --data_path $FILE --data $DATA --features M --freq $FREQ \
  --seq_len 96 --label_len $LABEL_LEN --pred_len $PRED_LEN \
  --enc_in $C --dec_in $C --c_out $C \
  --dlt_variant dltdyn --dlt_impl basis --dlt_k 8 --dlt_L 48 --dlt_R 4 \
  --learning_rate 0.003 --batch_size 32 --train_epochs 10 --patience 3 \
  --fix_seed $SEED --gpu 0 --num_workers 4 --itr 1 \
  --no_epoch_test --no_save_preds --result_jsonl ./results/quickstart.jsonl \
  $PROTOCOL $EXTRA_ARGS

python - <<'EOF'
import json
r = json.loads(open('./results/quickstart.jsonl').read().strip().splitlines()[-1])
sec = r.get('epoch_secs'); sec = sum(sec) / len(sec) if isinstance(sec, list) else sec
print(f"\n{r['model_id']} seed {r['seed']}: MSE {r['mse']:.4f}  MAE {r['mae']:.4f}"
      + (f"  RMSE {r['rmse']:.2f}  MAPE {100 * r['mape_clip5']:.2f}%" if r['inverse'] else '')
      + f"  | best epoch {r.get('best_epoch')} of {r.get('epochs')}, {sec:.1f} s per epoch, {r['n_params']} parameters")
EOF
