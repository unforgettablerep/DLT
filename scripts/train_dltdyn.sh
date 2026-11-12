#!/bin/bash
# One training run of the stand-alone DLT-dyn model: one GPU, one dataset.
#
#   bash scripts/train_dltdyn.sh                           # Weather, horizon 96, seed 2021
#   bash scripts/train_dltdyn.sh PEMS08 12                 # dataset and horizon
#   bash scripts/train_dltdyn.sh ECL 96 2022 --dlt_k 16    # seed, then any further run.py argument
#
# Dataset   Weather | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Exchange | ECL | Traffic | PEMS04 | PEMS08
#           read from the fixed directory dataset/ (scripts/paths.py): Time-Series-Library layout
#           (dataset/weather/weather.csv, dataset/PEMS/PEMS08.npz, ...) or all files directly in dataset/.
#           A missing CSV is downloaded by Time-Series-Library from Hugging Face (not the PEMS files).
# Settings  those of the reported runs: input 96, k = 8 CCF-selected neighbours, L = 48 lags, R = 4 basis
#           kernels, no gate; Adam 3e-3 halved every epoch, batch 32, 10 epochs, early stopping (patience 3).
#           PEMS at horizon 12 uses the TimeMixer protocol (L1 loss; MAE / MAPE / RMSE on the original scale).
# Output    in the fixed directory work/: ckpt/<setting>/checkpoint.pth and one JSON line per run in
#           results_quickstart.jsonl (metrics, epochs, seconds per epoch).
# Device    GPU 0; without CUDA the run falls back to the CPU.
set -e
cd "$(dirname "$0")/.."                      # run.py finds models/ relative to the repository root

DATASET=${1:-Weather}
PRED_LEN=${2:-96}
SEED=${3:-2021}
[ $# -ge 3 ] && shift 3 || shift $#          # what is left goes to run.py

case $DATASET in        # file, --data, variables, --freq
  Weather)  FILE=weather.csv;       DATA=custom; C=21;  FREQ=t ;;
  ETTh1)    FILE=ETTh1.csv;         DATA=ETTh1;  C=7;   FREQ=h ;;
  ETTh2)    FILE=ETTh2.csv;         DATA=ETTh2;  C=7;   FREQ=h ;;
  ETTm1)    FILE=ETTm1.csv;         DATA=ETTm1;  C=7;   FREQ=t ;;
  ETTm2)    FILE=ETTm2.csv;         DATA=ETTm2;  C=7;   FREQ=t ;;
  Exchange) FILE=exchange_rate.csv; DATA=custom; C=8;   FREQ=d ;;
  ECL)      FILE=electricity.csv;   DATA=custom; C=321; FREQ=h ;;
  Traffic)  FILE=traffic.csv;       DATA=custom; C=862; FREQ=h ;;
  PEMS04)   FILE=PEMS04.npz;        DATA=PEMS;   C=307; FREQ=t ;;
  PEMS08)   FILE=PEMS08.npz;        DATA=PEMS;   C=170; FREQ=t ;;
  *) echo "unknown dataset '$DATASET'"; exit 1 ;;
esac
ROOT=$(python scripts/paths.py root $FILE)
WORK=$(python scripts/paths.py work)

LABEL_LEN=48; PROTOCOL=""
if [ $DATA = PEMS ]; then
  LABEL_LEN=0
  [ "$PRED_LEN" = 12 ] && PROTOCOL="--loss MAE --inverse --vali_inverse"
fi

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --model DLT --model_id ${DATASET}_96_${PRED_LEN} --des quickstart \
  --root_path "$ROOT/" --data_path $FILE --data $DATA --features M --freq $FREQ \
  --seq_len 96 --label_len $LABEL_LEN --pred_len $PRED_LEN \
  --enc_in $C --dec_in $C --c_out $C \
  --dlt_variant dltdyn --dlt_impl basis --dlt_k 8 --dlt_L 48 --dlt_R 4 \
  --learning_rate 0.003 --batch_size 32 --train_epochs 10 --patience 3 \
  --fix_seed $SEED --gpu 0 --num_workers 4 --itr 1 \
  --checkpoints "$WORK/ckpt" --no_epoch_test --no_save_preds --result_jsonl "$WORK/results_quickstart.jsonl" \
  $PROTOCOL "$@"

python - "$WORK/results_quickstart.jsonl" <<'EOF'
import json, sys
r = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
sec = r.get('epoch_secs'); sec = sum(sec) / len(sec) if isinstance(sec, list) else sec
print(f"\n{r['model_id']} seed {r['seed']}: MSE {r['mse']:.4f}  MAE {r['mae']:.4f}"
      + (f"  RMSE {r['rmse']:.2f}  MAPE {100 * r['mape_clip5']:.2f}%" if r['inverse'] else '')
      + f"  | best epoch {r.get('best_epoch')} of {r.get('epochs')}, {sec:.1f} s per epoch, {r['n_params']} parameters")
EOF
