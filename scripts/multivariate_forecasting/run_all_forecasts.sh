#!/bin/bash
# Forecast scheduler: dynamic least-loaded multi-GPU dispatch over the per-dataset scripts of one model.
#   bash scripts/multivariate_forecasting/run_all_forecasts.sh [--model DLT] [--gpu 0,1] [--para 1] [--datasets "ETTh1 Weather"]
# --model: DLT (default) or a baseline (S_Mamba, iTransformer, PatchTST, RLinear, DLinear, TiDE); the scripts are
# scripts/multivariate_forecasting/<group>/<model>[_<dataset>].sh (make_scripts.py). SEEDS and EXTRA_ARGS are passed
# on to every script. Results: output/long_term_forecast/<dataset>.tsv; log: log/multivariate_forecasting/.
set -uo pipefail

DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather Solar PEMS08 PEMS04 PEMS03 ECL PEMS07 Traffic"
script_of() {   # <model> <dataset> -> path of its script
    case "$2" in
        ETT*)   echo "scripts/multivariate_forecasting/ETT/$1_$2.sh" ;;
        PEMS*)  echo "scripts/multivariate_forecasting/PEMS/$1_$2.sh" ;;
        Solar)  echo "scripts/multivariate_forecasting/SolarEnergy/$1.sh" ;;
        *)      echo "scripts/multivariate_forecasting/$2/$1.sh" ;;
    esac
}

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT"; exit 1; }

GPUS="0"
PARA=1
MODEL=DLT
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)      GPUS="${2:-0}";  shift 2 ;;
        --para)     PARA="${2:-1}"; shift 2 ;;
        --model)    MODEL="${2:-DLT}"; shift 2 ;;
        --datasets) DATASETS="${2:-$DATASETS}"; shift 2 ;;
        -h|--help)
            echo "usage: bash scripts/multivariate_forecasting/run_all_forecasts.sh [--model NAME] [--gpu IDS] [--para N] [--datasets \"DS ...\"]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
SCRIPTS=()
for ds in $DATASETS; do SCRIPTS+=("$(script_of "$MODEL" "$ds")"); done

LOCK_DIR="$REPO_ROOT/temp/forecasts"
LOG_DIR="$REPO_ROOT/log/multivariate_forecasting"
mkdir -p "$LOCK_DIR" "$LOG_DIR"
LOCK_FILE="$LOCK_DIR/run.lock"
SCHED_LOG="$LOG_DIR/scheduler.log"
JOBS_FILE="$LOCK_DIR/jobs.txt"

JOBS=()
for s in ${SCRIPTS[@]+"${SCRIPTS[@]}"}; do
    [[ -z "$s" ]] && continue
    if [[ ! -f "$s" ]]; then
        echo -e "${YELLOW}WARN: not found, skipped: $s${NC}" >&2
        continue
    fi
    JOBS+=("$s")
done
if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo -e "${RED}ERROR: no scripts to run for --model $MODEL (python scripts/multivariate_forecasting/make_scripts.py writes them)${NC}" >&2
    exit 1
fi
printf '%s\n' "${JOBS[@]}" > "$JOBS_FILE"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_GPUS=${#GPU_ARR[@]}
for g in "${GPU_ARR[@]}"; do
    [[ "$g" =~ ^[0-9]+$ ]] || { echo "invalid --gpu id: $g" >&2; exit 1; }
done
[[ "$PARA" =~ ^[0-9]+$ ]] || { echo "invalid --para: $PARA" >&2; exit 1; }
(( PARA < 1 )) && PARA=1

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo -e "${RED}ERROR: a forecast scheduler is already running.${NC}" >&2
    echo -e "${RED}       Log: tail -f ${SCHED_LOG#"$REPO_ROOT"/}${NC}" >&2
    exit 1
fi

export _DLT_REPO_ROOT="$REPO_ROOT"
export _DLT_LOG_DIR="$LOG_DIR"
export _DLT_JOBS_FILE="$JOBS_FILE"
export _DLT_GPUS="$GPUS"
export _DLT_PARA="$PARA"
export _DLT_TOTAL="${#JOBS[@]}"

nohup bash -c '
set -uo pipefail
cd "$_DLT_REPO_ROOT" || exit 1

REPO_ROOT="$_DLT_REPO_ROOT"
LOG_DIR="$_DLT_LOG_DIR"
PARA="$_DLT_PARA"
TOTAL="$_DLT_TOTAL"

IFS="," read -ra GPU_ARR <<< "$_DLT_GPUS"
NUM_GPUS=${#GPU_ARR[@]}
MAX_JOBS=$(( NUM_GPUS * PARA ))

declare -A GPU_JOBS=()
for g in "${GPU_ARR[@]}"; do GPU_JOBS[$g]=0; done
declare -A PID2GPU=()

cleanup() {
    echo "[$(date "+%F %T")] terminating — killing ${#PID2GPU[@]} running job(s)..." >&2
    for p in "${!PID2GPU[@]}"; do kill "$p" 2>/dev/null; done
    sleep 1
    for p in "${!PID2GPU[@]}"; do kill -9 "$p" 2>/dev/null; done
    exit 130
}
trap cleanup SIGINT SIGTERM

echo "=========================================================="
echo "[$(date "+%F %T")] forecast scheduler started (pid $$)"
echo "  GPUs:       ${GPU_ARR[*]}"
echo "  slots/GPU:  $PARA   (max concurrent: $MAX_JOBS)"
echo "  jobs:       $TOTAL"
echo "=========================================================="

mapfile -t JOBS < "$_DLT_JOBS_FILE"
QI=0
LAUNCH_N=0
DONE=0
FAILED=0

while [[ $QI -lt $TOTAL ]] || [[ ${#PID2GPU[@]} -gt 0 ]]; do
    while [[ $QI -lt $TOTAL ]] && [[ ${#PID2GPU[@]} -lt $MAX_JOBS ]]; do
        best=""; best_n=999999
        for g in "${GPU_ARR[@]}"; do
            if [[ ${GPU_JOBS[$g]} -lt $PARA ]] && [[ ${GPU_JOBS[$g]} -lt $best_n ]]; then
                best=$g; best_n=${GPU_JOBS[$g]}
            fi
        done
        [[ -z "$best" ]] && break
        script="${JOBS[$QI]}"
        QI=$((QI + 1))
        base="$(basename "$(dirname "$script")")_$(basename "$script" .sh)"
        logf="${LOG_DIR}/${base}_gpu${best}_${LAUNCH_N}.log"
        LAUNCH_N=$((LAUNCH_N + 1))
        rel="${logf#"$REPO_ROOT"/}"
        echo "[$(date "+%T")] START [${QI}/${TOTAL}] ${base} -> GPU $best  (log: ${rel})"
        (
            CUDA_VISIBLE_DEVICES="$best" bash "$script" > "$logf" 2>&1
        ) &
        PID2GPU[$!]="$best"
        GPU_JOBS[$best]=$(( GPU_JOBS[$best] + 1 ))
    done

    [[ ${#PID2GPU[@]} -eq 0 ]] && break

    wait -n
    for p in "${!PID2GPU[@]}"; do
        if ! kill -0 "$p" 2>/dev/null; then
            wait "$p"; rc=$?
            g="${PID2GPU[$p]}"
            GPU_JOBS[$g]=$(( GPU_JOBS[$g] - 1 ))
            if [[ $rc -eq 0 ]]; then
                DONE=$((DONE + 1))
                echo "[$(date "+%T")] DONE  (rc=0)   pid $p gpu $g   [ok=$DONE fail=$FAILED]"
            else
                FAILED=$((FAILED + 1))
                echo "[$(date "+%T")] FAIL  (rc=$rc)  pid $p gpu $g   [ok=$DONE fail=$FAILED]"
            fi
            unset "PID2GPU[$p]"
        fi
    done
done

echo "=========================================================="
echo "[$(date "+%F %T")] scheduler finished — ok=$DONE fail=$FAILED of $TOTAL"
echo "=========================================================="
' >> "$SCHED_LOG" 2>&1 &

WORKER_PID=$!
disown "$WORKER_PID" 2>/dev/null || true

echo -e "${GREEN}[$(date '+%T')] Launched forecast scheduler as PID $WORKER_PID${NC}"
echo "  GPUs:      ${GPU_ARR[*]}  ($NUM_GPUS)"
echo "  slots/GPU: $PARA   (max concurrent: $((NUM_GPUS * PARA)))"
echo "  jobs:      ${#JOBS[@]}"
echo -e "  Log:       ${CYAN}tail -f ${SCHED_LOG#"$REPO_ROOT"/}${NC}"
echo -e "  Stop:      ${YELLOW}kill $WORKER_PID${NC}  (children reaped automatically)"
