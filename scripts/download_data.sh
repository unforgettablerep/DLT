#!/usr/bin/env bash
# Download the datasets into the fixed data directory (dataset/, or ../data next to the repository; scripts/paths.py),
# the Time-Series-Library CSVs from the official THUML Hugging Face repository, PEMS08 flow data and its
# road-network edge list from a community mirror. PEMS04 (PEMS04.npz) is not on these mirrors; put it into
# dataset/PEMS/ by hand.
#   bash scripts/download_data.sh            # ETT, Weather, Exchange, Electricity, PEMS08
#   bash scripts/download_data.sh traffic    # also Traffic (136 MB)
set -euo pipefail
cd "$(dirname "$0")/.."
D=$(python scripts/paths.py data)
mkdir -p $D/ETT-small $D/weather $D/electricity $D/traffic $D/exchange_rate $D/PEMS

TSL=https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main
get() { [ -f "$2" ] || curl -fL -o "$2" "$1"; }
for f in ETTh1 ETTh2 ETTm1 ETTm2; do get "$TSL/ETT-small/$f.csv" $D/ETT-small/$f.csv; done
get "$TSL/weather/weather.csv"             $D/weather/weather.csv
get "$TSL/exchange_rate/exchange_rate.csv" $D/exchange_rate/exchange_rate.csv
get "$TSL/electricity/electricity.csv"     $D/electricity/electricity.csv
[ "${1:-}" = traffic ] && get "$TSL/traffic/traffic.csv" $D/traffic/traffic.csv

PEMS=https://huggingface.co/datasets/jimmygao3218/PEMS08/resolve/main
get "$PEMS/PEMS08.npz" $D/PEMS/PEMS08.npz          # (17856, 170, 3); channel 0 = flow
get "$PEMS/PEMS08.csv" $D/PEMS/PEMS08_edges.csv    # from,to,cost

find $D -type f | sort
