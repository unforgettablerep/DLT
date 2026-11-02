#!/usr/bin/env bash
# Download the datasets used in the feasibility study into ./data
# (Time-Series-Library CSVs from the official THUML Hugging Face repo;
#  PEMS08 flow data + road-network edge list from a community mirror).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

TSL=https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main
curl -fL -o data/ETTh1.csv       "$TSL/ETT-small/ETTh1.csv"
curl -fL -o data/weather.csv     "$TSL/weather/weather.csv"
curl -fL -o data/electricity.csv "$TSL/electricity/electricity.csv"
# optional (not used in the reported runs):
# curl -fL -o data/traffic.csv   "$TSL/traffic/traffic.csv"

PEMS=https://huggingface.co/datasets/jimmygao3218/PEMS08/resolve/main
curl -fL -o data/PEMS08.npz       "$PEMS/PEMS08.npz"   # (17856, 170, 3); channel 0 = flow
curl -fL -o data/PEMS08_edges.csv "$PEMS/PEMS08.csv"   # from,to,cost

ls -la data
