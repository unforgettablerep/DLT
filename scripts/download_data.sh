#!/usr/bin/env bash
# Download the forecasting datasets into the fixed data directory (dataset/, or ../data next to the repository;
# scripts/paths.py) in the Time-Series-Library layout, then check every file against its published size
# (scripts/check_data.py).
#
#   bash scripts/download_data.sh                  all groups (about 560 MB on disk, 380 MB to download)
#   bash scripts/download_data.sh PEMS Solar       only these groups
#   bash scripts/download_data.sh --mirror ...     Hugging Face through hf-mirror.com and GitHub files through
#                                                  jsDelivr first, for networks where huggingface.co or GitHub
#                                                  are blocked or slow
#
# group     files (under the data directory)                            source, tried in this order
# ETT       ETT-small/ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv            Hugging Face thuml/Time-Series-Library
# Exchange  exchange_rate/exchange_rate.csv                             (the THUML copy of the long-term
# Weather   weather/weather.csv                                         forecasting datasets)
# ECL       electricity/electricity.csv (96 MB)
# Traffic   traffic/traffic.csv (137 MB)
# ILI       illness/national_illness.csv
# PEMS      PEMS/PEMS03.npz PEMS04.npz PEMS07.npz PEMS08.npz (111 MB)   GitHub guoshnBJTU/ASTGNN (the authors of the
#           PEMS/PEMS0x_edges.csv: road network (from, to, distance)    PEMS03/04/07/08 benchmark), then the Hugging
#           PEMS/PEMS03_ids.txt: sensor ids used by PEMS03_edges.csv    Face mirror jimmygao3218/PEMS0x (04, 07, 08)
# Solar     Solar/solar_AL.txt (180 MB, a 9 MB download)                GitHub laiguokun/multivariate-time-series-data
#                                                                       (the LSTNet data), gzip
#
# A file that is already there, in this layout or directly in the data directory, is kept. PEMS04 and PEMS08 of ASTGNN are byte-identical to the copies used for
# P0-P2. Without network access, copy the files to the same places by hand (e.g. from the dataset archive of
# iTransformer or Time-Series-Library) and run python scripts/check_data.py.
set -uo pipefail
cd "$(dirname "$0")/.."
ALL="ETT Exchange Weather ECL Traffic ILI PEMS Solar"
MIRROR=0; SEL=()
for a in "$@"; do
  case "$a" in
    --mirror) MIRROR=1 ;;
    ETT|Exchange|Weather|ECL|Traffic|ILI|PEMS|Solar) SEL+=("$a") ;;
    *) echo "unknown argument '$a'; groups: $ALL; option: --mirror"; exit 2 ;;
  esac
done
[ ${#SEL[@]} -gt 0 ] || read -r -a SEL <<< "$ALL"
D=$(python scripts/paths.py data)
HF=$([ $MIRROR = 1 ] && echo https://hf-mirror.com || echo https://huggingface.co)
TSL=$HF/datasets/thuml/Time-Series-Library/resolve/main
echo "data directory: $D"

FAILED=()
get() {   # get <path under the data directory> <url> [<url> ...]: the first URL that works; a file that is there
          # (in the Time-Series-Library layout or directly in the data directory) is kept
  local rel=$1; shift
  local dest="$D/$rel"
  if [ -s "$dest" ] || [ -s "$D/$(basename "$rel")" ]; then echo "have  $rel"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  for url in "$@"; do
    if curl -fL --retry 2 --connect-timeout 20 -sS -o "$dest.part" "$url"; then
      mv "$dest.part" "$dest"; echo "got   $rel  <- $url"; return 0
    fi
  done
  rm -f "$dest.part"; echo "FAIL  $rel  (tried: $*)"; FAILED+=("$rel"); return 1
}
github() {   # github <owner/repo> <branch> <path> [<other URL>]: raw GitHub, jsDelivr (at most 20 MB per file),
             # then the other URL; with --mirror jsDelivr, the other URL, raw GitHub
  local raw="https://raw.githubusercontent.com/$1/$2/$3" cdn="https://cdn.jsdelivr.net/gh/$1@$2/$3" other=${4:-}
  if [ $MIRROR = 1 ]; then echo "$cdn $other $raw"; else echo "$raw $cdn $other"; fi
}

for g in "${SEL[@]}"; do
  case $g in
    ETT)      for f in ETTh1 ETTh2 ETTm1 ETTm2; do get "ETT-small/$f.csv" "$TSL/ETT-small/$f.csv"; done ;;
    Exchange) get exchange_rate/exchange_rate.csv "$TSL/exchange_rate/exchange_rate.csv" ;;
    Weather)  get weather/weather.csv "$TSL/weather/weather.csv" ;;
    ECL)      get electricity/electricity.csv "$TSL/electricity/electricity.csv" ;;
    Traffic)  get traffic/traffic.csv "$TSL/traffic/traffic.csv" ;;
    ILI)      get illness/national_illness.csv "$TSL/illness/national_illness.csv" ;;
    PEMS)
      for i in 03 04 07 08; do
        hf=""; [ $i = 03 ] || hf="$HF/datasets/jimmygao3218/PEMS$i/resolve/main/PEMS$i"
        # shellcheck disable=SC2046
        get "PEMS/PEMS$i.npz" $(github guoshnBJTU/ASTGNN main "data/PEMS$i/PEMS$i.npz" ${hf:+"$hf.npz"})
        # shellcheck disable=SC2046
        get "PEMS/PEMS${i}_edges.csv" $(github guoshnBJTU/ASTGNN main "data/PEMS$i/PEMS$i.csv" ${hf:+"$hf.csv"})
      done
      # shellcheck disable=SC2046
      get PEMS/PEMS03_ids.txt $(github guoshnBJTU/ASTGNN main data/PEMS03/PEMS03.txt) ;;
    Solar)
      if [ -s "$D/Solar/solar_AL.txt" ] || [ -s "$D/solar_AL.txt" ]; then echo "have  Solar/solar_AL.txt"
      # shellcheck disable=SC2046
      elif get Solar/solar_AL.txt.gz $(github laiguokun/multivariate-time-series-data master solar-energy/solar_AL.txt.gz); then
        gunzip -f "$D/Solar/solar_AL.txt.gz" && echo "      unpacked Solar/solar_AL.txt"
      fi ;;
  esac
done

echo
[ ${#FAILED[@]} -eq 0 ] || echo "not downloaded: ${FAILED[*]} (try again, with --mirror, or copy the files by hand)"
python scripts/check_data.py "${SEL[@]}"
