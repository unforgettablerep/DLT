#!/bin/bash
# result_done <file.tsv> [key=value ...] — awk breakpoint check against the TSV result store schema (src/utils/result_io.py).
result_done() {
    local f=$1; shift
    [[ -f "$f" ]] || return 1
    local want="" pair
    for pair in "$@"; do
        want+="${want:+;}${pair}"
    done
    awk -F'\t' -v want="$want" '
        NR == 1 {
            for (c = 1; c <= NF; c++) idx[$c] = c
            next
        }
        !($idx["status"] ~ /^ok/) { next }
        $idx["mse"] == "" { next }
        tolower($idx["mse"]) ~ /^([-+]?inf(inity)?|nan)$/ { next }
        {
            n = split(want, pairs, ";")
            ok = 1
            for (p = 1; p <= n && ok; p++) {
                split(pairs[p], kv, "=")
                col = idx[kv[1]]
                if (col == "" || $col != kv[2]) ok = 0
            }
            if (ok) found = 1
        }
        END { exit !found }
    ' "$f"
}
result_count_ok() {
    local f=$1; shift
    [[ -f "$f" ]] || { echo 0; return; }
    local want="" pair
    for pair in "$@"; do
        want+="${want:+;}${pair}"
    done
    awk -F'\t' -v want="$want" '
        NR == 1 {
            for (c = 1; c <= NF; c++) idx[$c] = c
            next
        }
        !($idx["status"] ~ /^ok/) { next }
        $idx["mse"] == "" { next }
        tolower($idx["mse"]) ~ /^([-+]?inf(inity)?|nan)$/ { next }
        {
            n = split(want, pairs, ";")
            ok = 1
            for (p = 1; p <= n && ok; p++) {
                split(pairs[p], kv, "=")
                col = idx[kv[1]]
                if (col == "" || $col != kv[2]) ok = 0
            }
            if (ok) count++
        }
        END { print count + 0 }
    ' "$f"
}
