#!/bin/bash
# Shared GPU queue: round-robin GPU assignment and per-GPU FIFO concurrency limits.

gpu_queue_init() {
    _GPU_ARR=()
    _GPU_FD=()
    _GPU_IDX=0
    _GPU_CNT=0
    _CURRENT_GPU=""
    _GPU_INFLIGHT=0
    _PARA_PER_GPU="${PARA_PER_GPU:-1}"
    ((_PARA_PER_GPU < 1)) && _PARA_PER_GPU=1

    if [[ -n "${GPU_LIST:-}" ]]; then
        read -ra _GPU_ARR <<< "${GPU_LIST//,/ }"
        _GPU_CNT=${#_GPU_ARR[@]}
        local next_fd=20
        local gpu sem
        for gpu in "${_GPU_ARR[@]}"; do
            if [[ -z "${_GPU_FD[$gpu]+x}" ]]; then
                sem="${GPU_QUEUE_SESSION:-/tmp}/_gpu_sem_${gpu}_$$"
                mkfifo "$sem"
                eval "exec ${next_fd}<>\"$sem\""
                rm -f "$sem"
                _GPU_FD[$gpu]=${next_fd}
                for ((i = 0; i < _PARA_PER_GPU; i++)); do
                    eval "echo >&${next_fd}"
                done
                ((next_fd++))
            fi
        done
        if [[ -z "${_GPU_COUNTER_FILE:-}" ]]; then
            _GPU_COUNTER_FILE="${GPU_QUEUE_SESSION:-/tmp}/_gpu_counter_$$"
            echo 0 > "$_GPU_COUNTER_FILE"
        fi
        _GPU_SLOTS=$((_GPU_CNT * _PARA_PER_GPU))
    else
        _GPU_SLOTS="${PARALLEL:-1}"
        ((_GPU_SLOTS < 1)) && _GPU_SLOTS=1
    fi
}

gpu_queue_next() {
    if [[ -n "${GPU_LIST:-}" ]]; then
        if [[ -n "${_GPU_COUNTER_FILE:-}" && -f "${_GPU_COUNTER_FILE:-}" ]]; then
            local idx
            idx=$(flock "${_GPU_COUNTER_FILE}" bash -c 'read -r v < "$1"; echo $((v+1)) > "$1"; printf "%d" "$v"' _ "${_GPU_COUNTER_FILE}")
            _CURRENT_GPU="${_GPU_ARR[$((idx % _GPU_CNT))]}"
        else
            _CURRENT_GPU="${_GPU_ARR[$((_GPU_IDX % _GPU_CNT))]}"
            _GPU_IDX=$((_GPU_IDX + 1))
        fi
    fi
}

gpu_queue_acquire() {
    if [[ -n "${GPU_LIST:-}" ]]; then
        eval "read -r _ <&${_GPU_FD[$_CURRENT_GPU]}"
    fi
}

gpu_queue_release() {
    if [[ -n "${GPU_LIST:-}" ]]; then
        eval "echo >&${_GPU_FD[$_CURRENT_GPU]}"
    fi
}

gpu_queue_dispatch() {
    local logfile=$1
    shift
    if [[ -n "${GPU_LIST:-}" ]]; then
        local gpu="$_CURRENT_GPU"
        (
            gpu_queue_acquire
            if [[ -n "$logfile" ]]; then
                CUDA_VISIBLE_DEVICES="$gpu" "$@" >> "$logfile" 2>&1
            else
                CUDA_VISIBLE_DEVICES="$gpu" "$@"
            fi
            gpu_queue_release
        ) &
    else
        if [[ -n "$logfile" ]]; then
            "$@" >> "$logfile" 2>&1 &
        else
            "$@" &
        fi
        _GPU_INFLIGHT=$((_GPU_INFLIGHT + 1))
        while (( _GPU_INFLIGHT >= _GPU_SLOTS )); do
            wait -n || true
            _GPU_INFLIGHT=$((_GPU_INFLIGHT - 1))
        done
    fi
}

gpu_queue_wait() {
    wait
    _GPU_INFLIGHT=0
}
