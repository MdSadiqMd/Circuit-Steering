#!/bin/bash
# Pause a run's whole process group while the GPU is hot: SIGSTOP at >= STOP_C, SIGCONT at <= RESUME_C.
# Queued kernels finish, then the process sleeps; nothing is lost. Logs one line a minute.
# Usage: thermal_guard.sh <pgid> <log>
pgid=$1 log=$2 STOP_C=${STOP_C:-80} RESUME_C=${RESUME_C:-70} paused=0 i=0
while kill -0 -- -"$pgid" 2>/dev/null; do
  read -r t p u m < <(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits | tr -d ',')
  if ((paused == 0 && t >= STOP_C)); then
    kill -STOP -- -"$pgid"
    paused=1
    echo "$(date +%T) PAUSE ${t}C" >>"$log"
  fi
  if ((paused == 1 && t <= RESUME_C)); then
    kill -CONT -- -"$pgid"
    paused=0
    echo "$(date +%T) RESUME ${t}C" >>"$log"
  fi
  ((i++ % 4 == 0)) && echo "$(date +%T) ${t}C ${p}W util ${u}% mem ${m}MiB paused=$paused" >>"$log"
  sleep 15
done
((paused)) && kill -CONT -- -"$pgid" 2>/dev/null
echo "$(date +%T) guard exit" >>"$log"
