#!/usr/bin/env bash
# Linux counterpart of run_paperB_complete.ps1: detached, low priority. The script
# checkpoints per pool x drift cell and resumes. 24-core / 62 GB machine -> 20 workers.
cd "$(dirname "$0")"
export WORKERS="${WORKERS:-20}" PYTHONIOENCODING=utf-8
log=logs/run_paperB.status
for s in paperB_complete; do
  echo "$s start $(date +%H:%M:%S) (linux, WORKERS=$WORKERS)" >> $log
  nice -n 10 python3 -u "$s.py" >> "logs/$s.linux.log" 2>&1
  echo "$s exit=$? $(date +%H:%M:%S)" >> $log
done
echo "COMPLETE DONE $(date +%H:%M:%S)" >> $log
