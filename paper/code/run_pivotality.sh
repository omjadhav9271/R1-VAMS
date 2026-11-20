#!/usr/bin/env bash
# Paper A pivotality experiment, detached and low priority; checkpoints per condition.
cd "$(dirname "$0")"
export WORKERS="${WORKERS:-20}" PYTHONIOENCODING=utf-8
echo "pivotality start $(date +%H:%M:%S)" >> logs/run_paperA.status
nice -n 10 python3 -u pivotality.py >> logs/pivotality.linux.log 2>&1
echo "pivotality exit=$? $(date +%H:%M:%S)" >> logs/run_paperA.status
