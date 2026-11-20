#!/usr/bin/env bash
# 2026-11-20: after paperB_complete finishes, swap in _staged_2026-11-20/ (whitewashing engine,
# tau=1000 grid point, n0 sweep), verify the new engine's defaults are bitwise identical to the
# old one, fill the new grid point, then paperB_final/robust --complete and paperB_attack.
# Originals are kept in _pre_2026-11-20/. Stops at the first failure.
cd "$(dirname "$0")"
export WORKERS="${WORKERS:-20}" PYTHONIOENCODING=utf-8
log=logs/run_paperB.status
say(){ echo "$* $(date +%H:%M:%S)" >> $log; }
step(){ say "$1 start"; nice -n 10 python3 -u "$@" > "logs/${1%.py}${2:+_${2#--}}.linux.log" 2>&1; rc=$?; say "$1 $2 exit=$rc"; [ $rc -eq 0 ] || { say "REST ABORT"; exit 1; }; }
until grep -q "COMPLETE DONE" $log; do sleep 20; done
grep -q "paperB_complete exit=0" $log || { say "REST ABORT: complete run failed"; exit 1; }
mkdir -p _pre_2026-11-20
cp engine_v.py paperB_complete.py paperB_final.py paperB_attack.py paperB_complete_results.json _pre_2026-11-20/
# bitwise check in a throwaway dir: new engine (engine_v) vs old (engine_orig)
T=$(mktemp -d); cp _staged_2026-11-20/engine_v.py stats_util.py xcheck_bitwise.py $T/; cp engine_v.py $T/engine_orig.py
(cd $T && nice -n 10 python3 xcheck_bitwise.py) > logs/xcheck_bitwise.log 2>&1 || { say "REST ABORT: bitwise check failed"; exit 1; }
rm -rf $T; say "bitwise ok"
cp _staged_2026-11-20/*.py .
step paperB_complete.py
step paperB_final.py --complete
step paperB_robust.py --complete
step paperB_attack.py
say "REST DONE"
