#!/usr/bin/env bash
# Prioriza las semillas 1 y 2 de Utonia (su semilla 0 dio el mejor resultado de prueba) antes que
# las semillas extra de PointNet++. Espera a que termine la evaluación de utonia_tiny_s0, detiene el
# guion maestro, entrena las semillas de Utonia y luego relanza run_all.sh (que retoma lo pendiente).
set -u
P="${PYTHON:-python}"   # intérprete con torch y point-transformer-v3
D=../data/processed
RAW=../data/3DAeroRelief
STATUS=../runs/logs/pipeline.log
say() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" | tee -a "$STATUS"; }

until grep -q "OK utonia_tiny_s0_fullres\|FALLO utonia_tiny_s0_fullres" "$STATUS"; do sleep 20; done
say "priorizando semillas de Utonia: se pausa el guion maestro"
powershell -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*run_all.sh*' -or \$_.CommandLine -like '*train.py*' -or \$_.CommandLine -like '*evaluate.py*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
sleep 5
rm -rf ../runs/pn2_s1   # corrida de PointNet++ recién iniciada e incompleta; se repetirá completa

for seed in 1 2; do
  run=utonia_tiny_s$seed
  if [ -f "../runs/$run/results.json" ]; then say "YA EXISTE $run"; continue; fi
  say "INICIO $run"
  if $P -u train_ptv3.py --data $D --out "../runs/$run" --model utonia_tiny --seed $seed --epochs 30 --iters 400 --bs 4 --val_every 3 > "../runs/logs/$run.log" 2>&1; then
    say "OK $run"
    say "INICIO ${run}_fullres"
    $P -u fullres.py --data $D --raw $RAW --preds "../runs/$run/test_tta_preds.npz" > "../runs/logs/${run}_fullres.log" 2>&1 && say "OK ${run}_fullres" || say "FALLO ${run}_fullres"
  else
    say "FALLO $run (ver runs/logs/$run.log)"
  fi
done

say "retomando el guion maestro (semillas de PointNet++)"
bash run_all.sh
