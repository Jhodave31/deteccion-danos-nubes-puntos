#!/usr/bin/env bash
# Guion maestro de la entrega final: corre todos los experimentos en cadena.
# Cada paso escribe en runs/<nombre>/ y su salida en runs/logs/<nombre>.log.
# Si un paso falla se registra y se continúa con el siguiente.
# Un paso ya terminado (con results.json) no se repite.
#
# Uso (desde la carpeta code/):  bash run_all.sh
set -u
P="${PYTHON:-python}"   # intérprete con torch y point-transformer-v3
D=../data/processed
RAW=../data/3DAeroRelief
mkdir -p ../runs/logs
STATUS=../runs/logs/pipeline.log
say() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" | tee -a "$STATUS"; }

step() {  # step <nombre> <comando...>
  local name=$1; shift
  say "INICIO $name"
  if "$@" > "../runs/logs/$name.log" 2>&1; then say "OK $name"; else say "FALLO $name (ver runs/logs/$name.log)"; fi
}

train_ptv3() {  # train_ptv3 <run> <modelo> <semilla>
  local run=$1 model=$2 seed=$3
  if [ -f "../runs/$run/results.json" ]; then say "YA EXISTE $run"; return; fi
  step "$run" $P -u train_ptv3.py --data $D --out "../runs/$run" --model "$model" --seed "$seed" \
       --epochs 30 --iters 400 --bs 4 --val_every 3
  [ -f "../runs/$run/test_tta_preds.npz" ] && step "${run}_fullres" $P -u fullres.py --data $D --raw $RAW --preds "../runs/$run/test_tta_preds.npz"
}

train_pn2() {  # train_pn2 <run> <semilla>
  local run=$1 seed=$2
  if [ -f "../runs/$run/results.json" ]; then say "YA EXISTE $run"; return; fi
  step "$run" $P -u train.py --data $D --out "../runs/$run" --block 12 --epochs 40 --iters 400 --bs 16 --val_every 4 --seed "$seed"
  [ -f "../runs/$run/best.pt" ] && step "${run}_eval" $P -u evaluate.py --data $D --raw $RAW --run "../runs/$run" --tta --fullres
}

say "===== comienza el guion maestro ====="

# E0: línea base PointNet++ del segundo avance (TTA + resolución completa)
[ -f ../runs/pn2/eval_tta_full.json ] || step pn2_eval $P -u evaluate.py --data $D --raw $RAW --run ../runs/pn2 --tta --fullres

# E1: PTv3 desde cero / E2: PTv3 preentrenado con Sonata
train_ptv3 ptv3_s0 ptv3 0
train_ptv3 sonata_s0 sonata_small 0

# E3: dos semillas más del mejor modelo PTv3 según la VALIDACIÓN (Área 8)
BEST=$($P -c "
import json
r={m: json.load(open(f'../runs/{d}/results.json'))['best_val_mIoU'] for m,d in (('ptv3','ptv3_s0'),('sonata_small','sonata_s0')) if __import__('os').path.exists(f'../runs/{d}/results.json')}
print(max(r, key=r.get) if r else 'sonata_small')")
say "mejor modelo PTv3 en validación: $BEST"
TAG=$([ "$BEST" = "ptv3" ] && echo ptv3 || echo sonata)
train_ptv3 ${TAG}_s1 "$BEST" 1
train_ptv3 ${TAG}_s2 "$BEST" 2

# E2b: Utonia (exploratorio) y semillas extra de PointNet++
train_ptv3 utonia_tiny_s0 utonia_tiny 0
train_pn2 pn2_s1 1
train_pn2 pn2_s2 2

say "===== guion maestro terminado ====="
