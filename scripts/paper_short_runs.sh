#!/bin/bash
# Paper, Table 1: short runs (2000 steps, Tiny Shakespeare), one GPU per run.
# Usage: bash scripts/paper_short_runs.sh <S|M|L|G1> <seed>
# Final number: the "BUYUK-EVAL" line (bits/char on 100 x 1024 fixed validation characters, same pieces for both methods).
set -e
cd "$(dirname "$0")/.."
cfg=${1:-S}; sd=${2:-0}
declare -A HP=([S]=3.6 [M]=3.6 [L]=8 [G1]=13.7)           # ZeroFloat initial threshold hp0 (cosine to 3.3 x hp0)
declare -A LR=([S]=3e-3 [M]=7.2e-4 [L]=2.9e-4 [G1]=8e-5)   # AdamW learning rate (best of a grid, short runs)
h=${HP[$cfg]}
python -u hizli_act8.py --cfg $cfg --nb 8 --phase all --stoch 2 --steps 2000 --eval_every 500 --compile 1 \
  --wscale 16 --vshift 5 --fact 1 --acc32 1 --att_e 1 --hp $h --hp_end $(python -c "print(round(3.3*$h,2))") --sched cos \
  --seed $sd --eval_final 100
# AdamW baseline as run in the paper's short runs (standard ReLU and PyTorch default initialization).
python -u train_gpu.py --cfg $cfg --steps 2000 --eval_every 500 --modes adam32 --adam_cos 1 --adam_warm 100 --adam_beta2 0.95 \
  --clip 1.0 --adam_lr ${LR[$cfg]} --seed $sd --eval_final 100 --out /dev/null
