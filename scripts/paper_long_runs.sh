#!/bin/bash
# Paper, Table 2: long runs (10k steps, WikiText-103 bytes), one GPU per run. Run  python get_wikitext.py  first.
# Usage: bash scripts/paper_long_runs.sh <M|G1> <seed>
# Final number: the "BUYUK-EVAL" line. 1B takes ~1 h on an A100-40GB plus ~35-50 min compile.
set -e
cd "$(dirname "$0")/.."
cfg=${1:-M}; sd=${2:-0}
if [ "$cfg" = "G1" ]; then
  # 1.0B: hp0 19 -> 62.7 (cosine), slow second moment, softmax fix, neighbor pull, tail brake x2 over the last quarter.
  python -u hizli_act8.py --cfg G1 --nb 8 --phase all --stoch 2 --steps 10000 --eval_every 2500 --compile 1 --wscale 16 --fact 1 \
    --acc32 1 --data wiki --seed $sd --eval_final 100 --hp 19 --hp_end 62.7 --sched cos --vshift 7 --att_e 1 --nbr 3 \
    --tail_from 7500 --tail_mult 2
  python -u train_gpu.py --cfg G1 --steps 10000 --eval_every 2500 --modes adam32 --adam_cos 1 --adam_warm 100 --adam_beta2 0.95 \
    --clip 1.0 --adam_lr 4e-5 --relu_clip 2 --int_init 1 --data wiki --seed $sd --eval_final 100 --out /dev/null
else
  # 10.7M (M): as run in the paper (without the softmax fix, which was added later for 1B).
  python -u hizli_act8.py --cfg M --nb 8 --phase all --stoch 2 --steps 10000 --eval_every 2500 --compile 1 --wscale 16 --fact 1 \
    --acc32 1 --data wiki --seed $sd --eval_final 100 --hp 5 --hp_end 16.5 --sched cos --vshift 7 --nbr 3
  python -u train_gpu.py --cfg M --steps 10000 --eval_every 2500 --modes adam32 --adam_cos 1 --adam_warm 100 --adam_beta2 0.95 \
    --clip 1.0 --adam_lr 3.6e-4 --relu_clip 2 --int_init 1 --data wiki --seed $sd --eval_final 100 --out /dev/null
fi
