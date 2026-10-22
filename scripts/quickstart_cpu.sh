#!/bin/bash
# Quickstart (CPU, about 2 minutes): train the small model (S, 0.8M parameters) on Tiny Shakespeare
# with ZeroFloat (integer-only) and with AdamW, and print validation bits/char for both.
# This is a smoke test with few steps; it is not one of the paper's results.
set -e
cd "$(dirname "$0")/.."
STEPS=${1:-1000}
echo "== ZeroFloat (integer-only), S, $STEPS steps"
python -u hizli_act8.py --cfg S --nb 8 --phase all --stoch 2 --steps $STEPS --eval_every $((STEPS / 3)) \
  --wscale 16 --vshift 5 --fact 1 --acc32 1 --att_e 1 --hp 3.6 --hp_end 11.88 --sched cos --eval_final 20 \
  | grep -E "adım|BUYUK"
echo "== AdamW (fp32), same model and data, $STEPS steps"
python -u train_gpu.py --cfg S --steps $STEPS --eval_every $((STEPS / 3)) --modes adam32 --adam_cos 1 --adam_warm 100 \
  --adam_beta2 0.95 --clip 1.0 --adam_lr 3e-3 --eval_final 20 --out /dev/null | grep -E "adım|BUYUK"
