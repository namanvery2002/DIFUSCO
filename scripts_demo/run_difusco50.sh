#!/bin/bash
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate difusco
cd /home/naman/Pictures/diffusion/DIFUSCO
export PYTHONPATH="$PWD:$PYTHONPATH"
export WANDB_MODE=offline
CKPT=models/tsp50_concorde_1epoch/fm6khhos/checkpoints/last.ckpt
mkdir -p outputs_demo/large50
RES=outputs_demo/large50/difusco50_results.txt
: > "$RES"
for name in ulysses22 att48 st70 eil76 pr76 rat99 kroA100 pr124 pr136 ch150 kroA200 pr299 rd400 d493; do
  f=data/tsplib_large/${name}_concorde.txt
  log=outputs_demo/large50/test_${name}.log
  t0=$(date +%s.%N)
  python -u difusco/train.py \
    --task tsp --diffusion_type categorical --do_test \
    --ckpt_path "$CKPT" --storage_path "$PWD" \
    --training_split "$f" --validation_split "$f" --test_split "$f" \
    --wandb_logger_name tsp50_test50_${name} \
    --batch_size 1 --num_epochs 1 --num_workers 1 --validation_examples 1 \
    --diffusion_steps 1000 --inference_schedule cosine \
    --inference_diffusion_steps 50 --two_opt_iterations 1000 > "$log" 2>&1
  t1=$(date +%s.%N)
  rt=$(echo "$t1 - $t0" | bc)
  sc=$(grep -E "wandb:\s+test/solved_cost\s+[0-9]" "$log" | grep -oE "[0-9.]+$" | tail -1)
  gt=$(grep -E "wandb:\s+test/gt_cost\s+[0-9]" "$log" | grep -oE "[0-9.]+$" | tail -1)
  echo "$name solved=$sc gt=$gt runtime=$rt" | tee -a "$RES"
done
echo "ALL DONE" | tee -a "$RES"
