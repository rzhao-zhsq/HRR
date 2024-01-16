#!/usr/bin/env bash

source activate ts
gpu_id="6"
gpu_nums=1
export CUDA_VISIBLE_DEVICES=$gpu_id
#export OMP_NUM_THREADS=4

start_date=$(date "+%Y-%m-%d %H:%M:%S")
ETT_DIR="/home2/rzhao/data/ts/ETT-small"

for dw in 5.0; do
  for dm in bvc vanilla; do
    python run.py \
      --is_training=1 --root_path=/home/rzhao/data/ts/LSTF/ETT-small \
      --data_path=ETTm2.csv --task_id=ETTm2 --data=ETTm2 \
      --enc_in=7 --dec_in=7 --c_out=7 --factor=3 \
      --target=OT --features=M --model=RATransformer \
      --moving_avg=24 --seq_len=96 --label_len=48 --pred_len=720 \
      --e_layers=2 --d_layers=1 --des=Exp --d_model=512 --itr=1 --num_workers=0 \
      --sub_space --sub_space_residual --out_space --out_space_residual \
      --diversity_weight=${dw} --diversity_metric=${dm}
  done
done

echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
