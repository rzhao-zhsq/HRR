#!/usr/bin/env bash

source activate ts
gpu_id="4"
gpu_nums=1
cpu_nums=4
export CUDA_VISIBLE_DEVICES=$gpu_id
export OMP_NUM_THREADS=${cpu_nums}

start_date=$(date "+%Y-%m-%d %H:%M:%S")
#ETT_DIR="/home2/rzhao/data/ts/ETT-small"

for seq_len in 720 360 180; do
  for pred_len in 720 360 180; do
    python train.py --train \
      --root_path=/home/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
      --data_path=gefcom2017.csv --data=gefcom_reg --exp_out=gefcom --freq=h \
      --model=Transformer \
      --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
      --moving_avg=24 --seq_len=${seq_len} --label_len=0 --pred_len=${pred_len} \
      --e_layers=3 --d_layers=3 --d_model=512 --n_head=8 --d_ff=2048 \
      --dropout=0.3 \
      --early_stop_patience=3 --batch_size=32 --num_workers=${cpu_nums}
  done
done

echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
