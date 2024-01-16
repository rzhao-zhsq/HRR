#!/usr/bin/env bash

source activate ts
gpu_id="3"
gpu_nums=1
cpu_nums=4
export CUDA_VISIBLE_DEVICES=$gpu_id
export OMP_NUM_THREADS=${cpu_nums}

start_date=$(date "+%Y-%m-%d %H:%M:%S")
#ETT_DIR="/home2/rzhao/data/ts/ETT-small"

for dw in 5.0; do
  python train.py --train --root_path=/home/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
    --data_path=gefcom2017.csv --data=gefcom --exp_out=gefcom --freq=h \
    --model=PatchTST \
    --target=demand --features=MS --enc_in=3 --lag=1464 \
    --moving_avg=24 --seq_len=360 --label_len=0 --pred_len=744 \
    --e_layers=3 \
    --dropout=0.3 --fc_dropout=0.3 \
    --early_stop_patience=5 --batch_size=128 --num_workers=${cpu_nums}
done

echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
