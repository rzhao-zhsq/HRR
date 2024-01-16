#!/usr/bin/env bash

source activate ts
gpu_id="5"
gpu_nums=1
cpu_nums=4
export CUDA_VISIBLE_DEVICES=$gpu_id
export OMP_NUM_THREADS=${cpu_nums}

start_date=$(date "+%Y-%m-%d %H:%M:%S")
#ETT_DIR="/home2/rzhao/data/ts/ETT-small"

for pred_len in 720 360 180; do
  python train.py --train \
    --root_path=/home/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
    --data_path=gefcom2017.csv --data=gefcom_reg --exp_out=gefcom --freq=h \
    --model=TransformerEO \
    --features=MS --target=demand --c_out=1 --enc_in=2 --lag=0 \
    --moving_avg=24 --seq_len=${pred_len} --label_len=0 --pred_len=${pred_len} \
    --e_layers=3 --d_model=512 --n_head=8 --d_ff=2048 \
    --dropout=0.3 \
    --early_stop_patience=3 --batch_size=64 --num_workers=${cpu_nums}
done

for pred_len in 720 360 180; do
  python train.py --train \
    --root_path=/home/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
    --data_path=gefcom2017.csv --data=gefcom_reg --exp_out=gefcom --freq=h \
    --model=TransformerEO \
    --revin \
    --features=MS --target=demand --c_out=1 --enc_in=2 --lag=0 \
    --moving_avg=24 --seq_len=${pred_len} --label_len=0 --pred_len=${pred_len} \
    --e_layers=3 --d_model=512 --n_head=8 --d_ff=2048 \
    --dropout=0.3 \
    --early_stop_patience=3 --batch_size=64 --num_workers=${cpu_nums}
done

echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
