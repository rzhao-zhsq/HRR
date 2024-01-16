#!/usr/bin/env bash

source activate ts
gpu_id="4"
gpu_nums=1
cpu_nums=4
export CUDA_VISIBLE_DEVICES=${gpu_id}
export OMP_NUM_THREADS=${cpu_nums}

start_date=$(date "+%Y-%m-%d %H:%M:%S")
#ETT_DIR="/home2/rzhao/data/ts/ETT-small"

# DLinear
python train.py --train --root_path=/home/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
  --data_path=gefcom2017.csv --data=gefcom --exp_out=gefcom --freq=h \
  --model=DLinear \
  --features=MS --target=demand --enc_in=3 --lag=1464 \
  --moving_avg=24 --seq_len=360 --label_len=0 --pred_len=744 \
  --early_stop_patience=5 --batch_size=128 --num_workers=${cpu_nums}

# PatchTST
python train.py --train --root_path=/home/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
  --data_path=gefcom2017.csv --data=gefcom --exp_out=gefcom --freq=h \
  --model=PatchTST \
  --target=demand --features=MS --enc_in=3 --lag=1464 \
  --moving_avg=24 --seq_len=360 --label_len=0 --pred_len=744 \
  --e_layers=3 \
  --dropout=0.3 --fc_dropout=0.3 \
  --early_stop_patience=5 --batch_size=128 --num_workers=${cpu_nums}

# Transformer
python train.py --train \
  --root_path=/home/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
  --data_path=gefcom2017.csv --data=gefcom --exp_out=gefcom --freq=h \
  --model=Transformer \
  --features=MS --target=demand --c_out=3 --enc_in=3 --dec_in=3 --lag=1464 \
  --moving_avg=24 --seq_len=360 --label_len=180 --pred_len=744 \
  --e_layers=3 --d_layers=3 --d_model=512 --n_head=8 --d_ff=2048 \
  --dropout=0.3 \
  --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums}


# RATransformer
for dw in 5.0 4.0 1.0 0.5 0.1; do
  for dm in bvc; do
    python train.py --train --root_path=/home/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
      --data_path=gefcom2017.csv --data=gefcom --exp_out=gefcom --freq=h \
      --model=RATransformer \
      --target=demand --features=MS --c_out=3 --enc_in=3 --dec_in=3 --lag=1464 \
      --moving_avg=24 --seq_len=360 --label_len=180 --pred_len=744 \
      --e_layers=3 --d_layers=3 --d_model=512 --n_head=8 --d_ff=2048 \
      --dropout=0.3 \
      --early_stop_patience=5 --batch_size=16 --num_workers=${cpu_nums}
      --sub_space --sub_space_residual --out_space --out_space_residual \
      --diversity_weight=${dw} --diversity_metric=${dm}
  done
done

echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
