#!/usr/bin/env bash

source activate ts
gpu_id="7"
gpu_nums=1
export CUDA_VISIBLE_DEVICES=$gpu_id
#export OMP_NUM_THREADS=4

start_date=$(date "+%Y-%m-%d %H:%M:%S")
ETT_DIR="/home/rzhao/data/ts/LSTF/ETT-small"

for dw in 0.1 0.3 0.5 2.0 5.0; do
  for dm in bvc vanilla; do
    for model in PatchTST_resemble PatchTST; do
      python ./train.py \
        --train --model=${model} --config=configs/default.yaml \
        --root_path=${ETT_DIR} --data_path=ETTh1.csv --data=ETTh1 \
        --enc_in=7 --dec_in=7 --c_out=7 --factor=3 \
        --e_layers=3 --d_layers=1 --moving_avg=24 --n_heads=4 --d_model=16 --d_ff=128 \
        --batch_size=128 --dropout=0.3 --fc_dropout=0.3 \
        --early_stop_patience=10 \
        --target=OT --features=M --seq_len=336 --label_len=48 --pred_len=720 \
        --num_workers=0 \
        --sub_space --sub_space_residual --out_space --out_space_residual \
        --diversity_weight=${dw} --diversity_metric=${dm}
    done
  done
done

echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
