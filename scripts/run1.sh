#!/usr/bin/env bash

source activate ts
gpu_id="1"
gpu_nums=1
export CUDA_VISIBLE_DEVICES=$gpu_id
#export OMP_NUM_THREADS=4

start_date=$(date "+%Y-%m-%d %H:%M:%S")
ETT_DIR="/home2/rzhao/data/ts/ETT-small"

for pred_len in 1 3 6 9; do
  for lag in 0 3 6 9 10; do
    for d_model in 512; do
      model_input=`expr ${lag} \* 1`
      if [ ${lag} -eq 0 ]; then model_input=1; fi
      python run.py \
        --is_training=1 --model=FEDformer \
        --root_path=/home2/rzhao/data/ts/spot \
        --data_path=WTI_spot_cleaned.csv --task_id=WTI --data=custom \
        --features=S --target=price --lag=${lag} \
        --moving_avg=4 --seq_len=12 --label_len=4 --pred_len=${pred_len} \
        --e_layers=2 --d_layers=1 --factor=3 \
        --enc_in=${model_input} --dec_in=${model_input} --c_out=1 \
        --d_model=512 --n_heads=8 --d_ff=2048 \
        --des=Exp --itr=1 --num_workers=4
    done
  done
done

echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
