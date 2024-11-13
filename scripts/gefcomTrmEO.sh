#!/usr/bin/env bash

source activate ts

gpu_id="3"
gpu_nums=1
cpu_nums=2
export CUDA_VISIBLE_DEVICES=${gpu_id}
export OMP_NUM_THREADS=${cpu_nums}

start_date=$(date "+%Y-%m-%d %H:%M:%S")

# w/o attack
for pred_len in 744 168 24; do
for e_layer in 6 3; do
for d_model in 2048; do
  for zone in CT MASS ME NEMASSBOST NH RI SEMASS TOTAL VT WCMASS; do
    model=TransformerEO
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
      --data_path=gefcom2017.csv --data=gefcom2017_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=1 --dec_in=1 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --zone=${zone} --model=${model} --embed=learned --dropout=0.3 \
      --e_layers=${e_layer} --d_model=512 --n_heads=8 --d_ff=2048 \
      --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --exp_out=gefcom2017/${model}Linear_single/${zone}
  done
done
done
done


# attack
#for pred_len in 24; do
#for e_layer in 3; do
#for increase in True; do
#for attack_form in normal; do
#for attack_rate in 0.4 0.7; do
#for param_a in 0.5; do
#for param_b in 1.0; do
#  for zone in CT MASS ME NEMASSBOST NH RI SEMASS TOTAL VT WCMASS; do
#    model=TransformerEO
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
#      --data_path=gefcom2017.csv --data=gefcom2017_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --e_layers=${e_layer} --d_model=512 --n_heads=8 --d_ff=2048 \
#      --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --zone=${zone} --model=${model} --embed=learned --dropout=0.3 \
#      --exp_out=gefcom2017Attack/Increase-${increase}_${attack_form}_k-${attack_rate}_a-${param_a}_b-${param_b}/${model}Linear/${zone} \
#      --attack_increase=${increase} --attack_form=${attack_form} --attack_rate=${attack_rate} \
#      --dist_param_a=${param_a} --dist_param_b=${param_b}
#done
#done
#done
#done
#done
#done
#done
#done


python /home2/rzhao/notify.py --t 48 --c ${PWD}/$(basename "$0")




echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"


