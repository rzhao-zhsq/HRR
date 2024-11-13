#!/usr/bin/env bash

#while [ true ]
#do
#	flag=`ps -ef | grep gefcomTrm.sh | grep -v grep`
#	if [ -z "$flag" ];
#	then
#	  echo "shell run out, start new shell."
#	  break
#	else
#	  echo "shell running..."
#	  sleep 2
#	fi
#done

source activate ts
gpu_id="2"
gpu_nums=1
cpu_nums=2
export CUDA_VISIBLE_DEVICES=$gpu_id
export OMP_NUM_THREADS=${cpu_nums}

start_date=$(date "+%Y-%m-%d %H:%M:%S")
#ETT_DIR="/home2/rzhao/data/ts/ETT-small"



# w/o attack
for pred_len in 744 168 24; do
for station in 1 2 3 4 5 6 7 8 9 10 11; do
  for zone in 1 2 3 4 5; do
    model=Linear
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
      --data_path=gefcom2012.csv --data=gefcom2012_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=1 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=6 --d_model=64 \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --model=${model} --embed=learned --norm=none \
      --cols=drybulb_${station} \
      --exp_out=gefcom2012/${model}_single/${zone}_${station} &
  done
  wait

  for zone in 6 7 8 10 11; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
      --data_path=gefcom2012.csv --data=gefcom2012_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=1 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=6 --d_model=64 \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --model=${model} --embed=learned --norm=none \
      --cols=drybulb_${station} \
      --exp_out=gefcom2012/${model}_single/${zone}_${station} &
  done
  wait

  for zone in 12 13 14 15 16; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
      --data_path=gefcom2012.csv --data=gefcom2012_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=1 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=6 --d_model=64 \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --model=${model} --embed=learned --norm=none \
      --cols=drybulb_${station} \
      --exp_out=gefcom2012/${model}_single/${zone}_${station} &
  done
  wait

  for zone in 17 18 19 20 21; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
      --data_path=gefcom2012.csv --data=gefcom2012_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=1 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=6 --d_model=64 \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --model=${model} --embed=learned --norm=none \
      --cols=drybulb_${station} \
      --exp_out=gefcom2012/${model}_single/${zone}_${station} &
  done
  wait
done
done



## attack
#for pred_len in 744 168 24; do
#for increase in False True; do
#for attack_form in uniform; do
#for attack_rate in 0.7; do
#for param_a in -0.2; do
#for param_b in 0.8; do
#  for zone in 1 2 3 4 5; do
#    model=Linear
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
#      --data_path=gefcom2012.csv --data=gefcom2012_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=11 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --e_layers=6 --d_model=64 \
#      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --zone=${zone} --model=${model} --embed=learned --norm=none \
#      --exp_out=gefcom2012Attack/Increase-${increase}_${attack_form}_k-${attack_rate}_a-${param_a}_b-${param_b}/${model}/${zone} \
#      --attack_increase=${increase} --attack_form=${attack_form} --attack_rate=${attack_rate} \
#      --dist_param_a=${param_a} --dist_param_b=${param_b} &
#  done
#  wait
#
#  for zone in 6 7 8 10 11; do
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
#      --data_path=gefcom2012.csv --data=gefcom2012_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=11 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --e_layers=6 --d_model=64 \
#      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --zone=${zone} --model=${model} --embed=learned --norm=none \
#      --exp_out=gefcom2012Attack/Increase-${increase}_${attack_form}_k-${attack_rate}_a-${param_a}_b-${param_b}/${model}/${zone} \
#      --attack_increase=${increase} --attack_form=${attack_form} --attack_rate=${attack_rate} \
#      --dist_param_a=${param_a} --dist_param_b=${param_b} &
#  done
#  wait
#
#  for zone in 12 13 14 15 16; do
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
#      --data_path=gefcom2012.csv --data=gefcom2012_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=11 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --e_layers=6 --d_model=64 \
#      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --zone=${zone} --model=${model} --embed=learned --norm=none \
#      --exp_out=gefcom2012Attack/Increase-${increase}_${attack_form}_k-${attack_rate}_a-${param_a}_b-${param_b}/${model}/${zone} \
#      --attack_increase=${increase} --attack_form=${attack_form} --attack_rate=${attack_rate} \
#      --dist_param_a=${param_a} --dist_param_b=${param_b} &
#  done
#  wait
#
#  for zone in 17 18 19 20 21; do
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
#      --data_path=gefcom2012.csv --data=gefcom2012_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=11 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --e_layers=6 --d_model=64 \
#      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --zone=${zone} --model=${model} --embed=learned --norm=none \
#      --exp_out=gefcom2012Attack/Increase-${increase}_${attack_form}_k-${attack_rate}_a-${param_a}_b-${param_b}/${model}/${zone} \
#      --attack_increase=${increase} --attack_form=${attack_form} --attack_rate=${attack_rate} \
#      --dist_param_a=${param_a} --dist_param_b=${param_b} &
#  done
#  wait
#done
#done
#done
#done
#done
#done




python /home2/rzhao/notify.py --t 48 --c ${PWD}/$(basename "$0")
echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
