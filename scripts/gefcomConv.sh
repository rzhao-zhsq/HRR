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
gpu_id="3"
gpu_nums=1
cpu_nums=2
export CUDA_VISIBLE_DEVICES=$gpu_id
export OMP_NUM_THREADS=${cpu_nums}

start_date=$(date "+%Y-%m-%d %H:%M:%S")
#ETT_DIR="/home2/rzhao/data/ts/ETT-small"


# CT MASS ME NEMASSBOST NH RI SEMASS TOTAL VT WCMASS

# w/o attack
for pred_len in 24 168 744; do
for d_model in 64; do
for e_layers in 6; do
for norm in none; do
for kernel in 15; do #
  for zone in CT MASS ME NEMASSBOST NH; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
      --data_path=gefcom2017.csv --data=gefcom2017_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=${e_layers} --d_model=${d_model} \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --exp_out=gefcom2017/Conv/${zone} --model=Conv \
      --embed=learned --conv_head --norm=${norm} --kernel=${kernel} &
#      --residual
  done
  wait
  for zone in RI SEMASS TOTAL VT WCMASS; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
      --data_path=gefcom2017.csv --data=gefcom2017_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=${e_layers} --d_model=${d_model} \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --exp_out=gefcom2017/Conv/${zone} --model=Conv \
      --embed=learned --conv_head --norm=${norm} --kernel=${kernel} &
#      --residual
  done
  wait
done
done
done
done
done


# w/o attack
#for pred_len in 24 168 744; do
#for d_model in 64; do
#for e_layers in 6; do
#for norm in none; do
#for kernel in 21 17 3; do #
#  for zone in CT MASS ME NEMASSBOST NH; do
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
#      --data_path=gefcom2017.csv --data=gefcom2017_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --e_layers=${e_layers} --d_model=${d_model} \
#      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --zone=${zone} --exp_out=gefcom2017/Conv/${zone} --model=Conv \
#      --embed=learned --conv_head --norm=${norm} --residual --kernel=${kernel} &
#  done
#  wait
#
#  for zone in RI SEMASS TOTAL VT WCMASS; do
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
#      --data_path=gefcom2017.csv --data=gefcom2017_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --e_layers=${e_layers} --d_model=${d_model} \
#      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --zone=${zone} --exp_out=gefcom2017/Conv/${zone} --model=Conv \
#      --embed=learned --conv_head --norm=${norm} --residual --kernel=${kernel} &
#  done
#  wait
#done
#done
#done
#done
#done

# attack
#for pred_len in 744 168 24; do
#for kernel in 23 19 15 11 7; do # 15 7
#for increase in False True; do
#for attack_form in uniform; do
#for attack_rate in 0.7; do
#for param_a in -0.2; do
#for param_b in 0.8; do
#  for zone in CT MASS ME NEMASSBOST NH; do
#    model=Conv
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
#      --data_path=gefcom2017.csv --data=gefcom2017_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --zone=${zone} --model=${model} \
#      --e_layers=6 --d_model=64 --embed=learned --conv_head --norm=none --residual --kernel=${kernel} \
#      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --exp_out=gefcom2017Attack/Increase-${increase}_${attack_form}_k-${attack_rate}_a-${param_a}_b-${param_b}/${model}/${zone} \
#      --attack_rate=${attack_rate} --attack_increase=${increase} --attack_form=${attack_form} \
#      --dist_param_a=${param_a} --dist_param_b=${param_b} &
#  done
#  wait
#
#  for zone in RI SEMASS TOTAL VT WCMASS; do
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
#      --data_path=gefcom2017.csv --data=gefcom2017_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --zone=${zone} --model=${model} \
#      --e_layers=6 --d_model=64 --embed=learned --conv_head --norm=none --residual --kernel=${kernel} \
#      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --exp_out=gefcom2017Attack/Increase-${increase}_${attack_form}_k-${attack_rate}_a-${param_a}_b-${param_b}/${model}/${zone} \
#      --attack_rate=${attack_rate} --attack_increase=${increase} --attack_form=${attack_form} \
#      --dist_param_a=${param_a} --dist_param_b=${param_b} &
#  done
#  wait
#
#done
#done
#done
#done


python /home2/rzhao/notify.py --t 48 --c ${PWD}/$(basename "$0")
echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
