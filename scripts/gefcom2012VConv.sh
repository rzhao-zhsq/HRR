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
gpu_id="0"
gpu_nums=1
cpu_nums=4
export CUDA_VISIBLE_DEVICES=$gpu_id
export OMP_NUM_THREADS=${cpu_nums}

start_date=$(date "+%Y-%m-%d %H:%M:%S")
#ETT_DIR="/home2/rzhao/data/ts/ETT-small"


# CT MASS ME NEMASSBOST NH RI SEMASS TOTAL VT WCMASS


for pred_len in 24 168 744; do
for d_model in 512 64; do
for d_layers in 1 2 3; do
for beta in 1e-4 1e-3; do
for combine_type in latent_residual latent_only; do
  e_layers=6
  norm=none
  for zone in 1 2 3 4 5; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
      --data_path=gefcom2012.csv --data=gefcom2012_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=11 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=${e_layers} --d_model=${d_model} \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --conv_head --embed=learned --norm=${norm} --residual \
      --exp_out=gefcom2012/VConv/${zone} --model=VConv \
      --conv_gaussian --d_layers=${d_layers} --combine_type=${combine_type} \
      --beta=${beta} --sample_size=1 --valid_temperature=0.0 &
  done
  wait

  for zone in 6 7 8 10 11; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
      --data_path=gefcom2012.csv --data=gefcom2012_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=11 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=${e_layers} --d_model=${d_model} \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --conv_head --embed=learned --norm=${norm} --residual \
      --exp_out=gefcom2012/VConv/${zone} --model=VConv \
      --conv_gaussian --d_layers=${d_layers} --combine_type=${combine_type} \
      --beta=${beta} --sample_size=1 --valid_temperature=0.0 &
  done
  wait

  for zone in 12 13 14 15 16; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
      --data_path=gefcom2012.csv --data=gefcom2012_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=11 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=${e_layers} --d_model=${d_model} \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --conv_head --embed=learned --norm=${norm} --residual \
      --exp_out=gefcom2012/VConv/${zone} --model=VConv \
      --conv_gaussian --d_layers=${d_layers} --combine_type=${combine_type} \
      --beta=${beta} --sample_size=1 --valid_temperature=0.0 &
  done
  wait

  for zone in 17 18 19 20 21; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCom2012/GEFCOM2012_Data/Load \
      --data_path=gefcom2012.csv --data=gefcom2012_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=11 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=${e_layers} --d_model=${d_model} \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --conv_head --embed=learned --norm=${norm} --residual \
      --exp_out=gefcom2012/VConv/${zone} --model=VConv \
      --conv_gaussian --d_layers=${d_layers} --combine_type=${combine_type} \
      --beta=${beta} --sample_size=1 --valid_temperature=0.0 &
  done
  wait

done
done
done
done
done
python /home2/rzhao/notify.py --t 48 --c ${PWD}/$(basename "$0")






echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
