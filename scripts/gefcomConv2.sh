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


for pred_len in 24 168 744; do
for d_model in 16 32 64 128 256; do
for e_layers in 9 12; do
for norm in none; do
for kernel in 23 19 15 11 7; do #
#  for zone in CT MASS ME NEMASSBOST NH; do
#    python train.py --train \
#      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
#      --data_path=gefcom2017.csv --data=gefcom_reg \
#      --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
#      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
#      --e_layers=${e_layers} --d_model=${d_model} \
#      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
#      --zone=${zone} --exp_out=gefcom2017/Conv/${zone} --model=Conv \
#      --embed=learned --conv_head --norm=${norm} --residual --kernel=${kernel} &
#  done
#  wait

  for zone in RI SEMASS TOTAL VT WCMASS; do
    python train.py --train \
      --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
      --data_path=gefcom2017.csv --data=gefcom_reg \
      --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
      --seq_len=0 --label_len=0 --pred_len=${pred_len} \
      --e_layers=${e_layers} --d_model=${d_model} \
      --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
      --zone=${zone} --exp_out=gefcom2017/Conv/${zone} --model=Conv \
      --embed=learned --conv_head --norm=${norm} --residual --kernel=${kernel} &
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
