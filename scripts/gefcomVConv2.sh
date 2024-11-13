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
cpu_nums=4
export CUDA_VISIBLE_DEVICES=$gpu_id
export OMP_NUM_THREADS=${cpu_nums}

start_date=$(date "+%Y-%m-%d %H:%M:%S")
#ETT_DIR="/home2/rzhao/data/ts/ETT-small"


# CT MASS ME NEMASSBOST NH RI SEMASS TOTAL VT WCMASS

for pred_len in 24 168 744; do
  for zone in CT MASS ME NEMASSBOST NH RI SEMASS TOTAL VT WCMASS; do
    for d_layers in 1 2 3; do
      for combine_type in latent_only latent_residual; do
        for beta in 1e-4 1e-3; do
          python train.py --train \
            --root_path=/home2/rzhao/data/ts/GEFCOM/GEFCOM-2017-luo \
            --data_path=gefcom2017.csv --data=gefcom_reg \
            --freq=h --features=MS --target=demand --c_out=1 --enc_in=2 --dec_in=2 --lag=0 \
            --seq_len=0 --label_len=0 --pred_len=${pred_len} \
            --e_layers=6 --d_layers=${d_layers} --d_model=64 \
            --dropout=0.3 --early_stop_patience=5 --batch_size=32 --num_workers=${cpu_nums} \
            --conv_head --embed=learned --residual \
            --zone=${zone} --exp_out=gefcom2017/VConvMC/${zone} --model=VConv \
            --conv_gaussian \
            --combine_type=${combine_type} --beta=${beta} --sample_size=100 --valid_temperature=1.0 &
        done
      wait
      done
    done
  done
done
python /home2/rzhao/notify.py --t 48 --c ${PWD}/$(basename "$0")


echo "Start  date: $start_date"
echo "Finish date: $(date "+%Y-%m-%d %H:%M:%S")"
