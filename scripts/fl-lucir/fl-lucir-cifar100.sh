#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export SEED=246

#SEED = 123 246 369


python utils/main.py --run original --seed $SEED --nowand 1 \
    --note "" \
    --name org-lucir-cifar100_${SEED} \
    --model fl-lucir --lr_finetune 0.01 --lamda_base 7 --k_mr 2 --fitting_epochs 20 --mr_margin 0.5 --lamda_mr 1.0 \
    --p_0 60 --sch0 1 \
    --n_epochs 60 --sch 0\
    --n_epochs_rd 0 --schf 0\
    --dataset seq-cifar100 --batch_size 128 --minibatch_size 128 --buffer_size 2000 \
    --lr 0.03 --optim_mom 0.9 --optim_wd 0 

#--wandb_entity leilaa-mahmoudi
python utils/main.py --run flashback --alpha_p 0.08 --seed $SEED --nowand 1 \
    --note "" \
    --name flbk-lucir-cifar100_${SEED} \
    --model fl-lucir --lr_finetune 0.01 --lamda_base 6 --k_mr 2 --fitting_epochs 20 --mr_margin 0.5 --lamda_mr 1.5 \
    --p_0 60 --sch0 1 \
    --n_epochs 10 --sch 0\
    --n_epochs_rd 50 --schf 1 --schf_ms "35 45"\
    --dataset seq-cifar100 --batch_size 128 --minibatch_size 128 --buffer_size 2000 \
    --lr 0.03 --optim_mom 0.9 --optim_wd 0 