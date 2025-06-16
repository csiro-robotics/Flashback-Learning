#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export SEED=369


#Host CL Method: LUCIR
python utils/main.py --run original --seed $SEED --wandb_entity leilaa-mahmoudi \
    --note "" \
    --model fl-lucir --lr_finetune 0.01 --lamda_base 7 --k_mr 2 --fitting_epochs 20 --mr_margin 0.5 --lamda_mr 1. \
    --name org-lucir-cifar10-5tasks_${SEED} \
    --epoch_base 60 --sch0 1 \
    --epoch_cl 60 --sch 1 \
    --epoch_fl 0 --schf 0 \
    --dataset seq-cifar10 --batch_size 256 --minibatch_size 256 --buffer_size 2000 \
    --lr 0.05 --optim_mom 0.9 --optim_wd 0 

#Integration with FL: LUCIR+FL
python utils/main.py --run flashback --alpha_p 0.01 --seed $SEED --wandb_entity leilaa-mahmoudi \
    --note "" \
    --model fl-lucir --lr_finetune 0.02 --lamda_base 5 --k_mr 2 --fitting_epochs 20 --mr_margin 0.5 --lamda_mr 1.5 \
    --name flbk-lucir-cifar10-5tasks_${SEED} \
    --epoch_base 60 --sch0 1 \
    --epoch_cl 10 --sch 0\
    --epoch_fl 60 --schf 1\
    --dataset seq-cifar10 --batch_size 256 --minibatch_size 256 --buffer_size 2000 \
    --lr 0.05 --optim_mom 0.9 --optim_wd 0 

 