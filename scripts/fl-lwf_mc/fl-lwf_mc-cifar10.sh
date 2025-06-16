#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export SEED=369

#Host CL Method: LwF-MC
python utils/main.py --run original --seed $SEED --wandb_entity leilaa-mahmoudi\
    --note "" \
    --name lwf-org-cifar10_${SEED} \
    --model lwf_mc --wd_reg 5e-5\
    --epoch_base 60 --sch0 0 \
    --epoch_cl 60 --sch 0\
    --epoch_fl 0 --schf 0\
    --dataset seq-cifar10 --batch_size 32  \
    --lr 0.3 --optim_mom 0 --optim_wd 0 

#Integration with FL: LwF-MC+FL
python utils/main.py --run flashback --alpha_p 0.01 --seed $SEED --wandb_entity leilaa-mahmoudi\
    --note "" \
    --name lwf-flbk-cifar10_${SEED} \
    --model lwf_mc --wd_reg 5e-5\
    --epoch_base 60 --sch0 0 \
    --epoch_cl 10 --sch 0 \
    --epoch_fl 50 --schf 0 \
    --dataset seq-cifar10 --batch_size 32 \
    --lr 0.3 --optim_mom 0 --optim_wd 0 
