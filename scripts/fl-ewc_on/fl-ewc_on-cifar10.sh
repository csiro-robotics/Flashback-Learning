#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export SEED=123


#Host CL Method: O-EWC
python utils/main.py --run original --seed $SEED --wandb_entity leilaa-mahmoudi \
    --note "mammoth online ewc" \
    --name "org-oewc-cifar10_${SEED}" \
    --model fl_ewc_on --e_lambda 10 --gamma 1 \
    --epoch_base 60 --sch0 0 \
    --epoch_cl 60 --sch 0 \
    --epoch_fl 0 --schf 0 \
    --dataset seq-cifar10 --batch_size 128 \
    --lr 0.2 --optim_mom 0 --optim_wd 0 

#Integration with FL: O-EWC+FL
python utils/main.py --run flashback --seed $SEED --alpha_p 0.1 --wandb_entity leilaa-mahmoudi \
    --note "mammoth online ewc" \
    --name "flbk-oewc-cifar10_${SEED}" \
    --model fl_ewc_on --e_lambda 15 --gamma 1 \
    --epoch_base 60 --sch0 0 \
    --epoch_cl 5 --sch 0 \
    --epoch_fl 55 --schf 0 \
    --dataset seq-cifar10 --batch_size 128 \
    --lr 0.03 --optim_mom 0 --optim_wd 0 
