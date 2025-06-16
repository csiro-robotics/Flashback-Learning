#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export SEED=246

#Host CL Method: iCaRL
python utils/main.py --run original --seed $SEED --wandb_entity leilaa-mahmoudi \
    --note "" \
    --name icarl-org-cifar10_${SEED} \
    --model fl_icarl \
    --epoch_base 60 --sch0 0 \
    --epoch_cl 60 --sch 0\
    --epoch_fl 0 --schf 0\
    --dataset seq-cifar10 --batch_size 512 --buffer_size 2000 --minibatch_size 256\
    --lr 0.3 --optim_mom 0 --optim_wd 1e-5 

#Integration with FL: iCaRL+FL
python utils/main.py --run flashback --alpha_p 0.01 --seed $SEED --wandb_entity leilaa-mahmoudi \
    --note "" \
    --name icarl-flbk-cifar10_${SEED} \
    --model fl_icarl \
    --epoch_base 60 --sch0 0 \
    --epoch_cl 10 --sch 0 \
    --epoch_fl 60 --schf 0 \
    --dataset seq-cifar10 --batch_size 512 --buffer_size 2000 --minibatch_size 256\
    --lr 0.3 --optim_mom 0 --optim_wd 1e-5 
