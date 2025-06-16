#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export SEED=123

#Host CL Method: iCaRL
python utils/main.py --run original --seed $SEED --wandb_entity leilaa-mahmoudi \
    --note "" \
    --name icarl-org-cifar100_${SEED} \
    --model fl_icarl \
    --epoch_base 60 --sch0 0 \
    --epoch_cl 60 --sch 0\
    --epoch_fl 0 --schf 0\
    --dataset seq-cifar100 --batch_size 32 --buffer_size 2000 \
    --lr 0.3 --optim_mom 0 --optim_wd 1e-5 

#Integration with FL: iCaRL+FL    
python utils/main.py --run flashback --alpha_p 0.01 --seed $SEED --wandb_entity leilaa-mahmoudi\
    --note "" \
    --name icarl-flbk-cifar100_${SEED} \
    --model fl_icarl \
    --epoch_base 60 --sch0 0 \
    --epoch_cl 10 --sch 0 \
    --epoch_fl 50 --schf 1 --schf_ms "35 45"\
    --dataset seq-cifar100 --batch_size 32 --buffer_size 2000 \
    --lr 0.3 --optim_mom 0 --optim_wd 1e-5 