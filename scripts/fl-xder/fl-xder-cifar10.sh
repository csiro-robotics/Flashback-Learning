#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export SEED=123


#Host CL Method: X-DER
python utils/main.py --run original --seed $SEED --wandb_entity leilaa-mahmoudi \
    --note "" \
    --name org-xder-cifar10_${SEED} \
    --model fl-xder --alpha 0.6 --beta 0.9 --m 0.2 --gamma 0.85 --lambd 0.05 --eta 0.01 \
    --epoch_base 60 --sch0 0 \
    --epoch_cl 60 --sch 1 --sch_ms "35 45"\
    --epoch_fl 0 --schf 0 \
    --dataset seq-cifar10 --batch_size 32 --minibatch_size 32 --simclr_batch_size 64 \
    --buffer_size 2000  \
    --lr 0.03 --optim_mom 0 --optim_wd 0 

#Integration with FL: X-DER+FL
 python utils/main.py --run flashback --alpha_p 0.001 --seed $SEED --wandb_entity leilaa-mahmoudi \
    --note "" \
    --name flbk-xder-cifar10_${SEED} \
    --model fl-xder --alpha 0.6 --beta 0.9 --m 0.2 --gamma 0.85 --lambd 0.05 --eta 0.01 \
    --epoch_base 60 --sch0 0 \
    --epoch_cl 10 --sch 0\
    --epoch_fl 50 --schf 0 --schf_ms "35 45" \
    --dataset seq-cifar10 --batch_size 32 --minibatch_size 32 --simclr_batch_size 64 \
    --buffer_size 2000  \
    --lr 0.03 --optim_mom 0 --optim_wd 0 