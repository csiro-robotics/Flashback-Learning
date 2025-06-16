

#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export SEED=123

#SEED=123 246 369

#original lwf-mc on split cifar100
python utils/main.py --run original --seed $SEED --nowand 1 \
    --note "" \
    --name lwf-org-cifar100_${SEED} \
    --model fl-lwf_mc --wd_reg 2e-5\
    --p_0 60 --sch0 0 \
    --n_epochs 60 --sch 0\
    --n_epochs_rd 0 --schf 0\
    --dataset seq-cifar100 --batch_size 32 \
    --lr 0.3 --optim_mom 0 --optim_wd 0 

# --wandb_entity leilaa-mahmoudi
#flashback lwf-mc on split cifar100
python utils/main.py --run flashback --alpha_p 0.01 --seed $SEED --nowand 1 \
    --note "" \
    --name lwf-flk-cifar100_${SEED} \
    --model fl-lwf_mc --wd_reg 3e-5\
    --p_0 60 --sch0 0 \
    --n_epochs 10 --sch 0 \
    --n_epochs_rd 50 --schf 0\
    --dataset seq-cifar100 --batch_size 32 \
    --lr 0.3 --optim_mom 0 --optim_wd 0 