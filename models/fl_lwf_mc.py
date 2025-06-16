# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via iCaRL.')

    add_management_args(parser)
    add_experiment_args(parser)
    
    parser.add_argument('--wd_reg', type=float, required=True,
                        help='L2 regularization applied to the parameters.')
    return parser


class FLLwFMC(ContinualModel):
    NAME = 'fl_lwf_mc'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(FLLwFMC, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.class_means = None
        self.old_net = None
        self.current_task = 0
        self.task = 0
        self.cpt = self.dataset.N_CLASSES_PER_TASK

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        if self.current_task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs))
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels, self.current_task, logits)
        loss.backward()

        self.opt.step()

        return loss.item()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.
        :param inputs: the images to be fed to the network
        :param labels: the ground-truth labels
        :param task_idx: the task index
        :return: the differentiable loss value
        """

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        outputs = self.net(inputs)[:, :ac]

        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0
       
        if self.args.wd_reg:
            loss += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)

        return loss

    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        self.current_task += 1
        self.task += 1

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    def flashback(self, initial_teacher, primary_new_model, inputs, labels, not_aug_inputs):

        pc = (self.task -1 ) * self.dataset.N_CLASSES_PER_TASK + self.dataset.N_CLASSES_TASK_ZERO
        ac = (self.task) * self.dataset.N_CLASSES_PER_TASK + self.dataset.N_CLASSES_TASK_ZERO

        outputs = self.net(inputs)

      
        with torch.no_grad():
            logits_pr = torch.sigmoid(initial_teacher(inputs)[:, :pc])
            logits_pl = torch.sigmoid(primary_new_model(inputs)[:, pc:ac])


        targets = self.eye[labels][:, pc:ac]
        comb_targets = torch.cat((logits_pr, targets), dim=1)
 
        
        self.opt.zero_grad()

        loss_preserve = F.binary_cross_entropy_with_logits(outputs[:, :ac], comb_targets)
        loss_preserve =  loss_preserve 

        if self.args.wd_reg:
            loss_preserve += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)

        loss_plasticity = F.binary_cross_entropy_with_logits(outputs[:, pc:ac], logits_pl)
        loss_plasticity = loss_plasticity * self.args.alpha_p 
       
        loss = loss_preserve + loss_plasticity

        loss.backward()

        self.opt.step()

        return loss.item()
