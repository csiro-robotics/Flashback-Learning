# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' online EWC.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--e_lambda', type=float, required=True,
                        help='lambda weight for EWC')
    parser.add_argument('--gamma', type=float, required=True,
                        help='gamma parameter for EWC online')

    return parser


class FLEwcOn(ContinualModel):
    NAME = 'fl_ewc_on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(FLEwcOn, self).__init__(backbone, loss, args, transform)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None
        self.checkpoint_pl = None
        self.fish_pl = None
        self.task = 0 
        self.dataset = get_dataset(args)
        self.tasks = self.dataset.N_TASKS
        self.cpt = self.dataset.N_CLASSES_PER_TASK
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)


    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            # print(penalty)
            return penalty

    def end_task(self, dataset):

        fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(dataset.train_loader) * self.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        self.task += 1

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        penalty = self.penalty()
        loss = self.get_loss(inputs, labels, self.task) + self.args.e_lambda * penalty
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()
    

    #Flashback 
    def get_fish_pl(self, dataset):

        fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(dataset.train_loader) * self.args.batch_size)

        self.fish_pl= fish
        self.checkpoint_pl = self.net.get_params().data.clone()

    def penalty_pl(self):
        if self.checkpoint_pl is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish_pl * ((self.net.get_params() - self.checkpoint_pl) ** 2)).sum()
            return penalty


    def flashback(self, initial_teacher, primary_new_model, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty_preserve = self.penalty()
        penalty_plasticity = self.penalty_pl()
        loss = self.get_loss(inputs, labels, self.task) + self.args.e_lambda * penalty_preserve
        loss += penalty_plasticity * self.args.alpha_p 
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int) -> torch.Tensor:
        

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        outputs = self.net(inputs)[:, pc:ac]
        targets = self.eye[labels][:, pc:ac]
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        assert loss >= 0

        return loss

