

import sys
from argparse import Namespace
from contextlib import suppress
from typing import List
import torch.nn.functional as F
from utils.distillation import pod

import torch
import torch.nn as nn
from torch.optim import SGD

from utils.conf import get_device
from utils.magic import persistent_locals

with suppress(ImportError):
    import wandb


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
        
        self.device = get_device()

        self.old_net = None

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        if 'wandb' in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            wandb.log({k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                      for k, v in locals.items() if k.startswith('_wandb_') or k.startswith('loss')})

        
    def meta_flashback(self,initial_teacher, primary_new_model,*args, **kwargs):
        ret = self.flashback(initial_teacher, primary_new_model,*args, **kwargs)
        return ret
    
    
    def get_old_opt(self):
        if self.task > 0 :
            opt_old = SGD(self.old_net.parameters(), lr=0.3, weight_decay=0, momentum=0)
        return opt_old
    
    def get_old_sch(self,opt_old, epochs_rd):
        if self.task > 0 :
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_old, [int(0.3*epochs_rd), int(0.6*epochs_rd)], gamma=0.1, verbose=False)
        return scheduler
    
    def get_old_scl(self):
        scaler = torch.cuda.amp.GradScaler()
        return scaler
    
    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.net.parameters():
            param.requires_grad = True
        
    def freeze_old(self):
        for param in self.old_net.parameters():
            param.requires_grad = False

    def unfreeze_old(self):
        for param in self.old_net.parameters():
            param.requires_grad = True


      
