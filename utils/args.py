# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--epoch_cl', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')
    #Add arguments for flashback
    parser.add_argument('--epoch_base', type=int, help='number of epochs for task0')
    parser.add_argument('--epoch_fl', type=int, help='number of epochs for flashback')
    # parser.add_argument('--n_epochs_f', type=int, help='number of epochs for final')
    parser.add_argument('--run', default= "original", type=str, help="run with flashback or original method?")
    parser.add_argument('--sch', default=0, choices=[0, 1], type=int, help='scheduler for original')
    parser.add_argument('--schf', default=0, choices=[0, 1], type=int, help='scheduler for flahback')
    parser.add_argument('--sch0', default=0, choices=[0, 1], type=int, help='scheduler for fisrt epoch')
    parser.add_argument('--sch_gamma', default=0.1, type=float, help='gamma scheduler for original')
    parser.add_argument('--schf_gamma', default=0.1, type=float, help='gamma scheduler for flashback')
    parser.add_argument('--sch0_gamma', default=0.1, type=float, help='gamma scheduler for first epoch')
    parser.add_argument('--sch_ms', default="42 54", type=str, help='milestone scheduler for flashback')
    parser.add_argument('--schf_ms', default="42 54", type=str, help='milestone scheduler for original')
    parser.add_argument('--sch0_ms', default="42 54", type=str, help='milestone scheduler for first epoch')
    parser.add_argument('--alpha_p', default=0.1, type=float, help='plasticity scaler')
    parser.add_argument('--name', default="test", type=str, help='wandb log name')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', default=0, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='regaz', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='mammoth', help='Wandb project name')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
