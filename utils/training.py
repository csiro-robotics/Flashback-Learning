import math
import sys
from argparse import Namespace
from typing import Tuple
from copy import deepcopy
import os

import time
import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar

try:
    import wandb
except ImportError:
    wandb = None



def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    
    if k == 0:
        outputs[:, dataset.N_CLASSES_TASK_ZERO:
               (dataset.N_TASKS-1) * dataset.N_CLASSES_PER_TASK + dataset.N_CLASSES_TASK_ZERO] = -float('inf')
    else:
        outputs[:, 0:(k-1) * dataset.N_CLASSES_PER_TASK + dataset.N_CLASSES_TASK_ZERO] = -float('inf')
        outputs[:, (k) * dataset.N_CLASSES_PER_TASK + dataset.N_CLASSES_TASK_ZERO:
               (dataset.N_TASKS-1) * dataset.N_CLASSES_PER_TASK + dataset.N_CLASSES_TASK_ZERO] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity , config=vars(args), mode = "offline", id = args.name)
        wandb.define_metric("task")
        wandb.define_metric("RESULT_class_mean_accs", step_metric = "task")
        wandb.define_metric("RESULT_task_mean_accs", step_metric = "task")
        for i in range(dataset.N_TASKS):
            wandb.define_metric(f"RESULT_class_acc_{i}",step_metric = "task")
        for i in range(dataset.N_TASKS):
            wandb.define_metric(f"RESULT_task_acc_{i}",step_metric = "task")


        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):

        start_time = time.time()

        model.net.train()

        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        if t==0:
            epochs = args.epoch_base
            if args.sch0:
                milestone = [int(x) for x in args.sch0_ms.split()]
                scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, milestone, gamma=args.sch0_gamma, verbose=False)
            else:
                scheduler = None
        else:
            epochs = args.epoch_cl
            scheduler = dataset.get_scheduler(model, args)

        
        initial_teacher = deepcopy(model.old_net)
        for epoch in range(epochs):
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

        if (model.args.run == "flashback" and t > 0) :

            model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)

            if model.NAME == "ewc_on":
                model.get_fish_pl(dataset)
            
        
            primary_new_model = deepcopy(model.net) 

            for param in primary_new_model.parameters():
                param.requires_grad = False

            for param in  initial_teacher.parameters():
                param.requires_grad = False

            model.net.load_state_dict(deepcopy(model.old_net.state_dict()))
            
            
            if model.NAME == "lucir":
                model.net.classifier.task -= 1
                model.begin_flback(dataset)

            scheduler = dataset.get_scheduler_f(model, args)

      
            for epoch in range(model.args.epoch_fl):
                for i, data in enumerate(train_loader):
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_flashback(initial_teacher, primary_new_model, inputs, labels, not_aug_inputs)
                    assert not math.isnan(loss)
                    progress_bar.prog(i, len(train_loader), epoch, t, loss)

                if scheduler:
                    scheduler.step()

        

            

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        end_time = time.time()
        training_time = end_time - start_time 
        print(f"Task training time :{training_time:.2f} seconds")
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        # old_acc = np.mean([accs[0][:t], accs[1][:t]], axis=1)
        # new_acc = np.mean([[accs[0][t]],[accs[1][t]]], axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
        # print_mean_accuracy(old_acc, t + 1, dataset.SETTING)
        # print_mean_accuracy(new_acc, t + 1, dataset.SETTING)
        

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:

            d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])},
                'task': t+1}

            wandb.log(d2)

    checkpoint_path = os.path.join(".", args.name + ".pth")


    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
