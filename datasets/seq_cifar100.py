

from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18, tiny_vit
# from backbone.ResNet32 import resnet32
from PIL import Image
from torchvision.datasets import CIFAR100

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


class TCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_CLASSES_TASK_ZERO = 10
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])
    

# Define the data transformations for CIFAR-100
    # TRANSFORM = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize images to 224x224
    #     transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    #     transforms.RandomCrop(224, padding=4),  # Randomly crop images with padding
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # Normalize with CIFAR-100 stats
    #     ])

    def get_examples_number(self):
        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * (SequentialCIFAR100.N_TASKS-1) + SequentialCIFAR100.N_CLASSES_TASK_ZERO)
        # return tiny_vit(nclasses = SequentialCIFAR100.N_CLASSES_PER_TASK
        #                * (SequentialCIFAR100.N_TASKS-1) + SequentialCIFAR100.N_CLASSES_TASK_ZERO, model_name='vit_small_patch16_224' )

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        # transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # Resize images to 224x224
        # transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        # transforms.RandomCrop(224, padding=4),  # Randomly crop images with padding
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # Normalize with CIFAR-100 stats
        # ])
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        # transforms = DeNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        if args.sch:
            milestone = [int(x) for x in args.sch_ms.split()]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, milestone, gamma=args.sch_gamma, verbose=False)
        else:
            scheduler = None
        return scheduler

    @staticmethod
    def get_scheduler_f(model, args) -> torch.optim.lr_scheduler:
        if args.schf:
            milestone = [int(x) for x in args.schf_ms.split()]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, milestone, gamma=args.schf_gamma, verbose=False)
        else:
            scheduler = None
        return scheduler



