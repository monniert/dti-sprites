from abc import ABCMeta

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset, ConcatDataset
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor, Compose, Resize

from utils import use_seed
from utils.path import DATASETS_PATH


VAL_SPLIT_RATIO = 0.1


class _AbstractTorchvisionDataset(TorchDataset):
    """_Abstract torchvision dataset"""
    __metaclass__ = ABCMeta
    root = DATASETS_PATH

    dataset_class = NotImplementedError
    name = NotImplementedError
    n_classes = NotImplementedError
    n_channels = NotImplementedError
    img_size = NotImplementedError  # Original img_size
    test_split_only = False
    n_samples = None

    def __init__(self, split, **kwargs):
        super().__init__()
        self.split = split
        self.eval_mode = kwargs.get('eval_mode', False)

        img_size = kwargs.get('img_size')
        if img_size is not None:
            self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
            assert len(img_size) == 2

        kwargs = {}
        if self.name in ['svhn']:
            kwargs['split'] = 'test'
        else:
            kwargs['train'] = False
        dataset = self.dataset_class(root=self.root, transform=self.transform, download=True, **kwargs)
        if self.n_samples is not None:
            assert self.n_samples < len(dataset)
            with use_seed(46):
                indices = np.random.choice(range(len(dataset)), self.n_samples, replace=False)
            dataset.data = dataset.data[indices]
            dataset.targets = dataset.targets[indices] if hasattr(dataset, 'targets') else dataset.labels[indices]

        if split == 'val':
            n_val = max(round(VAL_SPLIT_RATIO * len(dataset)), 100)
            if n_val < len(dataset):
                with use_seed(46):
                    indices = np.random.choice(range(len(dataset)), n_val, replace=False)

                dataset.data = dataset.data[indices]
                if hasattr(dataset, 'targets'):
                    dataset.targets = np.asarray(dataset.targets)[indices]
                else:
                    dataset.labels = np.asarray(dataset.labels)[indices]
        elif not self.test_split_only and self.n_samples is None:
            kwargs = {}
            if self.name in ['svhn']:
                kwargs['split'] = 'train'
            else:
                kwargs['train'] = True
            train_dataset = self.dataset_class(root=self.root, transform=self.transform, download=True, **kwargs)
            sets = [dataset, train_dataset]
            if self.name == 'svhn' and not self.eval_mode:
                sets.append(self.dataset_class(root=self.root, transform=self.transform, download=True, split='extra'))
            dataset = ConcatDataset(sets)

        self.dataset = dataset

    @property
    def transform(self):
        transform = []
        if self.img_size != self.__class__.img_size:
            transform.append(Resize(self.img_size))
        transform.append(ToTensor())
        return Compose(transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label


class SVHNDataset(_AbstractTorchvisionDataset):
    dataset_class = SVHN
    name = 'svhn'
    n_classes = 10
    n_channels = 3
    img_size = (32, 32)
