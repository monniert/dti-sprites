from abc import ABCMeta
from functools import lru_cache
from PIL import Image

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import ToTensor, Compose

from utils import coerce_to_path_and_check_exist, use_seed
from utils.path import DATASETS_PATH


class _AbstractMultiObjectDataset(TorchDataset):
    __metaclass__ = ABCMeta
    root = DATASETS_PATH
    name = NotImplementedError
    n_channels = 3
    n_classes = NotImplementedError
    img_size = NotImplementedError
    N = NotImplementedError
    instance_eval = True

    def __init__(self, split, **kwargs):
        self.data_path = coerce_to_path_and_check_exist(self.root / self.name)
        self.split = split
        self.eval_mode = kwargs.get('eval_mode', False) or split == 'test'
        self.eval_semantic = kwargs.get('eval_semantic', False)

        if self.eval_mode:
            self.size = 320
        elif split == 'val':
            with use_seed(42):
                self.val_indices = np.random.choice(range(self.N), 100, replace=False)
            self.size = 100
        else:
            self.size = self.N

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        path = self.data_path
        if self.split == 'val':
            idx = self.val_indices[idx]
        inp = self.transform(Image.open(path / 'images' / f'{idx}.png').convert('RGB'))
        if self.eval_semantic:
            label = (self.transform_gt(Image.open(path / 'sem_masks' / f'{idx}.png').convert('L')) * 255).long()
        else:
            label = (self.transform_gt(Image.open(path / 'masks' / f'{idx}.png').convert('L')) * 255).long()
        return inp, label

    @property
    @lru_cache()
    def transform(self):
        return Compose([ToTensor()])

    @property
    @lru_cache()
    def transform_gt(self):
        return Compose([ToTensor()])


class DSpritesGrayDataset(_AbstractMultiObjectDataset):
    name = 'dsprites_gray'
    img_size = (64, 64)
    N = 60000
    n_classes = 4


class CLEVR6Dataset(_AbstractMultiObjectDataset):
    name = 'clevr6'
    img_size = (128, 128)
    N = 34963
    n_classes = 7


class TetrominoesDataset(_AbstractMultiObjectDataset):
    name = 'tetrominoes'
    img_size = (35, 35)
    N = 60000
    n_classes = 20
