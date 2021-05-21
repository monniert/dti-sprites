from functools import lru_cache
from PIL import Image

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import Compose, Resize, ToTensor

from utils import coerce_to_path_and_check_exist, get_files_from_dir, use_seed
from utils.image import IMG_EXTENSIONS
from utils.path import DATASETS_PATH


class GTSRB8Dataset(TorchDataset):
    root = DATASETS_PATH
    name = 'gtsrb8'
    n_channels = 3
    n_classes = 8
    img_size = (28, 28)

    def __init__(self, split, **kwargs):
        self.data_path = coerce_to_path_and_check_exist(self.root / 'GTSRB')
        self.split = split
        input_files = get_files_from_dir(self.data_path / 'train', IMG_EXTENSIONS, sort=True, recursive=True)
        labels = [int(f.parent.name) for f in input_files]
        self.input_files = np.asarray(input_files)
        self.labels = np.asarray(labels)

        # We filter the dataset to keep 8 classes
        good_labels = {k: i for i, k in enumerate([3, 7, 9, 11, 17, 18, 25, 35])}
        mask = np.isin(self.labels, list(good_labels.keys()))
        self.input_files = self.input_files[mask]
        self.labels = np.asarray([good_labels[l] for l in self.labels[mask]])

        N = len(self.input_files)
        if split == 'val':
            with use_seed(46):
                indices = np.random.choice(range(N), 100, replace=False)
            self.input_files = self.input_files[indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp = self.transform(Image.open(self.input_files[idx]).convert('RGB'))
        return inp, self.labels[idx]

    @property
    @lru_cache()
    def transform(self):
        return Compose([Resize(self.img_size), ToTensor()])
