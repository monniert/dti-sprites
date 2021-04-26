from functools import lru_cache
from PIL import Image

from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, ToTensor

from utils import coerce_to_path_and_check_exist, get_files_from_dir
from utils.image import IMG_EXTENSIONS
from utils.path import DATASETS_PATH


class InstagramDataset(TorchDataset):
    root = DATASETS_PATH
    name = 'instagram'
    n_channels = 3
    inp_exts = IMG_EXTENSIONS
    img_size = (128, 128)

    def __init__(self, split, tag, **kwargs):
        self.data_path = coerce_to_path_and_check_exist(self.root / self.name / tag) / split
        self.split = split
        self.tag = tag
        try:
            input_files = get_files_from_dir(self.data_path, IMG_EXTENSIONS, sort=True)
        except FileNotFoundError:
            input_files = []
        self.input_files = input_files
        self.labels = [-1] * len(input_files)
        self.n_classes = 0
        self.size = len(self.input_files)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp = self.transform(Image.open(self.input_files[idx]).convert('RGB'))
        return inp, self.labels[idx]

    @property
    @lru_cache()
    def transform(self):
        return Compose([CenterCrop(self.img_size), ToTensor()])
