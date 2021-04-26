from functools import lru_cache
from PIL import Image

from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import ToTensor, Compose, Resize

from utils import coerce_to_path_and_check_exist, get_files_from_dir
from utils.path import DATASETS_PATH


class WeizmannHorseDataset(TorchDataset):
    root = DATASETS_PATH
    name = 'weizmann_horse'
    n_channels = 3
    n_classes = 2
    img_size = (128, 128)
    seg_eval = True

    def __init__(self, split, **kwargs):
        self.split = split
        self.data_path = coerce_to_path_and_check_exist(self.root / self.name)
        self.input_files = get_files_from_dir(self.data_path / 'images', 'jpg', sort=True)
        self.label_files = get_files_from_dir(self.data_path / 'masks', 'png', sort=True)
        assert len(self.input_files) == len(self.label_files)
        self.size = 30 if self.split == 'val' else len(self.input_files)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp = self.transform(Image.open(self.input_files[idx]).convert('RGB'))
        label = self.transform_gt(Image.open(self.label_files[idx]))
        return inp, label

    @property
    @lru_cache()
    def transform(self):
        return Compose([Resize(self.img_size), ToTensor()])

    @property
    @lru_cache()
    def transform_gt(self):
        return Compose([Resize(self.img_size, interpolation=Image.NEAREST), ToTensor()])
