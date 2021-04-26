import torch

from utils import coerce_to_path_and_check_exist
from .dti_sprites import DTISprites
from .tools import safe_model_state_dict


def get_model(name):
    return {
        'dti_sprites': DTISprites,
    }[name]


def load_model_from_path(model_path, dataset, device=None, attributes_to_return=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(coerce_to_path_and_check_exist(model_path), map_location=device.type)
    model_kwargs = checkpoint['model_kwargs']
    model = get_model(checkpoint['model_name'])(dataset, **model_kwargs)
    model = model.to(device)
    model.load_state_dict(safe_model_state_dict(checkpoint['model_state']))
    if hasattr(model, 'cur_epoch'):
        model.cur_epoch = checkpoint['epoch']
    if attributes_to_return is not None:
        if isinstance(attributes_to_return, str):
            attributes_to_return = [attributes_to_return]
        return model, [checkpoint.get(key) for key in attributes_to_return]
    else:
        return model
