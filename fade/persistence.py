from pathlib import *
from functools import reduce
import torch

dir_data = "data/"
dir_parameters = "../parameters"
dir_visualization = "../runs"
folder_results = "results"
folder_settings = "settings"


def join_path(filepath):
    if isinstance(filepath, list):
        return reduce(lambda a,b: Path(a) / b, filepath)
    else:
        return Path(filepath)


def wrapper_join_path(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        args[0] = join_path(args[0])
        return func(*args, **kwargs)
    return wrapper


@wrapper_join_path
def path(filepath):
    return filepath


@wrapper_join_path
def name(filepath):
    return PurePath(filepath).name.split('.')[0]


@wrapper_join_path
def rename(filepath, name):
    filepath.rename(Path(name))


@wrapper_join_path
def write(filepath, content):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content)


@wrapper_join_path
def read(filepath):
    return filepath.read_text()


@wrapper_join_path
def append(filepath, content):
    with filepath.open("a") as f:
        f.write(content)


@wrapper_join_path
def iterdir(filepath):
    return filepath.iterdir()


@wrapper_join_path
def save(filepath, object):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(object, filepath)


@wrapper_join_path
def read(filepath, torch_load=False, map_location=torch.device('cpu')):
    if torch_load:
        return torch.load(filepath, map_location)
    else:
        with open(filepath) as f:
            return f.read()


@wrapper_join_path
def is_dir(filepath):
    return filepath.is_dir()
