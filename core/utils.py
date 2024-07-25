__all__ = [
    'PathTools',
    'JsonTools',
    'Checkpoint',
    'ensure_list',
    'make_vector'
]

# Standard imports
import os
import glob
import json
import torch
import shutil
from types import GeneratorType as generator


# Third-party imports
# import numpy as np
# import torch
# from torchmetrics.functional import dice
# import torch.nn.functional as F
# from torch import nn
# import scipy.ndimage as ndimage
# from cornucopia.cornucopia import QuantileTransform


class PathTools(object):
    """
    Class to handle paths.
    """
    def __init__(self, path: str):
        """
        Parameters
        ----------
        path : str
            Path to deal with.
        """
        self.path = path

    def destroy(self):
        """
        Delete all files and subdirectories.
        """
        shutil.rmtree(path=self.path, ignore_errors=True)

    def makeDir(self):
        """
        Make new directory. Delete then make again if dir exists.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            try:
                self.destroy()
                os.makedirs(self.path)
            except OSError as e:
                print(f"Error creating directory {self.path}: {e}")

    def patternRemove(self, pattern):
        """
        Remove file in self.path that contains pattern

        Parameters
        ----------
        pattern : str
            Pattern to match to. Examples: {*.nii, *out*, 0001*}
        """
        regex = [
            f"{self.path}/**/{pattern}",
            f"{self.path}/{pattern}"
        ]
        for expression in regex:
            try:
                [os.remove(hit) for hit in glob.glob(
                    expression, recursive=True
                    )]
            except Exception as e:
                print(f'Error removing path: {e}')


class JsonTools(object):
    """
    Class for handling json files.
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            Path to json file.
        """
        self.path = path

    def log(self, dict):
        """
        Save Python dictionary as json file.

        Parameters
        ----------
        dict : dict
            Python dictionary to save as json.
        path : str
            Path to new json file to create.
        """
        if not os.path.exists(self.path):
            self.json_object = json.dumps(dict, indent=4)
            file = open(self.path, 'x')
            file.write(self.json_object)
            file.close()

    def read(self):
        f = open(self.path)
        dic = json.load(f)
        return dic


class Checkpoint(object):
    """
    Checkpoint handler.
    """
    def __init__(self, checkpoint_dir):
        """
        Parameters
        ----------
        checkpoint_dir : str
            Directory that holds checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_paths = glob.glob(f"{self.checkpoint_dir}/*")

    def best(self):
        """
        Return the first checkpoint that includes 'epoch=' in its filename, or
        None if no such file exists.

        Returns
        -------
        str or None
            The path to the best checkpoint, or None if no checkpoint matches.
        """
        # Find all files that include 'epoch=' in their filename
        hits = [hit for hit in self.checkpoint_paths if 'epoch=' in hit]
        # Return the first hit if available, otherwise None
        return hits[0] if hits else None

    def last(self):
        """
        Return the last checkpoint file, specifically named 'last.ckpt'.

        Returns
        -------
        str or None
            The path to the last checkpoint, or None if no such file exists.
        """
        hits = [hit for hit in self.checkpoint_paths if 'last.ckpt' in hit]
        return hits[0] if hits else None

    def get(self, type):
        """
        Retrieve a checkpoint based on a specified type ('best' or 'last').

        Parameters
        ----------
        type : str
            The type of checkpoint to retrieve ('best' or 'last').

        Returns
        -------
        str or None
            The path to the requested checkpoint, or None if no suitable
            checkpoint exists.
        """
        if type == 'best':
            return self.best()
        elif type == 'last':
            return self.last()


def ensure_list(x, size=None, crop=True):
    """Ensure that an object is a list (of size at last dim)

    If x is a list, nothing is done (no copy triggered).
    If it is a tuple, it is converted into a list.
    Otherwise, it is placed inside a list.
    """
    if not isinstance(x, (list, tuple, range, generator)):
        x = [x]
    elif not isinstance(x, list):
        x = list(x)
    if size and len(x) < size:
        x += x[-1:] * (size - len(x))
    if size and crop:
        x = x[:size]
    return x


def make_vector(input, n=None, crop=True, *args,
                dtype=None, device=None, **kwargs):
    """Ensure that the input is a (tensor) vector and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output vector.

    """
    input = torch.as_tensor(input, dtype=dtype, device=device).flatten()
    if n is None:
        return input
    if n is not None and input.numel() >= n:
        return input[:n] if crop else input
    if args:
        default = args[0]
    elif 'default' in kwargs:
        default = kwargs['default']
    else:
        default = input[-1]
    default = input.new_full([n-len(input)], default)
    return torch.cat([input, default])
