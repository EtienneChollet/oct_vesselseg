__all__ = [
    'PathTools',
    'JsonTools',
    'Checkpoint'
]

# Standard imports
import os
import sys
import glob
import json
import shutil

# Third-party imports
import numpy as np
import torch
from torchmetrics.functional import dice
import torch.nn.functional as F
from torch import nn
import scipy.ndimage as ndimage
from cornucopia.cornucopia import QuantileTransform


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
