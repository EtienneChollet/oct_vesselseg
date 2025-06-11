__all__ = [
    'PathTools',
    'Options',
    'JsonTools',
    'Checkpoint',
    'ensure_list',
    'make_vector',
    'quantile_norm',
]

# Standard imports
import os
import glob
import json
import torch
import shutil
from types import GeneratorType as generator


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


class Options(object):
    """
    Base class for managing and constructing file paths based on class
    attributes.

    Attributes
    ----------
    cls : object
        The class instance from which attributes are retrieved.
    attribute_dict : dict
        Dictionary of attributes from the class instance.
    out_dir : str
        Output directory for the prediction file.
    full_path : str
        Full file path for the prediction file.
    """

    def __init__(self, cls):
        """
        Initialize the Options class with the provided class instance.

        Parameters
        ----------
        cls : object
            The class instance from which attributes are retrieved.
        """
        self.cls = cls
        self.attribute_dict = self.cls.__dict__
        self.out_dir = None
        self.full_path = None

    def out_filepath(self, dir=None):
        """
        Construct the output file path based on class attributes.

        Parameters
        ----------
        dir : str, optional
            Directory to save the output file. If not provided, defaults to
            'predictions' subdirectory within the class instance's volume
            directory.

        Returns
        -------
        tuple
            A tuple containing the output directory and the full file path.
        """
        stem = f"{self.attribute_dict['tensor_name']}-prediction"
        stem += f"_stepsz-{self.attribute_dict['step_size']}"

        # Add accuracy details if available
        accuracy_name = self.attribute_dict.get('accuracy_name')
        accuracy_val = self.attribute_dict.get('accuracy_val')
        if accuracy_name and accuracy_val:
            stem += f"_{accuracy_name}-{accuracy_val}"

        stem += '.nii.gz'

        # Determine output directory
        if dir is None:
            self.out_dir = os.path.join(self.attribute_dict['volume_dir'],
                                        'predictions')
        else:
            self.out_dir = dir

        # Construct full file path
        self.full_path = os.path.join(self.out_dir, stem)

        return self.out_dir, self.full_path


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


def quantile_norm(
    input_tensor: torch.Tensor,
    pmin: float = 0.01,
    pmax: float = 0.99,
    vmin: float = 0.0,
    vmax: float = 1.0,
    sample_size: int = 10_000
) -> torch.Tensor:
    """
    Apply quantile-based normalization to a tensor.

    This function scales the input tensor so that the `pmin` and `pmax` 
    quantiles of the non-zero elements map linearly to `vmin` and `vmax`, 
    respectively. If the input contains only zeros, it returns a normalized 
    tensor of Gaussian noise.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to normalize. Can be any shape.
    pmin : float, optional
        The lower quantile to map to `vmin`. Should be in (0, 1). Default is 0.01.
    pmax : float, optional
        The upper quantile to map to `vmax`. Should be in (0, 1). Default is 0.99.
    vmin : float, optional
        The target value corresponding to the `pmin` quantile. Default is 0.0.
    vmax : float, optional
        The target value corresponding to the `pmax` quantile. Default is 1.0.
    sample_size : int, optional
        Number of samples to use for estimating quantiles. Default is 10,000.

    Returns
    -------
    torch.Tensor
        A tensor with the same shape and device as `input_tensor`, normalized.
    """
    flat = input_tensor.flatten()
    mask = flat != 0

    if mask.sum() == 0:
        noise = torch.randn_like(input_tensor)
        flat_noise = noise.flatten()

        n = flat_noise.numel()
        if n > sample_size:
            idx = torch.randint(0, n, (sample_size,), device=flat_noise.device)
            sampled = flat_noise[idx]
        else:
            sampled = flat_noise

        qmin = torch.quantile(sampled, pmin)
        qmax = torch.quantile(sampled, pmax)

        scale = (vmax - vmin) / (qmax - qmin)
        shift = vmin - qmin * scale

        return noise * scale + shift

    nonzero_vals = flat[mask]
    n = nonzero_vals.numel()
    if n > sample_size:
        idx = torch.randint(0, n, (sample_size,), device=flat.device)
        sampled = nonzero_vals[idx]
    else:
        sampled = nonzero_vals

    qmin = torch.quantile(sampled, pmin)
    qmax = torch.quantile(sampled, pmax)

    scale = (vmax - vmin) / (qmax - qmin)
    shift = vmin - qmin * scale

    return input_tensor * scale + shift
