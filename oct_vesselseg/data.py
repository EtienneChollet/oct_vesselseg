__all__ = [
    'RealOctConfig'
    'RealOct',
    'RealOctPatchLoader',
    'RealOctPredict',
]

# Standard library imports
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import List

# Third-party imports
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Optional, Tuple

# Local application/library specific imports
from oct_vesselseg.utils import quantile_norm
from oct_vesselseg.attenuators import SinusoidalAttenuator
from oct_vesselseg.utils import Options
from oct_vesselseg.callbacks import InferenceETA


@dataclass
class RealOctConfig:
    """
    Configuration container for RealOct and its subclasses.

    input : Union[torch.Tensor, str]
        A tensor containing the entire dataset or a string path to a NIfTI
        file.
    patch_size : int
        Size of the patches into which the tensor is divided.
    binarize : bool, optional
        Indicates whether to binarize the tensor. If True,
        `binary_threshold` must be specified. Default is False.
    binary_threshold : float, optional
        The threshold value for binarization. Only used if `binarize` is
        True.
    normalize : bool, optional
        Specifies whether to normalize the tensor. Default is False.
    pad_it : bool, optional
        If True, the tensor will be padded using the method specified by
        `padding_method`. Default is False.
    padding_method : {'replicate', 'reflect', 'constant'}, optional
        Specifies the method to use for padding. Default is 'reflect'.
    device : {'cuda', 'cpu'}, optional
        The device on which the tensor is loaded. Default is 'cuda'.
    dtype : torch.dtype, optional
        The data type of the tensor when loaded into a PyTorch tensor.
        Default is `torch.float32`.
    """
    input: Union[torch.Tensor, str]
    patch_size: int = 128
    redundancy: int = 3
    binarize: bool = False
    binary_threshold: float = 0.5
    normalize: bool = False
    pad_it: bool = False
    padding_method: str = 'reflect'
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32


class RealOct(object):
    """
    Base class for volumetric sOCT data (mus).
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        config : RealOctConfig
            Configuration configuration container for RealOct and subclasses.

        Attributes
        ----------
        volume_nifti : nib.Nifti1Image or None
            Represents the NIfTI image of the volumetric data if loaded from a
            file, otherwise None.
        """
        self.config = config
        self.input = self.config.input
        self.patch_size = self.config.patch_size
        self.redundancy = self.config.redundancy - 1
        self.step_size = int(
            self.config.patch_size * (1 / (2**self.redundancy)))
        self.binarize = self.config.binarize
        self.binary_threshold = self.config.binary_threshold
        self.normalize = self.config.normalize
        self.pad_it = self.config.pad_it
        self.padding_method = self.config.padding_method
        self.device = self.config.device
        self.dtype = self.config.dtype
        self.tensor, self.nifti, self.affine = self.load_tensor()

    def load_tensor(self) -> Tuple[
        torch.Tensor, Optional[nib.Nifti1Image], Optional[np.ndarray]
    ]:
        """
        Loads and processes the input volume, applying normalization, padding,
        and binarization as specified.

        Returns
        -------
        torch.Tensor
            The processed tensor data, either loaded from a NIfTI file or
            received directly, and transformed to the specified device and
            dtype.
        Optional[nib.Nifti1Image]
            The original NIfTI volume if the input is a file path, otherwise
            None.
        Optional[np.ndarray]
            The affine transformation matrix of the NIfTI image if the input is
            a file path, otherwise None.

        Notes
        -----
        - If `input` is a path, the NIfTI file is loaded and the data is
        converted to the specified dtype and moved to the specified device.
        - Normalization rescales tensor values to the 0-1 range if `normalize`
        is True.
        - Padding adds specified borders around the data if `pad_it` is True.
        - Binarization converts data to 0 or 1 based on `binary_threshold` if
        `binarize` is True.
        """
        tensor, nifti, affine = None, None, None
        if isinstance(self.input, str):
            self.tensor_name = self.input.split('/')[-1].strip('.nii')
            base_name = self.input.strip('.nii')
            clean_name = base_name.strip(self.tensor_name).strip('/')
            self.volume_dir = f"/{clean_name}"
            nifti = nib.load(self.input)
            tensor = torch.from_numpy(nifti.get_fdata()).to(
                self.device, dtype=torch.float32)
            affine = nifti.affine
        else:
            tensor = self.input.to(self.device, self.dtype)

        if self.normalize:
            tensor = self.normalize_volume(tensor)
        if self.pad_it:
            tensor = self.pad_volume(tensor)
        if self.binarize:
            tensor = torch.where(
                tensor > self.binary_threshold,
                torch.tensor(1.0, device=self.device),
                torch.tensor(0.0, device=self.device)
            )
        return tensor, nifti, affine

    def normalize_volume(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize the tensor to the range 0 to 1.
        """
        tensor -= tensor.min()
        tensor /= tensor.max()
        return tensor

    def pad_volume(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies symmetric padding to a tensor to increase its dimensions,
        ensuring that its size is compatible with the specified `patch_size`.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to be padded.

        Returns
        -------
        torch.Tensor
            The tensor after applying symmetric padding.

        Notes
        -----
        Padding is added symmetrically to all dimensions of the input tensor
        based on half of the `patch_size`. The padding mode used is determined
        by the `padding_method` attribute, which can be 'replicate', 'reflect',
        or 'constant'.
        """
        # Ensures padding does not exceed tensor dimensions
        padded_tensor = torch.nn.functional.pad(
            input=tensor.unsqueeze(0),
            pad=[self.patch_size] * 6,
            mode=self.padding_method
        ).squeeze()
        return padded_tensor


class RealOctPatchLoader(RealOct, Dataset):
    """
    A subclass for loading 3D volume patches efficiently using PyTorch,
    optimized to work with GPU. It inherits from RealOct and Dataset, and it
    extracts specific patches defined by spatial coordinates.

    Parameters
    ----------
    config : RealOctConfig
        Configuration configuration container for RealOct and subclasses.
    """

    def __init__(self, config: RealOctConfig):
        """
        Initialize the loader, setting up the internal structure and
        computing the coordinates for all patches to be loaded.
        """
        RealOct.__init__(self, config)
        self.patch_coords()

    def __len__(self):
        """
        Return the total number of patches.
        """
        return len(self.complete_patch_coords)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve a patch by index.

        Parameters:
            idx (int): The index of the patch to retrieve.

        Returns:
            tuple: A tuple containing the patch tensor and its slice indices.
        """
        x_slice, y_slice, z_slice = self.complete_patch_coords[idx]
        patch = self.tensor[x_slice, y_slice, z_slice].detach().cuda()
        return patch, (x_slice, y_slice, z_slice)

    def patch_coords(self):
        """
        Computes the coordinates for slicing the tensor into patches based on
        the defined patch size and step size. This method populates the
        complete_patch_coords list with slice objects.
        """
        self.complete_patch_coords = []
        tensor_shape = self.tensor.shape
        x_coords = [slice(x, x + self.patch_size) for x in range(
            self.step_size, tensor_shape[0] - self.patch_size, self.step_size)]
        y_coords = [slice(y, y + self.patch_size) for y in range(
            self.step_size, tensor_shape[1] - self.patch_size, self.step_size)]
        z_coords = [slice(z, z + self.patch_size) for z in range(
            self.step_size, tensor_shape[2] - self.patch_size, self.step_size)]
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    self.complete_patch_coords.append((x, y, z))


class RealOctPredict(RealOctPatchLoader, Dataset):
    """
    A class for predicting on OCT data patches using a pre-trained model,
    optimized for GPU. It extends `RealOctPatchLoader` for loading and
    processing 3D volume patches for predictions.

    Inherits all initialization parameters from `RealOctPatchLoader`, which
    in turn inherits from `RealOct`.

    Parameters
    ----------
    config : RealOctConfig
        Configuration container for RealOct and all subclasses.
    trainee : Optional[torch.nn.Module], default None
        The model used for predictions. If provided, it must be a PyTorch
        model.
    normalize_patches : bool, optional, default True
        Whether to normalize patches before prediction.

    Attributes
    ----------
    imprint_tensor : torch.Tensor
        Stores accumulated prediction outputs for the entire volume.
    patch_weight : torch.Tensor
        A 3D tensor that applies a sine-weighted attenuation to the prediction
        outputs to smooth the transitions between patches.
    """

    def __init__(
        self,
        config: RealOctConfig,
        trainee: torch.nn.Module = None,
        normalize_patches: bool = True,
        callbacks: List = [],
        *args, **kwargs
    ):
        """
        Initialize the predictor, setting up the model and prediction
        containers.
        """
        RealOctPatchLoader.__init__(self, config)
        # Set the configuration for tensors
        self.backend = {'dtype': self.dtype, 'device': self.device}

        # Set the model (if provided) to evaluation mode
        self.trainee = trainee.eval() if trainee else None

        # Initialize the imprint tensor with zeros
        self.imprint_tensor = torch.zeros(self.tensor.shape, **self.backend)

        # Initialize weight tracker
        self.weight_tracker = torch.zeros(self.tensor.shape, **self.backend)

        # Set normalization flag
        self.normalize_patches = normalize_patches

        # Prepare the 3D sine-weighted attenuation kernel
        self.patch_attenuator = SinusoidalAttenuator(
            size=self.patch_size, dimensions=3
        )().cuda()

        self.callbacks = [
            InferenceETA(len(self)),
        ]
        self.callbacks.extend(callbacks)

    def __getitem__(self, idx: int):
        """
        Predict on a single patch, optimized for GPU.

        Parameters
        ----------
        idx : int
            Patch ID number to predict on. Adds attenuated predictions to
            self.imprint_tensor.
        """
        with torch.no_grad():
            # Retrieve the intensity patch and its coordinates
            patch, coords = super().__getitem__(idx)
            # Transfer to GPU if not already there. Useful for predicting
            # on large volumes that are on CPU.
            if self.backend['device'] != 'cuda':
                patch = patch.to('cuda')
            # Add batch and channel dimensions
            patch = patch.unsqueeze(0).unsqueeze(0)
            # Normalize the patch if the flag is set
            if self.normalize_patches is True:
                try:
                    if patch.numel() == 0:
                        print("Tensor is empty.")
                        #raise RuntimeError(
                        #    f"Empty patch at index {idx}, "
                        #    f"coordinates {self._patch_coords(idx)}"
                        #)
    
                        patch = torch.randn_like(patch)

                    patch = quantile_norm(
                        input_tensor=patch.float(), 
                        vmin=0.2, vmax=0.8
                    )

                except ValueError as e:
                    print(
                        f"ValueError: {e}. Quantile transform failed.")
                    patch -= patch.min()
                    patch /= patch.max()

            # Predict with model
            prediction = self.trainee(patch)
            # Apply sigmoid activation to logits
            prediction = torch.sigmoid(prediction).squeeze()
            # Weight the prediction by applying sine-weighted attenuation
            # prediction = torch.ones(128, 128, 128).cuda().float()
            weighted_prediction = (
                prediction * self.patch_attenuator
                )

            # Add attenuatied probabilities to imprint tensor
            self.imprint_tensor[
                coords[0], coords[1], coords[2]
                ] += weighted_prediction

            # Add attenuation signature to the weight tracker
            self.weight_tracker[
                coords[0], coords[1], coords[2]] += self.patch_attenuator

    def predict_on_all(self):
        """
        Predict on all patches in the parent volume.
        """
        if self.tensor.dtype != torch.float32:
            self.tensor = self.tensor.to(torch.float32)
            print('Input tensor needs to be float32!!')
        # Get the total number of patches in the parent volume
        n_patches = len(self)
        t0 = time.time()
        print('Starting predictions!!')

        # Loop through each patch and make predictions
        for i in range(n_patches):
            self[i]

            # Check for cancellation
            if any(getattr(cb, "should_stop", False) for cb in self.callbacks):
                for attr in list(vars(self).keys()):
                    setattr(self, attr, None)  # or: delattr(obj, attr)
                return

            callback_kwargs = {
                "step_num": i,
                "t0": t0
            }

            for callback in self.callbacks:
                callback.on_step(**callback_kwargs)

        # Remove padding from the imprint tensor
        s = slice(self.patch_size, -self.patch_size)
        self.imprint_tensor = self.imprint_tensor[s, s, s]
        self.weight_tracker = self.weight_tracker[s, s, s]
        # Calculate redundancy factor for averaging
        redundancy = ((self.patch_size ** 3) // (self.step_size ** 3))

        print(f"\n\n{redundancy}x Averaging...")

        # Average the imprint tensor to account for redundancy in
        # overlapped predictions
        self.imprint_tensor /= self.weight_tracker
        # self.imprint_tensor /= redundancy
        self.imprint_tensor = self.imprint_tensor.cpu().numpy()

    def save_prediction(self, dir=None):
        """
        Save prediction volume.

        Parameters
        ----------
        dir : str
            Directory to save volume. If None, it will save volume to
            same path as the parent volume.
        """
        # Determine the output file path
        self.out_dir, self.full_path = Options(self).out_filepath(dir)
        # Create the output directory if it doesn't exist
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"\nSaving prediction to {self.full_path}...")
        # Create nifti image from imprint tensor
        out_nifti = nib.nifti1.Nifti1Image(
            dataobj=self.imprint_tensor,
            affine=self.affine)
        # Save nifti
        nib.save(out_nifti, self.full_path)
