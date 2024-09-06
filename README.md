# oct_vesselseg

Characterized by minimal priors and high variance sampling, this project builds on the emerging field of synthesis-based training by proposing an entirely data-free synthesis engine for training a Unet in the task of vascular labeling in sOCT data (mus modality). This project employs domain-randomized synthesis to create structured vessel and neural parenchyma labels, textures, and artifacts. It creates volumetric imaging data *similar* to, but not emulative of 3D sOCT data, with corresponding (perfect) ground truth labels for vasculature. The package contains a training module employing an on-the-fly image synthesis procedure to create a virtually infinite number of unique volumetric training data.

# Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Configuration](#configuration)
- [Usage](#usage)
    - [Vessel Synthesis](#vessel-synthesis)
    - [OCT Image Synthesis](#image-synthesis)
    - [Training](#training)
    - [Inference](#inference)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

# Introduction

This project focuses on generating synthetic datasets for training a 3D U-Net in the task of vasculature segmentation in OCT data. Using a cubic spline synthesis pipeline first established in [SynthSpline](https://github.com/balbasty/synthspline), and many data augmentation techniques from [Cornucopia](https://github.com/balbasty/cornucopia), this project employs domain-randomized synthesis to create structured labels, textures, and artifacts, enhancing the training of neural networks for vascular segmentation.

## Check out our papers!

📜 July 2024: [Journal Article Preprint](https://arxiv.org/abs/2407.01419v1)

📜 April 2024: [Conference/Short paper from Medical Imaging with Deep Learning Conference](https://openreview.net/forum?id=j8v7qc5bof&referrer=%5Bthe%20profile%20of%20Etienne%20Chollet%5D(%2Fprofile%3Fid%3D~Etienne_Chollet1))

![Pipeline](docs/pipeline.png "Synthesis, Training, and Inference Pipeline")

# Getting Started

Hard requirements include `Cppyy~=2.3` and `Python~=3.9`

## Installation

It is suggested that you create and activate a new mamba environment with python 3.9. You can learn how to install mamba by following the instructions provided in the [Miniforge repo](https://github.com/conda-forge/miniforge).

```bash
mamba create -n oct_vesselseg python=3.9
mamba activate oct_vesselseg
```

In order to synthesize vascular labels from splines, we will need to install the code from the synthspline repo.

```bash
pip install git+https://github.com/balbasty/synthspline.git#f78ba23
```

We need to identify and set our cuda version to make sure we install the right prebuilt wheel for cupy. You can find your cuda version by running the command `nvcc --version` or `nvidia-smi`.

```bash
export CUDA_VERSION=<cuda-version>  # OR nvidia-smi
```

Finally, we can install oct_vesselseg from pypi.

```bash
pip install oct_vesselseg
```

## Configuration

You first need to determine the directory you want all oct_vesselseg related files to be stored. *PLEASE ENSURE IT IS A FULL (ABSOLUTE) PATH*. This directory is reccommended to be empty, or if you specify a non-existent directory, one will be created for you.

After determining this path, run the configuration subcommand of the recently installed `oct_vesselseg` package without any arguments (bare). Once run, you will receive a prompt to input the path. This subcommand adds a line to your ~/.bashrc file, which sets a global environment variable `OCT_VESSELSEG_BASE_DIR` to the desired path.

```bash
oct_vesselseg configure
```

After configuring `oct_vesselseg`, you may proceed to the useage section (below).

# Usage

## Vessel Synthesis

The `vesselsynth` command is used to generate synthetic vascular labels that may be used, in conjunction with artifact and noise models, to train a 3D U-Net. These labels will be saved in the `synthetic_data` directory within `OCT_VESSELSEG_BASE_DIR`. This engine uses cubic splines and randomized domain-specific parameters to create highly variable yet structured vascular geometries.

### Basic Usage
To generate a synthetic vascular dataset with default parameters, use:

```bash
oct_vesselseg vesselsynth
```

There are many different parameters you can specify with this command using these flags:

### Customizing the Vessel Label Synthesis Engine

You can customize the vessel label synthesis engine with several flags, each corresponding to a specific aspect of the geometry of the volume generated, a single vascular tree, or individual branches. Below is a summary of some of the most important flags

-  `--shape`: This is the shape of the volume (input data) to be synthesized. This does not need to be a perfect cube. This will also be the shape of the UNet you will train.

- `--voxel-size`: This is the spatial resolution of the data to be synthesized measured in units of $\frac{mm^3}{voxel}$. The default value is $0.02 \frac{mm^3}{voxel}$.

- `--tree-levels`: Sampler bounds for the number of hierarchical levels in the vascular tree. The level of a branch in a vascular tree refers to the distance (in terms of number of branches) from the root node to the node from which the branch in question originates. This quantity is sampled from a discrete uniform distribution for each tree that is created.
    - Root Branch (level 0): The root branch is at level 0.
    - First Branch (level 1): The children of the root branch are at level 1.
    - Second level (level 2): The grandchildren of the root are at level 2.

- `--tree-density`: Sampler bounds for the number of trees (or root points) per volume in units of $\frac{trees}{mm^3}$. This quantity is sampled from a uniform distribution for each volume that is created.

- `--tree-root-radius`: Sampler bounds for the radius of the first branch (root, Level 0). This quantity is sampled from a uniform distribution for each tree that is created.

- `--branch-tortuosity`: Sampler bounds for the tortuosity of a given branch. This quantity is sampled from a uniform distribution for each branch that is created. Tortuosity is defined as such:
    - $tortuosity = \frac{cord}{length}$

- `--branch-radius-ratio`: Sampler bounds for the ratio of the radius of the child branch compared to the radius of the parent branch. This is sampled from a uniform distribution for each child branch that is created.


- `--branch-radius-change`: Sampler bounds for a multiplicative variation in radius along the legth of a vessel. This is sampled from a uniform distribution for each branch.

- `--branch-children`: Sampler bounds for the number of children per parent. This is sampled from a discrete uniform distribution for each parent branch.

### Example
To generate a dataset with data of shape (256, 256, 256) voxels, a resolution of 0.65 $\frac{mm^3}{voxel}$, and highly tortuous vessels:

```bash
oct_vesselseg vesselsynth --shape 256 256 256 --voxel-size 0.03 --branch-tortuosity 4.0 5.0
```

## OCT Image Synthesis

The `imagesynth` command generates synthetic OCT images which are made on-the-fly during the training process. By saving them to `$OCT_VESSELSEG_BASE_DIR/synthetic_data/exp000*/sample_vols` with this command, this allows us to visualize our desired noise and artifact parameters before using them to train the network.

### Basic Usage

To generate OCT images with default settings, run:

```bash
oct_vesselseg imagesynth
```

### Customizing the Image Synthesis Engine
You can customize the synthesis process using the following parameter flags:

- `--data-experiment-n`: Specifies the vesselsynth data experiment number for loading volumetric dat.

- `--n-samples`: The number of synthetic OCT images to generate.

- `--parenchyma-classes`: The upper bound for the number of classes of parenchyma (neural tissue) in the image synthesis.

- `--parenchyma-shape`: The upper bound for the number of control points defining the shape of each parenchyma class.

- `--vessel-intensity`: The bounds for the attenuation of the vessel intensities when blending them onto the parenchyma. Lower magnitude results in vessels that are darker than their surrounding tissue/background. 

- `--vessel-texture`: Optionally apply intra-vascular textures and artifacts.

- `--image-gamma`: The bounds for non-linear contrast adjustment. Higher values increase contrast, while lower values decrease it.

- `--image-z-decay`: The upper bound for z decay, roughly approximating the banding artifact due to serial sectioning.

- `--image-speckle`: The bounds for speckle noise parameters.

- `--image-spheres`: Whether to add spherical artifacts to the image.

- `--image-banding`: Optionally to apply slabwise banding (z-decay) artifacts to the image.

- `--image-dc-offset`: Optionally add a small DC offset to the parenchyma tensor.

### Example Usage

Here's how you might use the `imagesynth` command to turn 20 vascular labels into volumes with 7 classes (distinct intensities) of neural parenchyma, vessels that are much darker than the background tissue, and with low overall contrast:

```bash
oct_vesselseg imagesynth --n-samples 20 --parenchyma-classes 7 --vessel-intensity 0.1 0.2 --image-gamma 0.5 0.75
```

## Training

Train the model on the vessel labels and on-the-fly OCT image synthesis. The models will go into a subdirectory of `OCT_VESSELSEG_BASE_DIR` called `models`.

```bash
oct_vesselseg train
```

### Customizing the Training Process

You can customize the training process using the following groups of parameter flags:

#### Model Flags
The following flags allow you to specify parameters regarding the model's architecture, directory name, and the experimental version:

- `--model-version-n`: Specifies the version number of the model to train. This will represent its own subdirectory within the directory specified by the following flag.

- `--model-dir`: String that specifies the directory where all model versions/experimental runs are stored.

- `--model-levels`: Number of levels (encoding/decoding block pairs) of the model.

- `--model-features`: List of integers defining the number of features in each corresponding layer of the model.

#### Training Flags

The following flags define parameters regarding the training procedure:

- `--training-lr`: Global learning rate that will be used for the majority of training (after lr warmup and before lr cooldown)

- `--training-train-to-val`: Float representing the ratio of training data samples to testing data samples (train:test).

- `--training-steps`: Number of times to step the optimizer (update the model's parameters) based on gradients calculated in backward pass.

- `--training-batch-size`: Number of samples per batch.

#### Data Flags
The following flags set specific parameters regarding the on-the-fly synthesis engine

- `--synth-data-experiment-n`: Specifies the vesselsynth data experiment number for loading volumetric dat.

- `--synth-samples`: Total number of synthetic OCT image samples to generate per epoch (late split into train and validation sets based on `--training-train-to-val` flag).

- `--synth-parenchyma-classes`: The upper bound for the number of classes of parenchyma (neural tissue) in the image synthesis.

- `--synth-parenchyma-shape`: The upper bound for the number of control points defining the shape of each parenchyma class.

- `--synth-image-gamma`: Bounds for global gamma exponential factor for contrast shrinking/stretching.

- `--synth-image-z-decay`: The upper bound for size of z decay, roughly approximating the banding artifact due to serial sectioning.

- `--synth-image-speckle`: Mean and standard deviation of multiplicative gamma noise.

- `--synth-image-spheres`: Whether to add spherical artifacts to the image.

- `--image-banding`: Optionally to apply slabwise banding (z-decay) artifacts to the image.

- `--image-dc-offset`: Optionally add a small DC offset to the parenchyma tensor.

### Example Usage

Here's how you might use the `train` command to train on 40 vascular labels and validate on 10 (using a train to validation ratio of 0.8):

```bash
oct_vesselseg imagesynth --n-samples 20 --parenchyma-classes 7 --vessel-intensity 0.1 0.2 --image-gamma 0.5 0.75
```


## Inference

Run inference on a compatable NIfTI file. Th

```bash
oct_vesselseg test --in-path <path-to-NIfTI>
```

# Results

Here we provide some examples of synthetic vasculature generated by this method:

![Results](docs/synth_samples.png "Samples of fully synthetic sOCT mus data.")

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements

- [SynthSeg](https://github.com/BBillot/SynthSeg), [SynthSpline](https://github.com/balbasty/synthspline), and [Cornucopia](https://github.com/balbasty/cornucopia), for the inspiration and methodological foundation.
- Much of the computation resources required for this research was performed on computational hardware generously provided by the [Massachusetts Life Sciences Center](https://www.masslifesciences.com/).
