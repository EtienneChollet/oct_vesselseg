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

This project focuses on generating synthetic datasets for training a 3D U-Net for the task of vasculature segmentation in OCT data. Using a cubic spline synthesis pipeline first established in [SynthSpline](https://github.com/balbasty/synthspline), and many data augmentation techniques from [Cornucopia](https://github.com/balbasty/cornucopia) this project employs domain-randomized synthesis to create structured labels, textures, and artifacts, enhancing the training of neural networks for vascular segmentation.

## Check out our papers!

📜 July 2024: [Long paper preprint](https://arxiv.org/abs/2407.01419v1)

📜 April 2024: [Short paper preprint from Medical Imaging with Deep Learning Conference](https://arxiv.org/abs/2405.13757v1)

![Pipeline](docs/pipeline.png "Synthesis, Training, and Inference Pipeline")

# Getting Started

Hard requirements include `Cppyy~=2.3` and `Python~=3.9`

## Installation

It is suggested that you create and activate a new mamba environment with python 3.9. You can learn how to install mamba by following the instructions provided in the [Miniforge repo](https://github.com/conda-forge/miniforge).

```bash
mamba create -n oct_tissuemasking python=3.9
mamba activate oct_tissuemasking
```

In order to synthesize vascular labels from splines, we will need to install the code from the synthspline repo.

```bash
pip install git+https://github.com/balbasty/synthspline.git#f78ba23
```

We need to identify and set our cuda version to make sure we install the right prebuilt wheel for cupy. You can find your cuda version by running the command `nvcc --version`.

```bash
export CUDA_VERSION=<cuda-version>
```

Finally, we can install oct_tissuemasking (this might take a while so let's set the default timeout to 20,000 seconds).

```bash
pip install oct_tissuemasking --default-timeout=20000
```

## Configuration

Identify the directory you want all oct_vesselseg related files to go into (you might need to make a new directory), and run the following command. This will add a line to your ~/.bashrc file to set a global environment variable `OCT_VESSELSEG_BASE_DIR`. I recommend this to be an empty directory (or if you specify a non-existent directory, one will be made for you).

```bash
oct_vesselsynth configure
```

# Usage

## Vessel Synthesis

Synthesize the vascular labels for training. You can get help for this command with the `--help` flag. The labels will go into the `OCT_VESSELSEG_BASE_DIR` directory in the subdirectory called `synthetic_data`.

```bash
oct_vesselseg vesselsynth
```

There are many different parameters you can specify with this command using these flags:

### `--shape`

Example: `oct_vesselseg vesselsynth --shape 128,128,128`

This is the shape of the volume (input data) to be synthesized. This does not need to be a perfect cube. This will also be the shape of the UNet you will train.

### `--voxel-size`
This is the spatial resolution of the data to be synthesized measured in units of $\frac{mm^3}{voxel}$. The default value is $0.02 \frac{mm^3}{voxel}$.

### `--tree-levels`
Sampler bounds for the number of hierarchical levels in the vascular tree. The level of a branch in a vascular tree refers to the distance (in terms of number of branches) from the root node to the node from which the branch in question originates.

* Root Branch (level 0): The root branch is at level 0.
* First Branch (level 1): The children of the root branch are at level 1.
* Second level (level 2): The grandchildren of the root are at level 2.

This quantity is sampled from a discrete uniform distribution for each tree that is created.

### `--tree-density`
Sampler bounds for the number of trees (or root points) per volume in units of $\frac{trees}{mm^3}$. This quantity is sampled from a uniform distribution for each volume that is created.

### `--tree-root-radius`

Sampler bounds for the radius of the first branch (root, Level 0). This quantity is sampled from a uniform distribution for each tree that is created.

### `--branch-tortuosity`
Sampler bounds for the tortuosity of a given branch. This quantity is sampled from a uniform distribution for each branch that is created. Tortuosity is defined as such:

$tortuosity = \frac{cord}{length}$

### `--branch-radius-ratio`
Sampler bounds for the ratio of the radius of the child branch compared to the radius of the parent branch. This is sampled from a uniform distribution for each child branch that is created.


### `--branch-radius-change`
Sampler bounds for a multiplicative variation in radius along the legth of a vessel. This is sampled from a uniform distribution for each branch.

### `--branch-children`
Sampler bounds for the number of children per parent. This is sampled from a discrete uniform distribution for each parent branch.

## OCT Image Synthesis

Synthesize some examples of synthetic OCT images.

```bash
oct_vesselseg imagesynth
```

## Training

Train the model on the vessel labels and on-the-fly OCT image synthesis. The models will go into a subdirectory of `OCT_VESSELSEG_BASE_DIR` called `models`.

```bash
oct_vesselseg train
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