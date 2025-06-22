import os
from setuptools import setup, find_packages

# Get CUDA version from the env variable, defaulting to '11.6' if not set
cuda_version = os.getenv('CUDA_VERSION', '11.6')

# Strip the period from the version string (e.g., '11.6' -> '116')
cuda_major_version = cuda_version.split('.')[0]

# Format version to get prebuilt wheel
cupy_package = f'cupy-cuda{cuda_major_version}x'

setup(
    name='oct_vesselseg',
    version='0.1.0',
    description='A Label-Free and Data-Free Synthesis Engine and Training Framework for Vascular Segmentation of sOCT Data with PyTorch.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Etienne Chollet & Yaël Balbastre',
    author_email='etiennepchollet@gmail.com',
    project_urls={
        "Conference Preprint": "https://arxiv.org/abs/2405.13757v1",
        "Paper Preprint": "https://arxiv.org/abs/2407.01419v1",
        "Source": "https://github.com/EtienneChollet/oct_vesselseg"
    },

    packages=find_packages(),
    entry_points={
            'console_scripts': [
                'oct_vesselseg=oct_vesselseg:main.app',
            ],
        },

    install_requires=[
        'cyclopts',
    ],

    extras_require={

        'client': [
            'fastapi',
            'python-multipart',
            'uvicorn',
            'requests'
        ],

        'server': [
            cupy_package,
            'cppyy==2.3',
            'torch',
            'torchvision',
            'torchaudio',
            'torchmetrics',
            'jitfields',
            'torch-interpol',
            'torch-distmap',
            'nibabel',
            'pytorch-lightning',
            'scikit-learn',
            'matplotlib',
            'tensorboard',
            'pandas',
            'cyclopts',
            'cornucopia',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],

    python_requires='~=3.9',
)
