__all__ = [
    'SegNet',
    'UNet',
    'UnetWrapper'
]

# Standard Imports
import os
import torch
from torch import nn
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

# Custom Imports
from oct_vesselseg import modules, utils
from oct_vesselseg.losses import DiceLoss
from oct_vesselseg import train
from oct_vesselseg.synth import ImageSynthEngineOCT, VesselLabelDataset


def _init_from_defaults(self, **kwargs):
    for key, value in self.defaults.__dict__.items():
        if key[0] != '_':
            kwargs.setdefault(key, value)
    for key, value in kwargs.items():
        setattr(self, key, value)


vesselseg_outdir = os.getenv("OCT_VESSELSEG_BASE_DIR")


class SegNet(nn.Sequential):
    """
    A generic segmentation network that works with any backbone
    """

    def __init__(self, ndim: int = 3, in_channels: int = 1,
                 out_channels: int = 1, kernel_size: int = 3,
                 activation: str = 'Softmax', backbone: str = 'UNet',
                 kwargs_backbone=None) -> nn.Sequential:
        """

        Parameters
        ----------
        ndim : {2, 3}
            Number of spatial dimensions
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output classes
        kernel_size : int, default=3
            Kernel size of the initial feature extraction layer
        activation : str, default='softmax'
            Final activation function
        backbone : {'UNet', 'MeshNet', 'ATrousNet'} or Module, default='UNet'
            Generic backbone module. Can be already instantiated.
        kwargs_backbone : dict, optional
            Parameters of the backbone (if backbone is not pre-instantiated)
        """
        if isinstance(backbone, str):
            backbone_kls = globals()[backbone]
            backbone = backbone_kls(ndim, **(kwargs_backbone or {}))
        if activation and activation.lower() == 'softmax':
            activation = nn.Softmax(1)
        feat = modules.ConvBlock(ndim,
                                 in_channels, backbone.in_channels,
                                 kernel_size=kernel_size,
                                 activation=None)
        pred = modules.ConvBlock(ndim,
                                 backbone.out_channels, out_channels,
                                 kernel_size=1,
                                 activation=activation)
        super().__init__(feat, backbone, pred)


class UNet(nn.Module):
    """A highly parameterized U-Net (encoder-decoder + skip connections)

    conv ------------------------------------------(+)-> conv
         -down-> conv ---------------(+)-> conv -> up
                     -down-> conv -> up

    Reference
    ---------
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI (2015)
    https://arxiv.org/abs/1505.04597
    """

    class defaults:
        nb_levels = 6            # number of levels
        nb_features = (16, 24, 32, 48, 64, 96, 128, 192, 256, 320)
        nb_conv = 2              # number of convolutions per level
        kernel_size = 3          # kernel size
        activation = 'ReLU'      # activation function
        norm = 'instance'        # 'batch', 'instance', 'layer', None
        dropout = 0              # dropout probability
        residual = False         # use residual connections throughout
        factor = 2               # change resolution by this factor
        use_strides = False      # use strided conv instead of linear resize
        order = 'cand'           # c[onv], a[ctivation], n[orm], d[ropout]
        combine = 'cat'          # 'cat', 'add'

    def _feat_block(self, i, o=None):
        opt = dict(activation=None, kernel_size=self.kernel_size,
                   order=self.order, residual=self.residual)
        return modules.ConvBlock(self.ndim, i, o, **opt)

    def _conv_block(self, i, o=None):
        opt = dict(activation=self.activation, kernel_size=self.kernel_size,
                   order=self.order, nb_conv=self.nb_conv, norm=self.norm,
                   residual=self.residual, dropout=self.dropout)
        return modules.ConvGroup(self.ndim, i, o, **opt)

    def _down_block(self, i, o=None):
        if self.use_strides:
            opt = dict(activation=self.activation, strides=self.factor,
                       order=self.order, kernel_size=self.factor,
                       norm=self.norm, dropout=self.dropout)
            return modules.StridedConvBlockDown(self.ndim, i, o, **opt)
        else:
            opt = dict(activation=self.activation, factor=self.factor,
                       order=self.order, kernel_size=1,
                       norm=self.norm, dropout=self.dropout)
            return modules.ConvBlockDown(self.ndim, i, o, **opt)

    def _up_block(self, i, o=None):
        if self.use_strides:
            opt = dict(activation=self.activation, strides=self.factor,
                       order=self.order, kernel_size=self.factor,
                       norm=self.norm, dropout=self.dropout,
                       combine=self.combine)
            return modules.StridedConvBlockUp(self.ndim, i, o, **opt)
        else:
            opt = dict(activation=self.activation, factor=self.factor,
                       order=self.order, kernel_size=1,
                       norm=self.norm, dropout=self.dropout,
                       combine=self.combine)
            return modules.ConvBlockUp(self.ndim, i, o, **opt)

    def __init__(self, ndim, **kwargs):
        super().__init__()
        _init_from_defaults(self, **kwargs)
        self.ndim = ndim
        self.nb_features = utils.ensure_list(self.nb_features, self.nb_levels)
        self.in_channels = self.out_channels = self.nb_features[0]

        # encoder
        self.encoder = [self._conv_block(self.nb_features[0])]
        for n in range(1, len(self.nb_features)-1):
            i, o = self.nb_features[n-1], self.nb_features[n]
            self.encoder += [modules.EncoderBlock(self._down_block(i, o),
                                                  self._conv_block(o))]
        i, o = self.nb_features[-2], self.nb_features[-1]
        self.encoder += [self._down_block(i, o)]
        self.encoder = nn.Sequential(*self.encoder)

        # decoder
        self.decoder = []
        for n in range(len(self.nb_features)-1):
            i, o = self.nb_features[-n-1], self.nb_features[-n-2]
            m = i
            if self.combine == 'cat' and n > 0:
                i *= 2
            self.decoder += [modules.DecoderBlock(self._conv_block(i, m),
                                                  self._up_block(m, o))]
        i, o = self.nb_features[0], self.nb_features[0]
        if self.combine == 'cat':
            i *= 2
        self.decoder += [self._conv_block(i, o)]
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        # check shape
        nb_levels = len(self.encoder)
        # if any(s < 2**nb_levels for s in x.shape[2:]):
        if torch.any(torch.tensor(x.shape[2:]) < 2**nb_levels):
            raise ValueError(f'UNet with {nb_levels} levels requires input '
                             f'shape larger or equal to {2**nb_levels}, but '
                             f'got {list(x.shape[2:])}')

        # compute downstream pyramid
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        # compute upstream pyramid
        x = skips.pop(-1)
        for n in range(len(self.decoder)-1):
            x = self.decoder[n].conv(x)
            x = self.decoder[n].up(x, skips.pop(-1))

        # highest resolution
        x = self.decoder[-1](x)

        return x


class UnetWrapper(nn.Module):
    """
    Base class for UNet.
    """

    def __init__(
        self,
        version_n: int,
        model_dir: str = 'models',
        synth_params: str = None,
        synth_dtype: torch.dtype = torch.float32,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        json_path: str = None,
    ):
        """
        Parameters
        ----------
        version_n : int
            Version of model.
        model_dir : str
            Subdirectory in which to store training versions.
        synth_params : dict
            Parameters for data synthesis.
        device : {'cuda', 'cpu'}
            Device to load UNet onto.
        """
        super().__init__()
        self.version_n = version_n
        self.model_dir = model_dir
        self.device = device
        self.synth_params = synth_params
        self.output_path = vesselseg_outdir
        self.version_path = f"{self.output_path}/{model_dir}"\
                            f"/version_{version_n}"
        self.json_path = (
            json_path
            if json_path is not None
            else f"{self.version_path}/json_params.json"
            )
        # Where to save model checkpoints during training
        self.checkpoint_dir = f"{self.version_path}/checkpoints"
        self.losses = {0: DiceLoss(labels=[1], activation='Sigmoid')}
        self.metrics = torch.nn.ModuleDict({'dice': self.losses[0]})
        self.synth_dtype = synth_dtype
        self.learning_rate = learning_rate
        self.to(self.device)

    def fix_state_dict(self, state_dict):
        """Adjust the keys by adding 'trainee.' prefix."""
        return {'trainee.' + key: value for key, value in state_dict.items()}

    def load(self, backbone_dict=None, augmentation=True, type='best',
             mode='train'):
        """
        Load a unet from checkpoint

        Parameters
        ----------
        type : {'best', 'last'}
            Which checkpoint to load from version directory.
        mode : {'train', 'test'}
            Whether model will be used for training or testing purposes.
        """
        # Loading the backbone of the model from json file
        if backbone_dict is None:
            # TODO: make this a logger instead of a print
            print('Loading backbone params from json...')
            self.backbone_dict = utils.JsonTools(self.json_path).read()
        else:
            self.backbone_dict = backbone_dict
        # Instantiating segmentation network
        self.segnet = SegNet(backbone='UNet', activation=None,
                             kwargs_backbone=self.backbone_dict
                             ).to(self.device)

        # Setting up augmentation
        if mode == 'train' and augmentation:
            if isinstance(augmentation, bool) and augmentation:
                print('Using standard augmentation.')
                augmentation = ImageSynthEngineOCT(self.synth_params)
            elif not isinstance(augmentation, torch.nn.Module):
                print('No valid augmentation provided, skipping augmentation.')
                augmentation = None

        # Configure trainee settings
        trainee_config = {
            'network': self.segnet,
            'loss': self.losses[0],
            'metrics': self.metrics,
            'augmentation': augmentation,
            'lr': self.learning_rate
        }

        # Load trainee
        checkpoint_path = utils.Checkpoint(self.checkpoint_dir).get(type)
        if mode == 'train':
            trainee = train.SupervisedTrainee(**trainee_config)
        elif mode == 'test':
            trainee_config['augmentation'] = None
            trainee = train.SupervisedTrainee(**trainee_config)

        # Load from checkpoint
        # TODO: Find out what I'm doing wrong in saving model weights (which
        # warrants this fixing of state dict)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            fixed_state_dict = self.fix_state_dict(checkpoint['state_dict'])
            trainee = train.FineTunedTrainee.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=self.device,
                trainee=train.SupervisedTrainee(
                    network=self.segnet,
                    loss=self.losses[0],
                    metrics=self.metrics,
                    augmentation=augmentation,
                    lr=self.learning_rate
                    ),
                strict=False
            )
            trainee.load_state_dict(fixed_state_dict, strict=True)
        self.trainee = trainee.to(self.device)

        return trainee

    def new(self, nb_levels=4, nb_features=[32, 64, 128, 256], dropout=0,
            nb_conv=2, kernel_size=3, activation='ReLU', norm='instance',
            augmentation=True):
        """
        nb_levels : int
            Number of convolutional levels for Unet.
        nb_features : list[int]
            Features per layer. len(list) must equal nb_levels.
        dropout : float
            Percent of data to be dropped randomly.
        nb_conv : int
            Number of convolutions per layer.
        kernel_size : int
            Size of convolutional window.
        activation : str
            Activation to be used for all filters.
        norm : str
            How to normalize layers.
        """
        backbone_dict = {
            "nb_levels": nb_levels,
            "nb_features": nb_features,
            "dropout": dropout,
            "nb_conv": nb_conv,
            "kernel_size": kernel_size,
            "activation": activation,
            "norm": norm,
            "residual": True
        }
        utils.PathTools(self.version_path).makeDir()
        utils.JsonTools(self.json_path).log(backbone_dict)
        self.trainee = self.load(backbone_dict, augmentation)

    def train_it(self,
                 synth_data_experiment_n,
                 synth_samples: int = -1,
                 training_steps: int = int(1e5),
                 train_to_val: float = 0.8,
                 batch_size: int = 1,
                 check_val_every_n_epoch: int = 1,
                 accumulate_gradient_n_batches: int = 1,
                 num_workers: int = 1
                 ):
        """
        Train unet after defining or loading model.

        Parameters
        ----------
        synth_data_experiment_n : int
            Dataset that will be used for training model.
        synth_samples : int
            Number of samples in the combined training and testing set.
            Default is -1 (all samples available).
        training_steps : int
            Number of training steps to take. Default is 100,000
        train_to_val : float
            Ratio of training data to validation data for training loop.
            Default is 0.8
        batch_size : int
            Number of volumes per batch. Default is 1.
        check_val_every_n_epoch : int
            Number of times validation dice score is calculated (expressed in
            epochs). Default is 1.
        accumulate_gradient_n_batches : int
            Number of batches to compute before stepping optimizer.
        num_workers : int
            Number of workers for torch dataloader. Default is 1.
        """
        self.synth_samples = synth_samples
        self.train_to_val = train_to_val
        self.n_train = synth_samples * train_to_val
        self.batch_size = batch_size
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.accumulate_gradient_n_batches = accumulate_gradient_n_batches
        self.num_workers = num_workers
        self.exp_path = (f'{vesselseg_outdir}/synthetic_data'
                         f'/exp{synth_data_experiment_n:04d}')
        self.epochs = int(
            (training_steps * self.batch_size) // self.n_train)

        print(f'Training for {self.epochs} epochs')

        # Init dataset
        dataset = VesselLabelDataset(
            inputs=f'{self.exp_path}/*label*', subset=self.synth_samples)
        # Splitting up train and val sets with specified random seed
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.val_set = random_split(
            dataset, [self.train_to_val, 1 - self.train_to_val], seed)
        # Logger and checkpoint stuff
        self.logger = TensorBoardLogger(
            self.output_path, self.model_dir, self.version_n)
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_metric_dice", mode="min",
            every_n_epochs=check_val_every_n_epoch,
            save_last=True, filename='{epoch}-{val_loss:.5f}')
        self.sequential_train()

    def sequential_train(self):
        """
        Train on single GPU.
        """
        torch.multiprocessing.set_start_method('spawn')
        # Setting up trainer
        trainer_ = Trainer(
            accelerator='gpu',
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            accumulate_grad_batches=self.accumulate_gradient_n_batches,
            logger=self.logger,
            callbacks=[self.checkpoint_callback,
                       LearningRateMonitor(logging_interval='step')],
            max_epochs=self.epochs,
        )
        # Begin training
        trainer_.fit(
            self.trainee,
            DataLoader(self.train_set, self.batch_size, shuffle=True,
                       num_workers=self.num_workers),
            DataLoader(self.val_set, self.batch_size, shuffle=False,
                       num_workers=self.num_workers)
        )
