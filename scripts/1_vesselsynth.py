import argparse


def main(args):
    import torch
    from core.synth import VesselSynthEngineWrapper, VesselSynthEngineOCT
    from synthspline.random import Uniform, RandInt

    exp = int(args.exp)
    shape = [int(s) for s in args.shape.split(',')]
    voxel_size = float(args.voxel_size)
    levels = [int(s) for s in args.levels.split(',')]
    density = [float(p) for p in args.density.split(',')]
    tortuosity = [float(t) for t in args.tortuosity.split(',')]
    root_radius = [float(r) for r in args.root_radius.split(',')]
    radius_ratio = [float(R) for R in args.radius_ratio.split(',')]
    radius_change = [float(a) for a in args.radius_change.split(',')]
    children = [int(c) for c in args.children.split(',')]
    device = args.device

    synth_params = {
        'shape': shape,
        'voxel_size': voxel_size,
        'nb_levels': RandInt(*levels),
        'tree_density': Uniform(*density),
        'tortuosity': Uniform(*tortuosity),
        'radius': Uniform(*root_radius),
        'radius_ratio': Uniform(*radius_ratio),
        'radius_change': Uniform(*radius_change),
        'nb_children': RandInt(*children),
        'device': device
        }

    torch.no_grad()
    synth_engine = VesselSynthEngineOCT(**synth_params)
    VesselSynthEngineWrapper(
        experiment_number=exp,
        synth_engine=synth_engine,
        ).synth()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize vessel labels.")

    parser.add_argument('-e', '--exp', type=str, default='1', required=True,
                        help='Experiment number for saving experimental data \
                            (in output directory).')
    parser.add_argument('-s', '--shape', type=str, default='128,128,128',
                        help='Shape of (three dimensional) volume of vascular \
                            labels to be made.')
    parser.add_argument('-v', '--voxel-size', type=str, default='0.02',
                        help='Physical size of each voxel (measured in mm^3).')
    parser.add_argument('-l', '--levels', type=str, default='1,4',
                        help='Sampler bounds for the maximum number of levels \
                            in a given vascular tree.')
    parser.add_argument('-p', '--density', type=str, default='0.1,0.2',
                        help='Density of tree root points. (roots per vol)')
    parser.add_argument('-t', '--tortuosity', type=str, default='1,5',
                        help='Sampler bounds for the tortuosity of vessels')
    parser.add_argument('-r', '--root-radius', type=str, default='0.1,0.15',
                        help='Sampler bounds for radius of tree roots.')
    parser.add_argument('-R', '--radius-ratio', type=str, default='0.25,1',
                        help="Sampler bounds for ")
    parser.add_argument('-a', '--radius-change', type=str, default='0.9,1.1',
                        help='Sampler bounds for the change in radius along \
                            the length of a single vessel segment.')
    parser.add_argument('-c', '--children', type=str, default='1,4',
                        help='Sampler bounds for the number of children from \
                            a single parent segment.')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device to preform computations')

    args = parser.parse_args()
    main(args)
