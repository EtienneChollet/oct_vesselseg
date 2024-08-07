import gc
import time
import argparse


def main(args):
    import torch
    from core.models import UnetWrapper
    from core.data import RealOctPredict
    version = args.version
    in_vol = args.in_vol
    patch_size = args.patch_size
    redundancy = args.redundancy
    checkpoint = args.checkpoint

    # Starting timer
    t1 = time.time()
    # Make the prediction without gradient computations
    with torch.no_grad():
        # Init the unet
        unet = UnetWrapper(
            version_n=version,
            model_dir='models',
            device='cuda'
            )

        # Loading model weights and setting to test mode
        unet.load(type=checkpoint, mode='test')

        prediction = RealOctPredict(
            input=in_vol,
            patch_size=patch_size,
            redundancy=redundancy,
            trainee=unet.trainee,
            pad_it=True,
            padding_method='reflect',
            normalize_patches=True,
            )

        prediction.predict_on_all()
        out_path = f"{unet.version_path}/predictions"
        prediction.save_prediction(dir=out_path)
        t2 = time.time()
        print(f"Process took {round((t2-t1)/60, 2)} min")
        print('#' * 30, '\n')
        del unet
        del prediction
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test UNet model that \
                                     has been trained in default models\
                                     dir.")

    parser.add_argument('-v', '--version', type=str, default='1',
                        help='Model version to test (default: 1). Can test many versions seperated by commas.')
    parser.add_argument('-i', '--in-vol', type=str, required=False,
                        default='/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.nii',
                        help='Path to NIfTI data. Can test many input files seperated by commas.')
    parser.add_argument('-p', '--patch-size', type=int, default=128,
                        help='Size of the Unet input layer and desired patch size (default: 128).')
    parser.add_argument('-r', '--redundancy', type=int, default=3,
                        help='Redundancy factor for prediction overlap (default: 3).')
    parser.add_argument('-c', '--checkpoint', type=str, default='best',
                        help='Checkpoint to load. "best" or "last".')

    args = parser.parse_args()
    main(args)
