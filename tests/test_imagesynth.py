import os
import torch
import shutil
import unittest
import numpy as np
import nibabel as nib
from oct_vesselseg.synth import ImageSynthEngineWrapper


class TestImageSynth(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a test directory and other setup."""
        cls.test_dir = 'output/test_synthetic_data/exp0001'
        os.makedirs(cls.test_dir, exist_ok=True)

        cls.synth_params = {
            "parenchyma": {
                "nb_classes": 5,
                "shape": 10
            },
            "gamma": [0.2, 2],
            "z_decay": [32],
            "speckle": [0.2, 0.8],
            "imax": 0.80,
            "imin": 0.01,
            "vessel_texture": True,
            "spheres": True,
            "slabwise_banding": True,
            "dc_offset": True
        }

        # Create an instance of ImageSynthEngineWrapper
        cls.synth = ImageSynthEngineWrapper(
            exp_path=cls.test_dir,
            label_type='label',
            synth_params=cls.synth_params
        )

    #@classmethod
    #def tearDownClass(cls):
    #    """Clean up the test directory after tests."""
    #    shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_output_directory_created(self):
        """Test if output directory has been created."""
        self.assertTrue(
            os.path.exists(self.synth.sample_nifti_dir),
            "Output directory for NIfTI samples was not created.")
        self.assertTrue(
            os.path.exists(self.synth.sample_fig_dir),
            "Output directory for figure samples was not created.")

    def test_synth_params(self):
        """Test if the synthesis parameters are set correctly."""
        self.assertEqual(
            self.synth.synth_params, self.synth_params,
            "Synthesis parameters do not match.")

    def test_generate_volumes(self):
        """
        Test if the correct number of volumes are created and saved as
        NIfTI files.
        """
        # Save some dummy vascular labels to test dir
        dummy_data = torch.zeros((64, 64, 64))
        dummy_data[10:20, ...] = 1
        dummy_data[20:30, ...] = 2
        dummy_data[30:40, ...] = 3
        dummy_data[40:50, ...] = 4

        for i in range(12):
            dummy_path = f'output/test_synthetic_data/exp0001/{i:04d}_vessels_label.nii.gz'
            nib.save(
                nib.nifti1.Nifti1Image(
                    dummy_data.to(torch.uint8).numpy(), affine=np.eye(4, 4)
                    ),
                filename=dummy_path)

        for i in range(10):
            self.synth.__getitem__(i, save_nifti=True, make_fig=True,
                                   save_fig=True)
            volume_file = f'{self.synth.sample_nifti_dir}/volume-{i:04d}.nii'
            mask_file = f'{self.synth.sample_nifti_dir}'\
                        f'/volume-{i:04d}_MASK.nii'

            print(volume_file)
            self.assertTrue(
                os.path.isfile(volume_file),
                f"Volume file {volume_file} was not created.")
            self.assertTrue(
                os.path.isfile(mask_file),
                f"Mask file {mask_file} was not created.")

    def test_volume_content(self):
        """Test if the volume contents are valid NIfTI files."""
        for i in range(10):
            self.synth.__getitem__(i, save_nifti=True, make_fig=False,
                                   save_fig=False)
            volume_file = f'{self.synth.sample_nifti_dir}/volume-{i:04d}.nii'
            mask_file = f'{self.synth.sample_nifti_dir}'\
                        f'/volume-{i:04d}_MASK.nii'
            img = nib.load(volume_file)
            mask = nib.load(mask_file)
            self.assertIsInstance(
                img, nib.Nifti1Image,
                f"{volume_file} is not a valid NIfTI image.")
            self.assertIsInstance(
                mask, nib.Nifti1Image,
                f"{mask_file} is not a valid NIfTI image.")

    def test_generate_figures(self):
        """Test if the figures are generated and saved correctly."""
        for i in range(2):
            self.synth.__getitem__(i, save_nifti=False, make_fig=True, save_fig=True)
            figure_file = f'{self.synth.sample_fig_dir}/volume-{i:04d}.png'
            self.assertTrue(os.path.isfile(figure_file), f"Figure file {figure_file} was not created.")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestImageSynth('test_generate_volumes'))
    #unittest.main()

    runner = unittest.TextTestRunner()
    runner.run(suite)
