import unittest
import os
import shutil
import json
import nibabel as nib
from oct_vesselseg.synth import VesselSynthEngineWrapper


class TestVesselSynth(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a test directory and other setup."""
        cls.test_dir = 'test_synthetic_data'
        cls.experiment_number = 0
        cls.synth = VesselSynthEngineWrapper(
            experiment_dir=cls.test_dir,
            experiment_number=cls.experiment_number,
            n_volumes=2)
        cls.synth.synth()

    @classmethod
    def tearDownClass(cls):
        """Clean up the test directory after tests."""
        shutil.rmtree(f'output/{cls.test_dir}', ignore_errors=True)

    def test_output_directory_created(self):
        """Test if the output directory is created."""
        self.assertTrue(
            os.path.exists(self.synth.experiment_path),
            "Output directory was not created."
            )

    def test_notes_file_creation(self):
        """Test if the notes file is created."""
        self.assertTrue(
            os.path.isfile(
                f'{self.synth.experiment_path}/#_notes.txt'
                ), "Notes file was not created."
            )

    def test_params_file_created(self):
        """Test if the parameters file is created."""
        params_file = f'{self.synth.experiment_path}/#_vesselsynth_params.json'
        self.assertTrue(os.path.isfile(params_file),
                        "Parameters file was not created.")

        with open(params_file, 'r') as file:
            data = json.load(file)
            self.assertIn('shape', data,
                          "Parameter 'shape' not found in parameters file.")

    def test_volumes_created(self):
        """Test if correct number of volumes NIfTIs were created."""
        for n in range(self.synth.n_volumes):
            for name in ['prob', 'label', 'level', 'nb_levels',
                         'branch', 'skeleton']:
                volume_file = (f'{self.synth.experiment_path}'
                               f'/{n:04d}_vessels_{name}.nii.gz'
                               )
                self.assertTrue(
                    os.path.isfile(volume_file),
                    f"Volume file {volume_file} was not created."
                    )

    def test_volume_content(self):
        """Test if the volumes are valid NIfTI files."""
        for n in range(self.synth.n_volumes):
            for name in ['prob', 'label', 'level', 'nb_levels',
                         'branch', 'skeleton']:
                volume_file = (f'{self.synth.experiment_path}'
                               f'/{n:04d}_vessels_{name}.nii.gz')
                img = nib.load(volume_file)
                self.assertIsInstance(
                    img, nib.Nifti1Image,
                    f"{volume_file} is not a valid NIfTI image.")


if __name__ == '__main__':
    unittest.main()
