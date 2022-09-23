import pandas as pd
import numpy as np
from scipy import signal

import os
import pickle
import tqdm
import argparse
import pdb
from datetime import datetime
from pathlib import Path
import utils

# silence pandas warning
pd.options.mode.chained_assignment = None

class Spectral_Triplet_Generator:
    ''' This Class carries out all functions of the triplet generator.

        Inputs:
            triplet_files_directory --> Path to directory where outputs from the triplet generator are kept
            Export directory --> Path to directory where outputs from the spectral triplet generator should be kept

        Outputs:
            Spectral Triplets directory --> Directory with spectral triplets generated for every breath triplet in the import directory
            Statics File --> statics file with information on all patient-days provided in the import directory (csv and hdf)

        '''

    def __init__(self, triplet_directory, spectral_triplet_export_directory="..\\datasets\\spectral_triplets"):

        # # setup import and export directories
        self.triplet_directory, self.spectral_triplet_export_directory, self.triplet_statics_path, self.statics_output_path_csv, self.statics_output_path_hdf = self.setup_directories(triplet_directory, spectral_triplet_export_directory)

        # get triplet statics file path
        self.triplet_statics = os.path.join(triplet_directory, 'statics.hdf')

    def setup_directories(self, triplet_directory, spectral_triplet_export_directory):
        # strip quotes
        triplet_directory = triplet_directory.replace('"', '').replace("'", '')
        spectral_triplet_export_directory = spectral_triplet_export_directory.replace('"', '').replace("'", '')

        # make export directory with timestamp
        spectral_triplet_export_directory = os.path.join(spectral_triplet_export_directory,
                                                str(datetime.now()).replace(':', '-').replace(' ', ','))
        os.makedirs(spectral_triplet_export_directory)

        # get the triplet statics file path
        triplet_statics_path = os.path.join(triplet_directory, 'statics.hdf')

        # setup statics output path
        statics_output_path_csv = os.path.join(spectral_triplet_export_directory, 'statics_spectral.hdf')
        statics_output_path_hdf = os.path.join(spectral_triplet_export_directory, 'statics_spectral.csv')

        return triplet_directory,\
               spectral_triplet_export_directory,\
               triplet_statics_path,\
               statics_output_path_csv,\
               statics_output_path_hdf

    def setup_spectral_directories(self, subdir_name):

        triplet_subdir = os.path.join(self.triplet_directory, subdir_name)
        spectral_triplet_subdir = os.path.join(self.spectral_triplet_export_directory, subdir_name)

        if not os.path.exists(spectral_triplet_subdir):
            os.mkdir(spectral_triplet_subdir)

        triplet_csv_file_names = os.listdir(triplet_subdir)

        return triplet_subdir, spectral_triplet_subdir, triplet_csv_file_names

    def initialize_spectral_triplet(self, triplet_subdir, triplet_csv_file_name):

        triplet = pd.read_csv(os.path.join(triplet_subdir, triplet_csv_file_name))

        # Save the triplet id (which corresponds to the breath id) so
        # that we can use this to merge into statics as well
        triplet_id = triplet['triplet_id'].iloc[0]

        # Initialize the tensor and dictionary that will get pickled
        spectral_tensor = []
        tensor_and_truth = {}

        return triplet, triplet_id, spectral_tensor, tensor_and_truth

    def create_spectral_triplet(self, triplet, spectral_tensor, has_spectral_triplet, subdir_name, triplet_id):

        for mode in utils.MODES:
            for waveform_column in utils.WAVEFORM_COLUMNS:
                spectrogram = signal.spectrogram(triplet[waveform_column],
                                                 fs=utils.FS,
                                                 window=utils.WINDOW,
                                                 nperseg=utils.NPERSEG,
                                                 noverlap=utils.NOVERLAP,
                                                 nfft=utils.NFFT,
                                                 return_onesided=utils.RETURN_ONESIDED,
                                                 scaling=utils.SCALING,
                                                 mode=mode)

                # Only transform into decibel space when taking PSD
                # The x and y returned from signal.spectrogram are the frequency and temporal components
                # We only care about the magnitude, or z axis
                if mode == 'psd':
                    spectrogram_z = 10 * np.log10(spectrogram[2])
                else:
                    spectrogram_z = spectrogram[2]

                spectral_tensor.append(spectrogram_z[:utils.FREQ_BINS])

        # Convert the spectral tensor to an np array for easy slicing and dicing later
        spectral_tensor = np.array(spectral_tensor)

        # If infinities were produced (log(0) = -inf) we want to flag this in the statics file
        # so that we don't build models on these. We will still save them as spectral triplets
        # in case we want to analyze them.
        if np.isinf(spectral_tensor).sum() == 0:
            has_spectral_triplet.append([subdir_name, triplet_id])

        ## TODO: Remove this, you can just do triplet['triplet_id'] != triplet['breath_id']
        # Grab the first and last breath IDs to nan out
        first_and_last_breath = [triplet['breath_id'].min(),
                                 triplet['breath_id'].max()]

        # Clever indexing to grab the truth labels, which exist at the halfway point of each window and should step in sync with the window stride
        triplet = triplet.iloc[int(utils.NPERSEG / 2 - 1):-int(utils.NPERSEG / 2):utils.STRIDE]
        triplet.loc[triplet['breath_id'].isin(first_and_last_breath), list(set(utils.dyssynchrony.values())) +
                    list(set(utils.COOCCURRENCES_TO_CREATE_VIA_OR.keys())) +
                    list(set(utils.COOCCURRENCES_TO_CREATE_VIA_AND.keys())) +
                    ['No Double Trigger', 'No Inadequate Support']] = np.nan

        return triplet, spectral_tensor, has_spectral_triplet

    def save_spectral_triplet(self, tensor_and_truth, triplet, spectral_tensor, spectral_triplet_subdir, triplet_csv_file_name):

        # Fill in the object to be pickled with the tensor and the triplet file containing the truth values
        tensor_and_truth['tensor'] = spectral_tensor
        tensor_and_truth['truth'] = triplet

        # Pickle the object
        spectral_triplet_pickle_file_name = os.path.join(spectral_triplet_subdir, triplet_csv_file_name[:-4] + '.pickle')
        with open(spectral_triplet_pickle_file_name, 'wb') as file:
            pickle.dump(tensor_and_truth, file)

    def finalize_statics(self, has_spectral_triplet):

        # read in statics file
        statics = pd.read_hdf(self.triplet_statics)

        # Now that we've looped through all of the patient day triplet subdirectories,
        # we have a list of all breaths that map to a spectral triplet.
        # We'll append a binary indicator has_spectral_triplet to the statics file.
        has_spectral_triplet = pd.DataFrame(has_spectral_triplet, columns=utils.MERGE_COLUMNS)
        has_spectral_triplet['has_spectral_triplet'] = 1

        # Save this to restore the index after the merge
        statics_index = statics.index.names

        # Merge the has_spectral_triplet column into statics
        statics = statics.reset_index().merge(has_spectral_triplet,
                                              how='left',
                                              on=utils.MERGE_COLUMNS).set_index(statics_index)

        # The breaths that have spectral triplets have been flagged and the rest should be 0
        statics['has_spectral_triplet'] = statics['has_spectral_triplet'].fillna(0)

        # Write the statics file
        statics.to_hdf(self.statics_output_path_hdf, key='statics')
        statics.to_csv(self.statics_output_path_csv)


    def generate_spectral_triplets(self):

        # Grab the triplet folders from their directories
        p = Path(self.triplet_directory)
        subdir_names = [subdir.name for subdir in p.iterdir() if subdir.is_dir()]

        # A dictionary to identify which breaths in statics actually have spectral triplets
        has_spectral_triplet = []

        print(f'Creating spectrogram tensors from {len(subdir_names)} subdirectories of breath triplets...')

        for subdir_name in tqdm.tqdm(subdir_names):

            # setup spectral triplet directories and get list of files
            triplet_subdir, spectral_triplet_subdir, triplet_csv_file_names = self.setup_spectral_directories(subdir_name)

            for triplet_csv_file_name in triplet_csv_file_names:

                # initialize spectral triplet
                triplet, triplet_id, spectral_tensor, tensor_and_truth = self.initialize_spectral_triplet(triplet_subdir, triplet_csv_file_name)

                # create spectral triplet and save to pkl file
                self.create_spectral_triplet(triplet, spectral_tensor, has_spectral_triplet, subdir_name, triplet_id)

                # save spectral triplet to pickle file
                self.save_spectral_triplet(tensor_and_truth, triplet, spectral_tensor, spectral_triplet_subdir, triplet_csv_file_name)

        # after looping through every triplets file, finalize the statics file and save it out
        self.finalize_statics(has_spectral_triplet)

        return self.spectral_triplet_export_directory

# if running this file directly, only do do spectral triplet generation
if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--export_directory', type=str, default="..\\datasets\\spectral_triplets", help='Directory to export organized unannotated files for later processing')
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']
    export_directory = args['export_directory']

    # instantiate triplet generator class
    spectral_triplet_generator = Spectral_Triplet_Generator(input_directory, export_directory)

    # run triplet generator
    export_directory = spectral_triplet_generator.generate_spectral_triplets()
    statics_csv_output = spectral_triplet_generator.statics_output_path_csv

    print(f'Spectral Triplets generated at {os.path.abspath(export_directory)}')
    print(f"Statics file generated at {os.path.abspath(statics_csv_output)}")
