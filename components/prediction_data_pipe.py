import pandas as pd
import numpy as np
from scipy import signal

import os
import pickle
import tqdm
import argparse
from datetime import datetime
from pathlib import Path




class Data_Pipe:
    """
    This class takes a directory of spectral triplets and applies transforms to them in order to make them ready for
    input into the dyssynchrony model. It should work as an iterable, applying transforms to one spectral triplet
    at a time (aka batch size of 1)
    """

    def __init__(self, spectral_triplets_directory):

        # define the spectral triplets directory and path to the spectral statics file
        self.spectral_triplets_directory, self.spectral_triplets_statics_path = self.setup_directories(spectral_triplets_directory)

        # define how many breaths are in this dataset and what their corresponding uids are
        # create a mapper between uids, which is a triplet (patient_id, day, breath_no), to the directory where it lives
        # load statics dataset so that we have mappings between breath ids and paths
        spectral_triplets_statics = pd.read_hdf(self.spectral_triplets_statics_path)

        # make mapper between uids and their original subdirectory
        self.uid2dir = spectral_triplets_statics['original_subdirectory'].to_dict()

        # get all uids
        self.all_uids = spectral_triplets_statics.index


    def setup_directories(self, spectral_triplets_directory):

        # strip quotes
        spectral_triplets_directory = spectral_triplets_directory.replace('"', '').replace("'", '')


        # get the spectral triplet statics file path
        spectral_triplets_statics = os.path.join(spectral_triplets_directory, 'spectral_statics.hdf')

        return spectral_triplets_directory, spectral_triplets_statics

    def get_all_subdirs(self):

        # get a list of all spectral triplet subdirectories
        p = Path(self.spectral_triplets_directory)
        subdir_names = [subdir.name for subdir in p.iterdir() if subdir.is_dir()]

        return subdir_names

    def get_uids(self):

        return self.all_uids

    def transforms(self, spectral_triplet):
        pass


    def __len__(self):
        """ returns length of dataset (number of spectral triplets) """

        return len(self.all_uids)


    def __getitem__(self, uid):
        """ Serves the model one spectral triplet at a time
        Args:
            uid (triplet): breath id triplet (patient_id, day, breath no.)
        Returns:
            xs : spectral data for breath
            ys : truth for breath
            ts : timesteps of breath
            uid : uid of breath

        """
        pass