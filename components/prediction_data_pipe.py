import pandas as pd
import numpy as np

from .utilities.transforms import ChopDatBreathRightOnUp
from pathlib import Path

class Data_Pipe:
    """
    This class takes a directory of spectral triplets and applies transforms to them in order to make them ready for
    input into the dyssynchrony model. It should work as an iterable, applying transforms to one spectral triplet
    at a time (aka batch size of 1)
    """

    def __init__(self, spectral_triplets_directory, output_cols=['Double Trigger'], justification='left', max_length=900, offset=300):

        # define the spectral triplets directory and path to the spectral statics file
        self.spectral_triplets_directory, self.spectral_triplets_statics_path = self.setup_directories(spectral_triplets_directory)

        # define how many breaths are in this dataset and what their corresponding uids are
        # create a mapper between uids, which is a triplet (patient_id, day, breath_no), to the directory where it lives
        # load statics dataset so that we have mappings between breath ids and paths
        spectral_triplets_statics = pd.read_hdf(self.spectral_triplets_statics_path)

        # make mapper between uids and their original subdirectory
        self.uid2dir = spectral_triplets_statics['original_subdirectory'].to_dict()

        # get all uids
        self.all_uids = self.get_available_uids(spectral_triplets_statics)

        # use fsp and paw channels
        self.inputs = [
                        'Fsp PSD',
                        'Fsp Angular',
                        'Paw PSD',
                        'Paw Angular',
                      ]

        self.INPUT2IDX_MAP = {
            'Fsp PSD': 0,
            'Fsp Angular': 3,
            'Paw PSD': 1,
            'Paw Angular': 4,
            'Pes PSD': 2,
            'Pes Angular': 5
        }

        # set target columns
        self.output_cols = output_cols

        # transform attributes
        self.justification = justification
        self.max_length = max_length
        self.offset = offset

    def setup_directories(self, spectral_triplets_directory):

        # define path
        spectral_triplets_directory = Path(spectral_triplets_directory)

        # get the spectral triplet statics file path
        spectral_triplets_statics = Path(spectral_triplets_directory, 'spectral_statics.hdf')

        return spectral_triplets_directory, spectral_triplets_statics

    def get_available_uids(self, spectral_triplets_statics):
        """
        Gets all uids that have an existing pickle file to load
        """

        # look for uids with corresponding pickle files.  these are the only ones which should be accessible in this dataset.
        all_uids_pickle = []
        for uid in spectral_triplets_statics.index:
            uid_dir = self.uid2dir[uid]
            breath_no = uid[2]
            pickle_path = Path(self.spectral_triplets_directory, uid_dir, 'breath_%d.pickle' % breath_no)
            if pickle_path.exists():
                all_uids_pickle.append(uid)

        # turn self.all_uids_pickle to index object
        all_uids_pickle = pd.Index(all_uids_pickle)

        return all_uids_pickle

    def get_uids(self):

        return self.all_uids

    def transforms(self, xs, ys, ts, uid):
        """
        Applies any transforms needed
        """
        # instantiate transform classes
        spectral_transformer = ChopDatBreathRightOnUp(self.justification, self.max_length, self.offset)

        # apply any transforms
        xs, ys, ts, uids = spectral_transformer.forward(xs, ys, ts, uid)

        return xs, ys, ts, uids



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

        # get breath directory using the mapper
        uid_dir = self.uid2dir[uid]

        # read pickle file (!! WARNING !! assumes directory structure is consistent...)
        breath_no = uid[2]
        pickle_path = Path(self.spectral_triplets_directory, uid_dir, f'breath_{breath_no}.pickle')
        data = pd.read_pickle(pickle_path)

        # access x, y pairs
        xs = np.array(data['tensor'])

        # !! WARNING !! We assume that esophageal data is in channel 2 and 5 (PSD and angle, respectively),
        # so we only take the remaining indices if we're excluding it
        input_idxs = list(map(self.INPUT2IDX_MAP.get, self.inputs))
        xs = xs[input_idxs]

        # load truth and time columns
        ys = data['truth'][self.output_cols].values.reshape(-1, len(self.output_cols))
        ts = data['truth'].index.values

        # transform data, then return xs, ys, ts, and uids to whoever calls this function
        xs, ys, ts, uid = self.transforms(xs, ys, ts, uid)
        return xs, ys, ts, uid

