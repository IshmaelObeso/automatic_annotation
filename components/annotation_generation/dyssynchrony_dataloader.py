import pandas as pd
import numpy as np

from components.annotation_generation.utilities.transforms import ChopDatBreathRightOnUp
from pathlib import Path
from typing import Union


class DyssynchronyDataLoader:
    """
    This class takes a directory of spectral triplets and applies transforms to them in order to make them ready for
    input into the dyssynchrony model. It should work as an iterable, applying transforms to one spectral triplet
    at a time (aka batch size of 1)

    Attributes
        _spectral_triplets_directory (Union[str, Path]): path to the directory where spectral triplets are stored
        _output_cols: list of the output columns
        _justification: whether to begin window on left or center of the central breath in triplet
        _max_length: max length of spectral triplet window
        _offset: how much to offset spectral triplet window to the left

    """

    def __init__(self, spectral_triplets_directory: Union[str, Path], output_cols: Union[list[str], None] = None, justification: str = 'left',
                 max_length: int = 900,
                 offset: int = 300) -> None:
        """
        Sets initial class attributes

        Args:
            spectral_triplets_directory (Union[str, Path]): path to the directory where spectral triplets are stored
            output_cols: list of the output columns
            justification: whether to begin window on left or center of the central breath in triplet
            max_length: max length of spectral triplet window
            offset: how much to offset spectral triplet window to the left
        """

        # define the spectral triplets directory and path to the spectral statics file
        self._spectral_triplets_directory, self._spectral_triplets_statics_path = DyssynchronyDataLoader._setup_directories(spectral_triplets_directory)

        # define how many breaths are in this dataset and what their corresponding uids are
        # create a mapper between uids, which is a triplet (patient_id, day, breath_no), to the directory where it lives
        # load statics dataset so that we have mappings between breath ids and paths
        spectral_triplets_statics = pd.read_hdf(self._spectral_triplets_statics_path)

        # make mapper between uids and their original subdirectory
        self._uid2dir = spectral_triplets_statics['original_subdirectory'].to_dict()

        # get all uids
        self._all_uids = self._get_available_uids(spectral_triplets_statics)

        # use fsp and paw channels
        self._inputs = [
                        'Fsp PSD',
                        'Fsp Angular',
                        'Paw PSD',
                        'Paw Angular',
                      ]

        self._INPUT2IDX_MAP = {
            'Fsp PSD': 0,
            'Fsp Angular': 3,
            'Paw PSD': 1,
            'Paw Angular': 4,
            'Pes PSD': 2,
            'Pes Angular': 5
        }

        # set target columns
        if output_cols is None:
            raise TypeError('Output Columns must not be None')
        else:
            self._output_cols = output_cols

        # transform attributes
        self._justification = justification
        self._max_length = max_length
        self._offset = offset

    @staticmethod
    def _setup_directories(spectral_triplets_directory: Union[str, Path]) -> tuple[Path, Path]:
        """

        Args:
            spectral_triplets_directory (Union[str, Path]): path to the directory where spectral triplets are stored

        Returns:
            spectral_triplets_directory (Path): path to the directory where spectral triplets are stored
            spectral_triplets_statics (Path): path to the directory where spectral statics file is stored
        """
        # define path
        spectral_triplets_directory = Path(spectral_triplets_directory)

        # get the spectral triplet statics file path
        spectral_triplets_statics = Path(spectral_triplets_directory, 'spectral_statics.hdf')

        return spectral_triplets_directory, spectral_triplets_statics

    def _get_available_uids(self, spectral_triplets_statics: pd.DataFrame) -> pd.Index:
        """
        Gets all uids that have an existing pickle file to load

        Args:
            spectral_triplets_statics (pd.DataFrame): pandas dataframe with statics information

        Returns:
            all_uids_pickle (pd.Index): pandas index that has all uids that have a corresponding pickle file in
                                        this dataset
        """

        # look for uids with corresponding pickle files.  these are the only ones which should be accessible in this dataset.
        all_uids_pickle = []
        for uid in spectral_triplets_statics.index:
            uid_dir = self._uid2dir[uid]
            breath_no = uid[2]
            pickle_path = Path(self._spectral_triplets_directory, uid_dir, 'breath_%d.pickle' % breath_no)
            if pickle_path.exists():
                all_uids_pickle.append(uid)

        # turn self.all_uids_pickle to index object
        all_uids_pickle = pd.Index(all_uids_pickle)

        return all_uids_pickle

    def _transforms(self, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, uid: tuple[str, str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, str, int]]:
        """
        Applies any transforms needed

        Args:
            xs (np.ndarray): spectral data for breath
            ys (np.ndarray): truth for breath
            ts (np.ndarray): timesteps of breath
            uid (tuple[str, str, int]): uid of breath

        Returns:
            xs (np.ndarray): transformed spectral data for breath
            ys (np.ndarray): transformed truth for breath
            ts (np.ndarray): transformed timesteps of breath
            uid (tuple[str, str, int]): transformed uid of breath
        """
        # instantiate transform classes
        spectral_transformer = ChopDatBreathRightOnUp(self._justification, self._max_length, self._offset)

        # apply any transforms
        xs, ys, ts, uids = spectral_transformer.forward(xs, ys, ts, uid)

        return xs, ys, ts, uids

    @property
    def all_uids(self) -> pd.Index:
        """
        Returns all uids in this dataset

        Returns:
            self._all_uids (pd.Index): All uids in this dataset
        """

        return self._all_uids

    @property
    def __len__(self) -> int:
        """
        Returns length of dataset (number of spectral triplets)

        Returns:
            len (int): How many uids there are in this dataset (aka. how many batches, because batch size will be 1)
         """

        return len(self._all_uids)

    def __getitem__(self, uid: tuple[str, str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, str, int]]:
        """
        Serves the model one spectral triplet at a time

        Args:
            uid (tuple[str, str, int]): breath id triplet (patient_id, day, breath no.)

        Returns:
            xs (np.ndarray): spectral data for breath
            ys (np.ndarray): truth for breath
            ts (np.ndarray): timesteps of breath
            uid (tuple[str, str, int]): uid of breath
        """

        # get breath directory using the mapper
        uid_dir = self._uid2dir[uid]

        # read pickle file (!! WARNING !! assumes directory structure is consistent...)
        breath_no = uid[2]
        pickle_path = Path(self._spectral_triplets_directory, uid_dir, f'breath_{breath_no}.pickle')
        data = pd.read_pickle(pickle_path)

        # access x, y pairs
        xs = np.array(data['tensor'])

        # !! WARNING !! We assume that esophageal data is in channel 2 and 5 (PSD and angle, respectively),
        # so we only take the remaining indices if we're excluding it
        input_idxs = list(map(self._INPUT2IDX_MAP.get, self._inputs))
        xs = xs[input_idxs]

        # load truth and time columns
        ys = data['truth'][self._output_cols].values.reshape(-1, len(self._output_cols))
        ts = data['truth'].index.values

        # transform data, then return xs, ys, ts, and uids to whoever calls this function
        xs, ys, ts, uid = self._transforms(xs, ys, ts, uid)
        return xs, ys, ts, uid
