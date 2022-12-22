import pandas as pd
import numpy as np
import pickle
import warnings
import argparse

from scipy import signal
from pathlib import Path
from components.dataset_generation.utilities import utils
from components.dataset_generation.datacleaner import DataCleaner
from pqdm.processes import pqdm
from typing import Union

# silence pandas warning
pd.options.mode.chained_assignment = None

# silence scipy divide by zero warning when waveform has discontinuity
warnings.filterwarnings("ignore", message=".*divide")

# define some complex types for typehints
FilterFileInfo = dict[[str, bool], [str, str], dict[str, str], [str, str]]


class SpectralTripletGenerator:
    """
    This class generates spectral triplets from triplet in the dataset

    Attributes:
        _triplet_directory (Path): directory where triplets from every patient-day are stored
        _filter_file_info (FilterFileInfo): dict that contains information about the filter file (if it is included), including
                                what columns to filter over
        _spectral_triplet_export_directory (Path): Path to the directory where spectral triplets will be stored
        _triplet_statics_path (Path): path to the triplet statics file
        _statics_directory (Path): path to the directory where statics file will be stored
        _data_cleaner (DataCleaner): DataCleaner class that can do checks on the dataset and make sure it's clean
        _triplet_statics (Path): Path to the triplet statics file

    """

    def __init__(self, triplet_directory: Union[str, Path], filter_file_info: FilterFileInfo = None) -> None:
        """
        Sets initial class attributes

        Args:
            triplet_directory (Path): directory where triplets from every patient-day are stored
            filter_file_info (FilterFileInfo): dict that contains information about the filter file (if it is included), including
                                               what columns to filter over

        Returns:
            None:
        """

        # save attributes
        self._triplet_directory = triplet_directory
        self._filter_file_info = filter_file_info
        # we will modify these later
        self._spectral_triplet_export_directory = None
        self._triplet_statics_path = None
        self._statics_directory = None
        self._data_cleaner = None
        self._triplet_statics = None

    @staticmethod
    def _setup_directories(triplet_directory: Union[str, Path]) -> tuple[Path, Path, Path, Path]:
        """
        Sets up spectral_triplet_export_directory ,statics_directory, and triplet_statics_path from triplet_directory path

        Args:
            triplet_directory: directory where triplets from every patient-day are stored

        Returns:
            triplet_directory (Path): Path to directory where triplets are saved
            spectral_triplet_export_directory (Path): Path to directory where spectral triplets will be saved
            triplet_statics_path (Path): Path to directory where statics files are saved
            statics_directory (Path): Path to directory where spectral statics files will be saved
        """

        # define paths
        triplet_directory = Path(triplet_directory)

        # put spectral triplets directory in parent directory of triplets directory
        triplets_parent_directory = triplet_directory.parents[0]
        spectral_triplet_export_directory = Path(triplets_parent_directory, 'spectral_triplets')
        spectral_triplet_export_directory.mkdir(parents=True, exist_ok=True)

        # create directory for statics files
        statics_directory = Path(triplets_parent_directory, 'statics')
        statics_directory.mkdir(parents=True, exist_ok=True)

        # get the triplet statics file path
        triplet_statics_path = Path(statics_directory, 'statics.hdf')

        return triplet_directory.resolve(),\
               spectral_triplet_export_directory.resolve(),\
               triplet_statics_path.resolve(),\
               statics_directory.resolve()

    def _setup_spectral_subdirectories(self, subdir_name: str) -> tuple[Path, Path, list[str]]:
        """
        Sets up paths to the spectral_triplet_subdirectory where we will save our spectral triplets for a specific patient-day

        Args:
            subdir_name: string of subdirectory name (patient-day triplet subdirectory)

        Returns:
            triplet_subdir (Path): Path to directory where triplet is saved
            spectral_triplet_subdir (Path): Path to directory where spectral triplet will be saved
            triplet_csv_file_names (list[str]): List of csv files with triplets in them
        """

        # setup paths to triplet subdirectory and the spectral triplet subdirectory where spectral triplet will be saved
        triplet_subdir = Path(self._triplet_directory, subdir_name)
        spectral_triplet_subdir = Path(self._spectral_triplet_export_directory, subdir_name)
        spectral_triplet_subdir.mkdir(parents=True, exist_ok=True)

        # get list of triplet files from the triplet subdirectory
        triplet_csv_file_names = [x.name for x in triplet_subdir.iterdir() if x.is_file()]

        return triplet_subdir, spectral_triplet_subdir, triplet_csv_file_names

    @staticmethod
    def _initialize_spectral_triplet(triplet_subdir: Path, triplet_csv_file_name: str) -> tuple[pd.DataFrame, int, list[None], dict[None:None]]:
        """
        Initialize empty tensor and dictionary to store spectral triplets data

        Args:
            triplet_subdir (Path): path to the patient-day subdirectory of triplets we are working with
            triplet_csv_file_name (str): string of name of one triplet csv file we are working with

        Returns:
            tuple[pd.DataFrame, int, list[None], dict[None:None]]: Returns triplet, triplet_id, spectral_tensor, and tensor_and_truth
            triplet (pd.DataFrame): DataFrame of triplet
            triplet_id (int): Id of triplet
            spectral_tensor (list[None]): empty list that will contain the spectral tensor
            tensor_and_truth (dict[None:None]]): empty dict that will contain spectral tensor and truth values
        """

        # load a triplet from csv
        triplet = pd.read_csv(Path(triplet_subdir, triplet_csv_file_name))

        # Save the triplet id (which corresponds to the breath id) so
        # that we can use this to merge into statics as well
        triplet_id = triplet['triplet_id'].iloc[0]

        # Initialize the tensor and dictionary that will get pickled
        spectral_tensor = []
        tensor_and_truth = {}

        return triplet, triplet_id, spectral_tensor, tensor_and_truth

    def _create_spectral_triplet(self, triplet: pd.DataFrame, spectral_tensor: list[None], has_spectral_triplet: list, keep_triplets: list,
                                 subdir_name: str,
                                 triplet_id: int) -> tuple[pd.DataFrame, np.ndarray, list, list]:
        """
        Generates spectrogram of triplet waveform and saves it as numpy array, also keeps truth values from triplet
        and stores indicators of whether a spectrogram was generated without infs (discontinuities) and
        whether we should filter this triplet out.

        Args:
            triplet (pd.DataFrame): DataFrame with triplet information
            spectral_tensor (list[None]): empty list where spectrogram generated from triplet will be stored
            has_spectral_triplet (list): list that stores indicator of whether a spectral triplet was generated for each triplet
            keep_triplets (list): list that stores indicator if we should keep the triplet based on filter file
            subdir_name (str): string that stores the name of the patient-day subdirectory we are working with
            triplet_id (int): the id of the triplet we are turning into spectrogram

        Returns:
            triplet (pd.DataFrame): DataFrame of triplet
            spectral_tensor (np.ndarray): Array of spectrogram of triplet
            has_spectral_triplet (list): list that stores indicator of whether a spectral triplet was generated for each triplet
            keep_triplets (list): list that stores indicator if we should keep the triplet based on filter file
        """

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

        # Convert the spectral tensor to nparray for easy slicing and dicing later
        spectral_tensor = np.array(spectral_tensor)

        # If infinities were produced (log(0) = -inf) we want to flag this in the statics file
        # so that we don't build models on these. We will still save them as spectral triplets
        # in case we want to analyze them.
        if np.isinf(spectral_tensor).sum() == 0:
            has_spectral_triplet.append([subdir_name, triplet_id])

        ## flag triplet if we want to keep it in our dataset
        keep_triplet = self._data_cleaner.check_for_validity(subdir_name)

        if keep_triplet:
            keep_triplets.append([subdir_name, triplet_id])

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

        return triplet, spectral_tensor, has_spectral_triplet, keep_triplets

    @staticmethod
    def _save_spectral_triplet(tensor_and_truth: dict, triplet: pd.DataFrame, spectral_tensor: np.ndarray, spectral_triplet_subdir: Path,
                               triplet_csv_file_name: str) -> None:
        """
        Saves the spectral triplet into the spectral patient-day subdirectory as a .pickle file

        Args:
            tensor_and_truth (list[None]): Empty dict that gets filled with the ndarray of the spectrogram and the truth values for that spectrogram
            triplet (pd.DataFrame): DataFrame of truth values of the triplet
            spectral_tensor (np.ndarray): ndarray of the spectrogram generated from triplet waveform
            spectral_triplet_subdir (Path): patient-day subdirectory where spectral triplet will be saved
            triplet_csv_file_name (str): filename of the triplet, spectral triplet .pickle file will have same name, but
                                        will be saved in spectral triplet subdirectory

        Returns:
            None:
        """
        # Fill in the dict to be pickled with the tensor and the triplet file containing the truth values
        tensor_and_truth['tensor'] = spectral_tensor
        tensor_and_truth['truth'] = triplet

        # Pickle the object
        spectral_triplet_pickle_file_name = Path(spectral_triplet_subdir, triplet_csv_file_name[:-4] + '.pickle')
        with open(spectral_triplet_pickle_file_name, 'wb') as file:
            pickle.dump(tensor_and_truth, file)

    def _finalize_statics(self, has_spectral_triplet: list, keep_triplets: list) -> None:
        """
        Adds columns to statics file that indicate whether a spectral triplet has been generated for a given patient, day, breath,
        and a column that indicates whether a given patient, day, breath should be filtered out of the dataset based
        on a given filtering file

        Args:
            has_spectral_triplet (list): list that stores indicator of whether a spectral triplet was generated for each triplet
            keep_triplets (list): list that stores indicator if we should keep the triplet based on filter file

        Returns:
            None:
        """

        # read in statics file
        # explicitly return DataFrame object for clarity
        statics = pd.DataFrame(pd.read_hdf(self._triplet_statics))

        # Now that we've looped through patient day triplet subdirectories,
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

        # if keep_triplets has entries
        try:
            # do the same with keep_triplet
            keep_triplets = pd.DataFrame(keep_triplets, columns=utils.MERGE_COLUMNS)
            keep_triplets['keep_triplet'] = 1

            # Merge the keep_triplets column into statics
            statics = statics.reset_index().merge(keep_triplets,
                                                  how='left',
                                                  on=utils.MERGE_COLUMNS).set_index(statics_index)
        # if keep triplets has no entries
        except:

            # make keep_triplet column filled with 0
            statics['keep_triplet'] = 0

        # the breaths that we should keep have been flagged, and the rest should be 0
        statics['keep_triplet'] = statics['keep_triplet'].fillna(0)

        # save out statics files to spectral triplets export directory
        statics.to_hdf(Path(self._spectral_triplet_export_directory, 'spectral_statics.hdf'), key='statics')
        statics.to_csv(Path(self._spectral_triplet_export_directory, 'spectral_statics.csv'))

        # save out statics files to statics directory
        statics.to_hdf(Path(self._statics_directory, 'spectral_statics.hdf'), key='statics')
        statics.to_csv(Path(self._statics_directory, 'spectral_statics.csv'))

    def _loop_through_spectral_triplets(self, subdir_name: str) -> tuple[list, list]:
        """
        Loops through all triplets in a patient-day, creates spectrogram's from their waveforms,
        saves them as spectral triplets and then returns information about has_spectral_triplets and keep_triplets
        for inclusion in the final statics


        Args:
            subdir_name (str): string that stores the name of the patient-day subdirectory we are working with

        Returns:
            has_spectral_triplet (list): list that stores indicator of whether a spectral triplet was generated for each triplet
            keep_triplets (list): list that stores indicator if we should keep the triplet based on filter file
        """

        # A list to identify which breaths in statics actually have spectral triplets
        has_spectral_triplet = []

        # a list to identify which breaths in statics should be filtered out based on Ben's csv file
        keep_triplets = []

        # setup spectral triplet directories and get list of files
        triplet_subdir, spectral_triplet_subdir, triplet_csv_file_names = self._setup_spectral_subdirectories(
            subdir_name)

        for triplet_csv_file_name in triplet_csv_file_names:

            # initialize spectral triplet
            triplet, triplet_id, spectral_tensor, tensor_and_truth = SpectralTripletGenerator._initialize_spectral_triplet(triplet_subdir,
                                                                                                       triplet_csv_file_name)

            # create spectral triplet and save to pkl file
            triplet, spectral_tensor, has_spectral_triplet, keep_triplets = self._create_spectral_triplet(triplet,
                                                                                                          spectral_tensor,
                                                                                                          has_spectral_triplet,
                                                                                                          keep_triplets,
                                                                                                          subdir_name,
                                                                                                          triplet_id)

            # save spectral triplet to pickle file
            SpectralTripletGenerator._save_spectral_triplet(tensor_and_truth, triplet, spectral_tensor, spectral_triplet_subdir,
                                        triplet_csv_file_name)

        return has_spectral_triplet, keep_triplets

    def generate_spectral_triplets(self, multiprocessing: bool = False) -> tuple[Path, Path]:
        """
        Creates spectral triplets in parallel (if flag is true), then saves them out and creates a statics file for them

        Args:
            multiprocessing: Boolean flag whether to multiprocess or not

        Returns:
            spectral_triplet_export_directory (Path): returns path to the directory where we saved all our spectral triplets
            statics_export_directory (Path): returns path to the directory where statics files are saved
        """

        # setup directories
        self._triplet_directory, \
            self._spectral_triplet_export_directory, \
            self._triplet_statics_path, \
            self._statics_directory = SpectralTripletGenerator._setup_directories(self._triplet_directory)

        # get the triplet statics filepath
        self._triplet_statics = Path(self._triplet_directory, 'statics.hdf')

        # instantiate data cleaner
        self._data_cleaner = DataCleaner(self._filter_file_info, parent_directory=self._triplet_directory.parents[0])

        # Grab the triplet folders from their directories
        p = Path(self._triplet_directory)
        subdir_names = [subdir.name for subdir in p.iterdir() if subdir.is_dir()]

        print(f'Creating spectrogram tensors from {len(subdir_names)} subdirectories of breath triplets...')

        # find the number of workers to use
        n_workers = utils.num_workers(multiprocessing)
        print(f'{n_workers} Processes assigned to generating Spectral Triplets')
        # multiprocessing requires a list to loop over, a function object, and number of workers
        # for each subdir name in subdir names, get the results from each function call and append them to a list called results
        results = pqdm(subdir_names, self._loop_through_spectral_triplets, n_jobs=1, desc='Patient-Days of Spectral Triplets Generated')

        # put together all results
        # nested list comprehension (does .extend instead of .append essentially)
        # [variable_to_extend_with for OUTER LOOP (item in list) for INNER LOOP (variable_to_extend_with in list)]
        has_spectral_triplet = [has_spectral_triplet_result for result in results for has_spectral_triplet_result in result[0]]
        keep_triplets = [keep_triplets_result for result in results for keep_triplets_result in result[1]]

        # after looping through every triplet file, finalize the statics file and save it out
        self._finalize_statics(has_spectral_triplet, keep_triplets)

        spectral_triplet_export_directory = self._spectral_triplet_export_directory
        statics_export_directory = self._statics_directory

        return spectral_triplet_export_directory, statics_export_directory


# if running this file directly, only do spectral triplet generation
if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with triplets files')
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']

    # instantiate spectral triplet generator class
    spectral_triplet_generator = SpectralTripletGenerator(input_directory)

    # run spectral triplet generator
    export_directory, statics_directory = spectral_triplet_generator.generate_spectral_triplets()

    print(f'Spectral Triplets generated at {export_directory}')
    print(f"Spectral Statics file generated at {statics_directory}")
