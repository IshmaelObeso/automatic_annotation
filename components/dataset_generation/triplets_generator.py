import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from components.dataset_generation.utilities import utils, deltaPes_utils
from pqdm.processes import pqdm
from typing import Union

# define some complex types for typehints
FilterFileInfo = dict[[str, bool], [str, str], dict[str, str], [str, str]]
DeltaPesList = list[list[float, str, str]]


class TripletGenerator:
    """
    This class generates triplets from every patient-day in the dataset

    Attributes:
        _batch_files_directory (Path): directory where batch-processed patient-day files are stored
        _filter_file_info (FilterFileInfo): dict that contains information about the filter file (if it is included), including
                                what columns to filter over
        _triplet_export_directory (Path): path to directory where triplets will be stored
        _statics_directory (Path): path to directory where statics files will be stored

    """

    def __init__(self, batch_files_directory: Union[str, Path], filter_file_info: FilterFileInfo = None) -> None:
        """
        Sets initial class attributes

        Args:
            batch_files_directory (Union[str, Path]): Path object that stores path to the batch files directory, where outputs from
                                   the batch processor are stored
            filter_file_info (FilterFileInfo): dict that contains information about the filter file,
                                               including which columns to filter over

        Returns:
            None:
        """

        # save attributes
        self._batch_files_directory = batch_files_directory
        self._filter_file_info = filter_file_info
        # we will modify these later
        self._statics_directory = None
        self._triplet_export_directory = None

    @staticmethod
    def _setup_directories(batch_files_directory: Union[str, Path]) -> tuple[Path, Path, Path]:
        """
        Sets up triplet_export_directory and statics_directory to store outputs from batch_files_directory path
        
        Args:
            batch_files_directory (Union[str, Path]): Path to the directory where batch files are located

        Returns:
            batch_files_directory (Path): Path to the directory where batch files are located
            triplet_export_directory (Path): Path to directory where triplets will be saved
            statics_directory (Path): Path to directory where statics files will be saved

        """
        # define paths
        batch_files_directory = Path(batch_files_directory)

        # put triplets directory in directory where batch files directory is
        batch_files_parent_directory = batch_files_directory.parents[0]
        triplet_export_directory = Path(batch_files_parent_directory, 'triplets')

        # create directory for statics files
        statics_directory = Path(batch_files_parent_directory, 'statics')

        # create export directory
        triplet_export_directory.mkdir(parents=True, exist_ok=True)
        statics_directory.mkdir(parents=True, exist_ok=True)

        return batch_files_directory.resolve(), triplet_export_directory.resolve(), statics_directory.resolve()

    def _get_patient_day(self, subdir_name: str) -> tuple[str, str, Path, Path, pd.DataFrame]:
        """
        Gets the patient-day from a subdirectory name
        
        Args:
            subdir_name (str): string of subdirectory name (patient-day subdirectory)

        Returns:
            patient_id (str): Patient ID of the triplet
            day_id (str): Day ID of the riplet
            patient_day_dir (Path): Directory where triplets for this patient-day is located
            patient_day_output_dir (Path): Directory where spectral triplets for this patient-day will be saved
            patient_day (pd.DataFrame): DataFrame with statics information about this patient-day
        """
        
        # get the patient and day id
        patient_id = utils.get_patient_id(subdir_name)
        day_id = utils.get_day_id(subdir_name)

        # get paths for patient day folder
        patient_day_dir = Path(self._batch_files_directory, subdir_name)
        patient_day_output_dir = Path(self._triplet_export_directory, subdir_name)

        patient_day_output_dir.mkdir(parents=True, exist_ok=True)

        # get all csv files in patient day folder
        csv_files = [x.name for x in patient_day_dir.iterdir()]

        # Find which file is the TriggersAndArtifacts file we care about
        ta_file_index = np.where([utils.TA_CSV_SUFFIX in csv for csv in csv_files])[0][0]

        patient_day = pd.read_csv(Path(patient_day_dir, csv_files[ta_file_index]))

        return patient_id, day_id, patient_day_dir, patient_day_output_dir, patient_day

    @staticmethod
    def _create_dyssynchrony_mask(patient_day: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a mask to only select rows to search for dyssynchronies
        i.e. Only look for dyssynchronies between inspiration and expiration

        Args:
            patient_day (pd.Dataframe): DataFrame with waveform information, inspiration and expiration triggers,
                                        and dyssynchrony information about the patient-day

        Returns:
           patient_day (pd.DataFrame): Patient-day dataframe with added dyssynchrony_mask column

        """

        # Create a mask to select only rows in which I should search for dyssynchronies
        # By creating column where rows between inspiration trigger (1) and expiration trigger (-1)
        # are equal to 1, which will be used for filtering
        patient_day['dyssynchrony_mask'] = patient_day[utils.DELINEATION_COLUMN].cumsum()

        # If the first trigger in the patient day is an expiration (-1) then the cumsum will
        # need 1 added to the entire dyssynchrony_mask column
        if patient_day['dyssynchrony_mask'].min() == -1:
            patient_day['dyssynchrony_mask'] += 1

        return patient_day

    @staticmethod
    def _create_breath_ids(patient_day: pd.DataFrame) -> pd.DataFrame:
        """
        Creates breath id's by looking at where expiration triggers occur, every time one occurs increase breath id by 1


        Args:
            patient_day (pd.DataFrame): DataFrame with waveform information, inspiration and expiration triggers,
                                        and dyssynchrony information about the patient-day

        Returns:
            patient_day (pd.DataFrame): Patient-day dataframe with breath_ids included

        """

        # Create breath ids by replacing expiration triggers
        patient_day['breath_id'] = patient_day[utils.DELINEATION_COLUMN].replace(-1, 0)
        patient_day['breath_id'] = patient_day['breath_id'].cumsum()

        return patient_day

    @staticmethod
    def _build_statics_file(patient_day: pd.DataFrame, patient_id: str, day_id: str, subdir_name: str) -> pd.DataFrame:
        """
        Builds a statics DataFrame from a patient_day dataframe and information about that patient-day

        Args:
            patient_day (pd.DataFrame): DataFrame with waveform information, inspiration and expiration triggers,
                                        and dyssynchrony information about the patient-day
            patient_id (str): string of the patient day id
            day_id (str): string of the day id
            subdir_name (str): string of subdirectory name (patient-day subdirectory)

        Returns:
            patient_day_statics (pd.DataFrame): Built statics file with summary information about the patient-day.
                                                Has columns with info about dyssynchrony counts in this patient-day

        """

        # Build the statics file, creating ID columns for the patient and day based on the filename
        patient_day_statics = utils.create_breath_statics(patient_day)

        patient_day_statics['patient_id'] = patient_id
        patient_day_statics['day_id'] = day_id
        patient_day_statics['original_subdirectory'] = subdir_name

        return patient_day_statics

    @staticmethod
    def _build_triplet(patient_day: pd.DataFrame, patient_day_output_dir: Path, breath_id: int, one_hot_dyssynchronies: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
        """
        Builds a triplet DataFrame from a patient-day by taking the breaths surrounding the current breath_id,
         including the current breath

        Args:
            patient_day (pd.DataFrame): DataFrame with waveform information, inspiration and expiration triggers,
                                        and dyssynchrony information about the patient-day
            patient_day_output_dir (Path): Path to output directory for this patient-day triplet
            breath_id (int): string of breath id for this triplet
            one_hot_dyssynchronies (pd.DataFrame): Dataframe with dyssynchronies one-hot-encoded in columns

        Returns:
            triplet (pd.DataFrame): DataFrame of Triplet
            triplet_csv_filename (Path): Path to where triplet csv should be saved
        """

        # Take the breath before, current breath and breath after as a triplet
        triplet = patient_day[patient_day['breath_id'].isin([breath_id - 1,
                                                             breath_id,
                                                             breath_id + 1])].reset_index()

        # Create a triplet ID, which will just be the ID of the middle breath
        triplet['triplet_id'] = breath_id

        # Merge so that the one hot columns span the entire triplet
        triplet = triplet.merge(one_hot_dyssynchronies[one_hot_dyssynchronies.index == breath_id],
                                left_on='triplet_id',
                                right_index=True,
                                how='left')

        triplet = triplet[utils.KEEP_COLUMNS]

        triplet_csv_filename = Path(patient_day_output_dir,
                                    'breath_{}.csv'.format(breath_id))

        return triplet, triplet_csv_filename

    @staticmethod
    def _calculate_deltaPes(triplet: pd.DataFrame, breath_id: int, subdir: str,
                            deltaPes_list: DeltaPesList) -> tuple[pd.DataFrame, DeltaPesList]:
        """
        Calculates deltaPes of the central breath of the triplet and saves that information for later

        Args:
            triplet (pd.DataFrame): DataFrame with information about the triplet
            breath_id (int): string of breath id of this triplet
            subdir (str): string of subdirectory where triplet should be stored
            deltaPes_list (DeltaPesList): list of breaths that have deltaPes recorded, and information about those breaths

        Returns:
            triplet (pd.DataFrame): DataFrame of triplet with deltaPes information added
            deltaPes_list (DeltaPesList): list of breaths with deltaPes calculated
        """

        # try to calculate deltaPes for the central breath in this triplet
        try:
            deltaPes = deltaPes_utils.calculate_deltaPes(triplet)
        # if there is a problem calculating deltaPES, save deltaPES as NaN
        except:
            deltaPes = np.nan

        deltaPes_list.append([deltaPes, subdir, breath_id])

        # get binary truth for deltaPES, 0 if <20, 1 if >= 20
        deltaPes_binary = int(deltaPes >= 20)

        # save deltaPes information in the triplet dataframe
        triplet['deltaPes'] = deltaPes
        triplet['deltaPes_binary'] = deltaPes_binary
        triplet['deltaPes_clipped'] = triplet['deltaPes'].clip(upper=40)

        return triplet, deltaPes_list

    @staticmethod
    def _add_deltaPes_to_statics(all_patient_day_statics: pd.DataFrame, deltaPes_list: DeltaPesList) -> pd.DataFrame:
        """
        Adds deltaPes information to the statics file

        Args:
            all_patient_day_statics (pd.DataFrame): DataFrame with summary information (statics) about all patient-days
                                                    in the dataset
            deltaPes_list (DeltaPesList): list of all triplets in the dataset that have deltaPes calculated

        Returns:
            all_patient_day_statics (pd.DataFrame): Dataframe of all_patient_days_statics with deltaPes information added

        """

        merge_columns = ['original_subdirectory', 'breath_id']

        # Save this to restore the index after the merge
        statics_index = all_patient_day_statics.index.names

        # merge deltaPes into statics
        deltaPes_df = pd.DataFrame(deltaPes_list, columns=['deltaPes'] + merge_columns)

        all_patient_day_statics = all_patient_day_statics.reset_index().merge(deltaPes_df,
                                                                              how='left',
                                                                              on=merge_columns).set_index(statics_index)

        # threshold deltaPes, values above 40 will be clipped to 40
        all_patient_day_statics['deltaPes_clipped'] = all_patient_day_statics['deltaPes'].clip(upper=40)

        # binarize deltaPES in the statics file
        # 0 if deltaPes is <20, 1 if deltaPes >=20
        all_patient_day_statics['deltaPes_binary'] = (all_patient_day_statics['deltaPes'] >= 20).astype(int)

        return all_patient_day_statics

    def _create_final_statics(self, patient_day_statics_list: list[pd.DataFrame], deltaPes_list: DeltaPesList) -> None:
        """
        Concantenates all the patient_day statics DataFrames into one large dataframe for all patient-days in the dataset

        Args:
            patient_day_statics_list (list[pd.DataFrame]): list of all triplets that have a statics file made for them
            deltaPes_list (DeltaPesList): list of all triplets in the dataset that have deltaPes calculated

        Returns:
            None:
        """

        all_patient_day_statics = pd.concat(patient_day_statics_list)
        all_patient_day_statics = all_patient_day_statics.set_index(['patient_id', 'day_id', 'breath_id'])

        # Now we'll produce the h0_with_adjacent_h1 column and merge
        all_patient_day_statics['h0_with_adjacent_h1'] = utils.find_h0s_with_adjacent_h1(
            statics=all_patient_day_statics,
            truth_col='Double Trigger',
            num_breaths=utils.NUM_BREATHS,
            patient_id_col='patient_id',
            day_id_col='day_id')

        # For multiclass classification of the underlying double trigger dyssynchronies,
        # we need an indicator of where there is a double trigger without an underlying dyssynchrony
        # to filter these out in split_train_val_test.py.

        # This next line searches for any column that starts with "Double Trigger" (including "Double Trigger" itself)
        # and if only one column (which would have to be "Double Trigger", since the others are derived from it)
        # is present (== 1) then we know there is a Double Trigger with no cooccurring dyssynchrony
        all_patient_day_statics['no_cooccurrence_double_trigger'] = (
                all_patient_day_statics.filter(regex='^Double Trigger *', axis=1).sum(axis=1) == 1).astype(int)

        # put deltaPes information into statics file
        all_patient_day_statics = TripletGenerator._add_deltaPes_to_statics(all_patient_day_statics, deltaPes_list)

        # save out statics files to triplets export directory
        all_patient_day_statics.to_hdf(Path(self._triplet_export_directory, 'statics.hdf'), key='statics')
        all_patient_day_statics.to_csv(Path(self._triplet_export_directory, 'statics.csv'))

        # save out statics files to statics directory
        all_patient_day_statics.to_hdf(Path(self._statics_directory, 'statics.hdf'), key='statics')
        all_patient_day_statics.to_csv(Path(self._statics_directory, 'statics.csv'))

    def _loop_through_triplets(self, subdir_name: str) -> tuple[pd.DataFrame, DeltaPesList]:
        """
        Loops through all triplets in a patient-day, saves them, and then returns information
        about deltaPes and statics for inclusion in the final statics

        Args:
            subdir_name (str): string of subdirectory where triplet should be stored

        Returns:
            patient_day_statics (pd.DataFrame): DataFrame with statics information about this patient-day
            deltaPes_list (DeltaPesList): list of all triplets in the dataset that have deltaPes calculated
        """
        # a temporary list of deltaPes calculations per breath
        deltaPes_list = []

        # get the patient day and set up patient_day directories
        patient_id, day_id, patient_day_dir, patient_day_output_dir, patient_day = self._get_patient_day(subdir_name)

        # create dyssynchrony mask
        patient_day = TripletGenerator._create_dyssynchrony_mask(patient_day)

        # create breath ids
        patient_day = TripletGenerator._create_breath_ids(patient_day)

        # Convert the time column to datetime format (be explicit with the format or it's slooowwwww)
        patient_day['TimeRel'] = pd.to_datetime(patient_day['TimeRel'], format='%H:%M:%S.%f')

        # setup statics file
        patient_day_statics = TripletGenerator._build_statics_file(
            patient_day,
            patient_id,
            day_id,
            subdir_name
        )

        # then get blacklist of breaths that are too long/short so we don't do anything with them
        breath_length_blacklist = utils.get_breath_length_blacklist(patient_day)

        breath_id_blacklist = np.unique(breath_length_blacklist)

        # Get dyssynchrony columns, so that the one hot columns span the entire triplet,
        # we will merge within the next loop
        one_hot_dyssynchronies = utils.one_hot_dyssynchronies(patient_day)

        # Loop through each breath
        for breath_id in range(1, patient_day['breath_id'].max()):

            if breath_id not in breath_id_blacklist:
                # create triplet
                triplet, triplet_csv_filename = TripletGenerator._build_triplet(patient_day,
                                                                    patient_day_output_dir,
                                                                    breath_id,
                                                                    one_hot_dyssynchronies)

                # calculate deltaPes
                triplet, deltaPes_list = TripletGenerator._calculate_deltaPes(triplet,
                                                                  breath_id,
                                                                  subdir_name,
                                                                  deltaPes_list
                                                                  )
                # save triplet to csv
                triplet.to_csv(triplet_csv_filename, index=False)

        # return first element in these lists, with multithreading there will only be one item per list
        return patient_day_statics, deltaPes_list

    def generate_triplets(self, multiprocessing: bool = False) -> tuple[Path, Path]:
        """
        Creates triplets in parallel (if flag is true), then saves them out and creates a statics file for them.

        Args:
            multiprocessing (bool): Boolean flag whether to multiprocess or not

        Returns:
            self.triplet_export_directory (Path): returns path to the directory where we saved all our triplets
            triplet_export_directory (Path): Path to the directory where we saved all our triplets
            statics_csv_output (Path): Path to the directory where we saved our statics
        """

        # setup directories
        self._batch_files_directory, \
            self._triplet_export_directory, \
            self._statics_directory = TripletGenerator._setup_directories(self._batch_files_directory)

        # Grab the triplet folders from their directories
        p = Path(self._batch_files_directory)
        subdir_names = [subdir.name for subdir in p.iterdir() if subdir.name not in utils.ERROR_DIRS]

        print(f'Creating CSVs from {len(subdir_names)} subdirectories of patient-days...')

        # find the number of workers to use
        n_workers = utils.num_workers(multiprocessing)
        print(f'{n_workers} Processes assigned to generating Triplets')
        # multiprocessing requires a list to loop over, a function object, and number of workers
        # for each subdir name in subdir names, get the results from each function call and append them to a list called results
        results = pqdm(subdir_names, self._loop_through_triplets, n_jobs=n_workers, desc='Patient-Days of Triplets Generated')

        # put together results
        patient_day_statics_list = [result[0] for result in results]
        # nested list comprehension (does .extend instead of .append essentially)
        # [variable_to_extend_with for OUTER LOOP (item in list) for INNER LOOP (variable_to_extend_with in list)]
        deltaPes_list = [deltaPes_result for result in results for deltaPes_result in result[1]]

        # make sure patient_day_statics_list has something, aka. there is a patient day with triplets. if not interrupt and raise exception
        assert len(patient_day_statics_list) > 0, f'There must be patients in the dataset with triplets. None found, exiting'

        # after looping through all triplet, create final statics file
        self._create_final_statics(patient_day_statics_list, deltaPes_list)

        triplet_export_directory = self._triplet_export_directory
        statics_csv_output = self._statics_directory

        return triplet_export_directory, statics_csv_output


# if running this file directly, only do triplet generation
if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with batch files')
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']

    # instantiate triplet generator class
    triplet_generator = TripletGenerator(input_directory)

    # run triplet generator
    export_directory, statics_csv_output = triplet_generator.generate_triplets()

    print(f'Triplets generated at {export_directory}')
    print(f"Statics file generated at {statics_csv_output}")
