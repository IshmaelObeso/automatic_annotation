import tqdm
import numpy as np
import pandas as pd
import shutil

from pathlib import Path
from components.dataset_generation.utilities import utils
from typing import Union

# define some complex types for typehints
FilterFileInfo = dict[[str, bool], [str, str], dict[str, str], [str, str]]

class DataCleaner:
    """
    Catches and fixes errors in the pipe before they cause issues with the processor classes, like the triplets class.

    Attributes:
        _filter_file_info (FilterFileInfo): dict that contains information about the filter file (if it is included), including
                                what columns to filter over
        _parent_directory (Path): Path to the parent directory, where we will save the filter file if it exists
        _include_pt_days (list[tuple[int, int]]): list of patient days to include in the dataset

    """

    def __init__(self, filter_file_info: FilterFileInfo = None, parent_directory: Path = None) -> None:
        """
        Sets initial class attributes for use if this class gets called in a loop somewhere

        Args:
            filter_file_info (filter_file_info): dict that contains information about the filter file (if it is included), including
                                what columns to filter over
            parent_directory (Path): Path to the parent directory, where we will save the filter file if it exists

        Returns:
            None:
        """

        # save info about filtered triplets if given
        self._filter_file_info = filter_file_info
        self._parent_directory = parent_directory

        # if we want to use the filter file info, generate the pt days to exclude
        try:
            if self._filter_file_info['use']:
                self._include_pt_days = self._get_include_pt_days(parent_directory)

        # except if filter file info is None
        except TypeError:
            pass

    def _get_include_pt_days(self, parent_directory: Path) -> list[tuple[int, int]]:
        """
        This function takes the filter file info and saves a list of patient days to include in our dataset

        Args:
            parent_directory (Path): Path to the parent directory, where we will save the filter file if it exists

        Returns:
            include_pt_days (list[tuple[int, int]]): list of patient-days to include in our dataset
        """
        # make sure filter filepath exists
        assert self._filter_file_info['filepath'] is not None, 'Filter file filepath must exist if use_filter_file = True'

        # get the filepath to csv filter file
        csv_filepath = self._filter_file_info['filepath']
        # turn info Path object if not already
        csv_filepath = Path(csv_filepath)
        # load csv into dataframe
        filtering_file = pd.read_excel(csv_filepath, engine='openpyxl')

        # get the columns and values we should filter over
        columns_and_exclude_values = self._filter_file_info['exclude_columns_and_values']
        # this dictionary must have values if the 'use' variable is true
        assert len(
            columns_and_exclude_values), 'If filtering with csv file, must include columns to check and values to exclude'

        # get a list of patient_days that exist in the file
        filtering_file['patient_id'] = filtering_file['File'].apply(lambda row: int(utils.get_patient_id(row)))
        filtering_file['day_id'] = filtering_file['File'].apply(lambda row: int(utils.get_day_id(row)))
        filtering_file['patient_day'] = tuple(zip(filtering_file['patient_id'], filtering_file['day_id']))

        filtering_file_exclude = []

        # get a list of those patient_days that we should filter out
        for column, exclude_value in columns_and_exclude_values.items():

            # if value is string, it must be either 'NaN' or 'not NaN'
            if isinstance(exclude_value, str):
                # make sure
                assert (exclude_value == 'NaN') or (
                            exclude_value == 'not NaN'), 'If exclude value is string, it must be either "NaN" or "not NaN" '

                # if string is NaN, find rows with nan
                if exclude_value == 'NaN':
                    filtering_file_subset = filtering_file[filtering_file[column].isna()]
                # if string is 'not Nan' find rows that are not nan
                elif exclude_value == 'not NaN':
                    filtering_file_subset = filtering_file[filtering_file[column].notna()]

            else:
                filtering_file_subset = filtering_file[filtering_file[column] != exclude_value]

            filtering_file_exclude.append(filtering_file_subset)

        # concat all rows we should exclude
        filtering_file_exclude = pd.concat(filtering_file_exclude).drop_duplicates()

        # get the unique patient days to exclude
        exclude_pt_days = pd.Index(filtering_file_exclude['patient_day'].unique())

        # get all patient days in the file
        all_pt_days = pd.Index(filtering_file['patient_day'].unique())

        # now filter out the patient days we want to exclude, and save all the patient days to include
        include_pt_days = all_pt_days.difference(exclude_pt_days).tolist()

        # save filter file we used to our directory
        filtering_file.to_excel(Path(parent_directory, 'filter_file_excel.xlsx'))
        filtering_file.to_csv(Path(parent_directory, 'filter_file_csv.csv'))

        return include_pt_days

    def check_for_validity(self, subdir_name: str) -> bool:
        """
        Function that checks a breath to see if we should include it in our dataset based on the filter file

        Args:
            subdir_name: string that stores the name of the patient-day subdirectory we are working with

        Returns:
            bool: True if we should keep breath, False if we should not keep breath

        """
        # get the patient_id, day_id, and breath_id of the triplet
        patient_id = utils.get_patient_id(subdir_name)
        day_id = utils.get_day_id(subdir_name)

        # save triplet information to check against csv
        patient_day = (int(patient_id), int(day_id))

        # check if filter file info exists, it always should. if not return false for every triplet
        if self._filter_file_info is not None:

            # check if we should filter out breaths
            if self._filter_file_info['use']:

                # now check if the pt day we are checking should be included in the dataset
                if patient_day in self._include_pt_days:

                    return True

                else:

                    return False

            # if we shouldn't use filter, return false for every breath
            else:
                return False

        else:
            return False

    @staticmethod
    def check_for_duplicate_pt_days(directory: Path) -> None:
        """
        Method that checks directory for duplicate patient days,
        then moves the duplicates into the duplicates directory and print out a message

        Args:
            directory (Path): Path to the directory we want to check for duplicate patient-days

        Returns:
            None:
        """

        # Grab all the folders from their directories
        p = Path(directory)
        subdir_names = [subdir.name for subdir in p.iterdir() if subdir.name not in utils.ERROR_DIRS]

        # instantiate dictionary of pt days to subdir names
        pt_day_dict = {}

        # loop through each subdirectory
        for subdir_name in tqdm.tqdm(subdir_names):

            # get the patient and day id
            patient_id = utils.get_patient_id(subdir_name)
            day_id = utils.get_day_id(subdir_name)

            # define patient_day
            patient_day = (patient_id, day_id)

            # if patient_day doesn't exist already, add it to dictionary
            if patient_day not in pt_day_dict.keys():

                pt_day_dict[patient_day] = [subdir_name]

            elif patient_day in pt_day_dict.keys():

                # if patient_day already exists, append subdir to key
                pt_day_dict[patient_day].append(subdir_name)

        # after looping through all directories, check if any pt_days have multiple subdirectories assigned
        multi_pt_days = {k: v for (k, v) in pt_day_dict.items() if len(pt_day_dict[k]) > 1}

        num_duplicates = len(multi_pt_days.keys())
        # if duplicates are found
        if num_duplicates > 0:
            # move all duplicate subdirectories to duplicates directory
            # setup duplicates subdir if duplicates are found
            duplicates_subdir = Path(directory, 'duplicates')

            duplicates_subdir.mkdir(parents=True, exist_ok=True)

            # loop through all subdirectories that need to be moved
            for subdirs in multi_pt_days.values():

                # loop through all subdirectories
                for subdir in subdirs:
                    subdir_path = Path(directory, subdir)
                    duplicates_path = Path(duplicates_subdir, subdir)
                    shutil.move(subdir_path, duplicates_path)

            # print that duplicates were found, and which ones
            print(f'{num_duplicates} Duplicate patient days found! Moved to {duplicates_subdir}')
        else:
            print('No Duplicate Patient Days Found!')

    @staticmethod
    def check_for_invalid_subdirs(directory: Path) -> tuple[int, Union[Path, None]]:
        """
        Method that checks directory for a TriggersAndArtifacts file, if none is found move the directory
        to the invalid directory and print out a message

        Args:
            directory (Path): Path to the directory we want to check for duplicate patient-days

        Returns:
            None:

        """
        # Grab all the folders from their directories
        p = Path(directory)
        subdir_names = [subdir.name for subdir in p.iterdir() if subdir.name not in utils.ERROR_DIRS]

        # instantiate list of invalid subdirectories
        invalid_subdirs = []

        # loop through each subdirectory
        for subdir_name in tqdm.tqdm(subdir_names):

            # get all files in subdirectory
            subdir_path = Path(directory, subdir_name)
            subdir_files = [x.name for x in subdir_path.iterdir() if x.is_file()]


            # look for a TriggersandArtifacts file
            try:
                ta_file_index = np.where([utils.TA_CSV_SUFFIX in csv for csv in subdir_files])[0][0]

            # if not found, add subdir to invalid files
            except:
                invalid_subdirs.append(subdir_name)

        num_invalid = len(invalid_subdirs)
        # if invalid patient days are found
        if num_invalid > 0:
            # after looping, move all invalid subdirectories into invalid directory
            # setup invalid subdir
            invalid_dir = Path(directory, 'invalid')
            # if the invalid subdir exists already, delete it and its contents before remaking it and filling it
            if invalid_dir.is_dir():
                shutil.rmtree(invalid_dir)
            # setup directory
            invalid_dir.mkdir(parents=True, exist_ok=True)

            # loop through all subdirectories that need to be moved
            for invalid_subdir in invalid_subdirs:

                # grab the files from those subdirectories and move them into the invalid folder
                orig_invalid_subdir_path = Path(directory, invalid_subdir)
                orig_invalid_file_paths = [f for f in orig_invalid_subdir_path.iterdir() if f.is_file()]

                # move each file from subdir into invalid folder (There should only be 1 per folder, but loop just in case)
                for orig_invalid_file_path in orig_invalid_file_paths:

                    new_invalid_file_path = Path(invalid_dir, orig_invalid_file_path.name)
                    shutil.move(orig_invalid_file_path, new_invalid_file_path)

            # print that duplicates were found, and which ones
            print(f'{num_invalid} patient days with no TriggersAndArtifacts File found! Moved to {invalid_dir}')

            return num_invalid, invalid_dir

        # else do nothing and print
        else:
            print('No patient days without TriggerAndArtifacts File Found!')
            return num_invalid, None

