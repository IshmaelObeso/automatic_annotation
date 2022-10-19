import tqdm
import numpy as np
from pathlib import Path
import shutil
from components.dataset_generation.utilities import utils


class Data_Cleaner:
    """
    Class that catches and fixes errors in the pipe before they cause issues with the processor classes,
     like the triplets class.
    """

    def __init__(self):

        pass

    def check_for_duplicate_pt_days(self, directory):

        """
        Function that checks directory for duplicate patient days,
         then moves the duplicates into the duplicates directory and print out a message

        :param directory: path to directory of patient days
        :return:
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



    def check_for_invalid_subdirs(self, directory):
        """
        Function that checks directory for a TriggersAndArtifacts file, if none is found move the directory
        to the invalid directory and print out a message

        :param directory: path to directory of patient days
        :return:
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
        if num_invalid > 0 :
            # after looping, move all invalid subdirectories into invalid directory
            # setup invalid subdir
            invalid_dir = Path(directory, 'invalid')
            invalid_dir.mkdir(parents=True, exist_ok=True)

            # loop through all subdirectories that need to be moved
            for invalid_subdir in invalid_subdirs:

                orig_invalid_subdir_path = Path(directory, invalid_subdir)
                new_invalid_subdir_path = Path(invalid_dir, invalid_subdir)
                shutil.move(orig_invalid_subdir_path, new_invalid_subdir_path)

            # print that duplicats were found, and which ones
            print(f'{num_invalid} patient days with no TriggersAndArtifacts File found! Moved to {invalid_dir}')
         # else do nothing and print
        else:
            print('No patient days without TriggerAndArtifacts File Found!')