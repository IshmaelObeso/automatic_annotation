import os
import re
import tqdm
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import shutil
from .utilities import utils, deltaPes_utils
from datetime import datetime

class Error_Handler:
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

        # move all duplicate subdirectories to duplicates directory
        # setup duplicates subdir
        duplicates_subdir = os.path.join(directory, 'duplicates')
        if not os.path.exists(duplicates_subdir):
            os.mkdir(duplicates_subdir)

        # loop through all subdirectories that need to be moved
        for subdirs in multi_pt_days.values():

            # loop through all subdirectories
            for subdir in subdirs:
                subdir_path = os.path.join(directory, subdir)
                duplicates_path = os.path.join(duplicates_subdir, subdir)
                shutil.move(subdir_path, duplicates_path)


        # print that duplicates were found, and which ones
        print(f'{len(multi_pt_days.keys())} Duplicate patient days found! Moved to {duplicates_subdir}')



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
            subdir_files = os.listdir(os.path.join(directory, subdir_name))

            # look for a TriggersandArtifacts file
            try:
                ta_file_index = np.where([utils.TA_CSV_SUFFIX in csv for csv in subdir_files])[0][0]

            # if not found, add subdir to invalid files
            except:
                invalid_subdirs.append(subdir_name)

        # after looping, move all invalid subdirectories into invalid directory
        # setup invalid subdir
        invalid_dir = os.path.join(directory, 'invalid')
        if not os.path.exists(invalid_dir):
            os.mkdir(invalid_dir)

        # loop through all subdirectories that need to be moved
        for invalid_subdir in invalid_subdirs:

            orig_invalid_subdir_path = os.path.join(directory, invalid_subdir)
            new_invalid_subdir_path = os.path.join(invalid_dir, invalid_subdir)
            shutil.move(orig_invalid_subdir_path, new_invalid_subdir_path)

        # print that duplicats were found, and which ones
        print(f'{len(invalid_subdirs)} patient days with no TriggersAndArtifacts File found! Moved to {invalid_dir}')
