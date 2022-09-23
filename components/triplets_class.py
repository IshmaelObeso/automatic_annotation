
import os
import re
import tqdm
import pandas as pd
import numpy as np
import argparse
import utils
import deltaPes_utils
from datetime import datetime

class Triplet_Generator:
    ''' This Class carries out all functions of the triplet generator.

        Inputs:
            batch_files_directory --> Path to directory where outputs from the batch annotator are kept
            Export directory --> Path to directory where outputs from the triplet generator should be kept

        Outputs:
            Triplets directory --> Directory with triplets generated for every patient-day csv provided in the import directory
            Statics File --> statics file with information on all patient-days provided in the import directory(csv and hdf)

        '''

    def __init__(self, batch_files_directory, triplet_export_directory="..\\datasets\\triplets"):

        # # setup import and export directories
        self.batch_files_directory, self.triplet_export_directory, self.statics_output_path_csv, self.statics_output_path_hdf = self.setup_directories(batch_files_directory, triplet_export_directory)


    def setup_directories(self, batch_files_directory, triplet_export_directory):
        # strip quotes
        batch_files_directory = batch_files_directory.replace('"', '').replace("'", '')
        triplet_export_directory = triplet_export_directory.replace('"', '').replace("'", '')

        # make export directory with timestamp
        triplet_export_directory = os.path.join(triplet_export_directory, str(datetime.now()).replace(':', '-').replace(' ', ','))
        os.makedirs(triplet_export_directory)

        # setup statics output path
        statics_output_path_csv = os.path.join(triplet_export_directory, 'statics.csv')
        statics_output_path_hdf = os.path.join(triplet_export_directory, 'statics.hdf')

        return batch_files_directory, triplet_export_directory, statics_output_path_csv, statics_output_path_hdf

    def get_patient_day(self, subdir_name):

        # get the patient and day id
        patient_id = utils.get_patient_id(subdir_name)
        day_id = utils.get_day_id(subdir_name)


        patient_day_dir = os.path.join(self.batch_files_directory, subdir_name)
        patient_day_output_dir = os.path.join(self.triplet_export_directory, subdir_name)

        # setup patient_day directories
        if not os.path.exists(patient_day_output_dir):
            os.mkdir(patient_day_output_dir)

        csv_files = os.listdir(patient_day_dir)

        # Find which file is the TriggersAndArtifacts file we care about
        ta_file_index = np.where([utils.TA_CSV_SUFFIX in csv for csv in csv_files])[0][0]

        patient_day = pd.read_csv(os.path.join(patient_day_dir, csv_files[ta_file_index]))

        return patient_id, day_id, patient_day_dir, patient_day_output_dir, patient_day

    def create_dyssynchrony_mask(self, patient_day):

        # Create a mask to select only rows in which I should search for dyssynchronies
        # By creating column where rows between inspiration trigger (1) and expiration trigger (-1)
        # are equal to 1, which will be used for filtering
        patient_day['dyssynchrony_mask'] = patient_day[utils.DELINEATION_COLUMN].cumsum()

        # If the first trigger in the patient day is an expiration (-1) then the cumsum will
        # need 1 added to the entire dyssynchrony_mask column
        if patient_day['dyssynchrony_mask'].min() == -1:
            patient_day['dyssynchrony_mask'] += 1

        return patient_day

    def create_breath_ids(self, patient_day):

        # Create breath ids by replacing expiration triggers
        patient_day['breath_id'] = patient_day[utils.DELINEATION_COLUMN].replace(-1, 0)
        patient_day['breath_id'] = patient_day['breath_id'].cumsum()

        return patient_day

    def build_statics_file(self, patient_day, patient_id, day_id, patient_day_statics_list, subdir_name):

        # Build the statics file, creating ID columns for the patient and day based on the filename
        patient_day_statics = utils.create_breath_statics(patient_day)

        patient_day_statics['patient_id'] = patient_id
        patient_day_statics['day_id'] = day_id
        patient_day_statics['original_subdirectory'] = subdir_name

        # Stack the statics files (they'll be merged once the loop ends)
        patient_day_statics_list.append(patient_day_statics)

        return patient_day_statics, patient_day_statics_list

    def build_triplet(self, patient_day, patient_day_output_dir, breath_id, one_hot_dyssynchronies):

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

        triplet_csv_filename = os.path.join(patient_day_output_dir,
                                    'breath_{}.csv'.format(breath_id))

        # Save to csv
        triplet.to_csv(triplet_csv_filename, index=False)

        return triplet, triplet_csv_filename

    def calculate_deltaPes(self, triplet, breath_id, subdir, deltaPes_list, has_deltaPes, triplet_csv_filename):

        try:
            deltaPes = deltaPes_utils.calculate_deltaPes(triplet_csv_filename)
            deltaPes_list.append([deltaPes, subdir, breath_id])
            has_deltaPes.append([subdir, breath_id])
        # if there is a problem calculating deltaPES, save deltaPES as NaN
        except:
            print(f'Error Calculating DeltaPES for triplet: {breath_id}, saving deltaPES as NaN')
            deltaPes = np.nan

        # get binary truth for deltaPES, 0 if <20, 1 if >= 20
        deltaPes_binary = int(deltaPes>=20)

        triplet['deltaPes'] = deltaPes
        triplet['deltaPes_binary'] = deltaPes_binary
        triplet['deltaPes_clipped'] = triplet['deltaPes'].clip(upper=40)

        return triplet, deltaPes_list, has_deltaPes

    def add_deltaPes_to_statics(self, all_patient_day_statics, deltaPes_list, has_deltaPes):

        merge_columns = ['original_subdirectory', 'breath_id']

        has_deltaPes = pd.DataFrame(has_deltaPes, columns=merge_columns)
        has_deltaPes['has_deltaPes'] = 1

        # Save this to restore the index after the merge
        statics_index = all_patient_day_statics.index.names

        # Merge the has_deltaPes column into statics
        all_patient_day_statics = all_patient_day_statics.reset_index().merge(has_deltaPes,
                                                                              how='left',
                                                                              on=merge_columns).set_index(statics_index)

        # merge deltaPes into statics
        deltaPes_df = pd.DataFrame(deltaPes_list, columns=['deltaPes'] + merge_columns)

        all_patient_day_statics = all_patient_day_statics.reset_index().merge(deltaPes_df,
                                                                              how='left',
                                                                              on=merge_columns).set_index(statics_index)

        # The breaths that have deltaPes have been flagged and the rest should be 0
        all_patient_day_statics['has_deltaPes'] = all_patient_day_statics['has_deltaPes'].fillna(0)

        # threshhold deltaPes, values above 40 will be clipped to 40
        all_patient_day_statics['deltaPes_clipped'] = all_patient_day_statics['deltaPes'].clip(upper=40)

        # binarize deltaPES in the statics file
        # 0 if deltaPes is <20, 1 if deltaPes >=20
        all_patient_day_statics['deltaPes_binary'] = (all_patient_day_statics['deltaPes'] >= 20).astype(int)

        return all_patient_day_statics

    def create_final_statics(self, patient_day_statics_list, deltaPes_list, has_deltaPes):

        all_patient_day_statics = pd.concat(patient_day_statics_list)
        all_patient_day_statics = all_patient_day_statics.set_index(['patient_id', 'day_id', 'breath_id'])

        # Now we'll produce the h0_with_adjacent_h1 column and merge
        all_patient_day_statics['h0_with_adjacent_h1'] = utils.find_h0s_with_adjacent_h1(
            statics=all_patient_day_statics,
            truth_col='Double Trigger',
            num_breaths=utils.NUM_BREATHS,
            patient_id_col='patient_id',
            day_id_col='day_id')

        # For mutliclass classification of the underlying double trigger dyssynchronies,
        # we need an indicator of where there is a double trigger without an underlying dyssynchrony
        # to filter these out in split_train_val_test.py.

        # This next line searches for any column that starts with "Double Trigger" (including "Double Trigger" itself)
        # and if only one column (which would have to be "Double Trigger", since the others are derived from it)
        # is present (== 1) then we know there is a Double Trigger with no cooccurring dyssynchrony
        all_patient_day_statics['no_cooccurrence_double_trigger'] = (
                all_patient_day_statics.filter(regex='^Double Trigger *', axis=1).sum(axis=1) == 1).astype(int)

        # put deltaPes information into statics file
        all_patient_day_statics = self.add_deltaPes_to_statics(all_patient_day_statics, deltaPes_list, has_deltaPes)

        all_patient_day_statics.to_hdf(self.statics_output_path_hdf, key='statics')
        all_patient_day_statics.to_csv(self.statics_output_path_csv)

    def generate_triplets(self):
        # Grab all the subdirectories, which contain individual patient days, in the export directory
        subdir_names = os.listdir(self.batch_files_directory)

        # Here's where we'll accumulate all of the individual statics files within the loop
        patient_day_statics_list = []

        # a list to idenfiy which breaths actually had a deltaPes calculated
        has_deltaPes = []

        # a temporary list of detlaPes calculations per breath
        deltaPes_list = []

        print(f'Creating CSVs from {len(subdir_names)} subdirectories of patient-days...')

        # Loop through each subdirectory
        for subdir_name in tqdm.tqdm(subdir_names):

            # get the patient day and set up patient_day directories
            patient_id, day_id, patient_day_dir, patient_day_output_dir, patient_day = self.get_patient_day(subdir_name)

            # create dyssynchrony mask
            patient_day = self.create_dyssynchrony_mask(patient_day)

            # create breath ids
            patient_day = self.create_breath_ids(patient_day)

            # Convert the time column to datetime format (be explicit with the format or it's slooowwwww)
            patient_day['TimeRel'] = pd.to_datetime(patient_day['TimeRel'], format='%H:%M:%S.%f')

            # setup statics file
            patient_day_statics, patient_day_statics_list = self.build_statics_file(
                patient_day,
                patient_id,
                day_id,
                patient_day_statics_list,
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
                    triplet, triplet_csv_filename = self.build_triplet(patient_day,
                                       patient_day_output_dir,
                                       breath_id,
                                       one_hot_dyssynchronies)

                    # calculate deltaPes
                    triplet, deltaPes_list, has_deltaPes = self.calculate_deltaPes(triplet,
                                                      breath_id,
                                                      subdir_name,
                                                      deltaPes_list,
                                                      has_deltaPes,
                                                      triplet_csv_filename)

        # after looping through all triplet, create final statics file
        self.create_final_statics(patient_day_statics_list, deltaPes_list, has_deltaPes)

        return self.triplet_export_directory

# if running this file directly, only do do triplet generation
if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--export_directory', type=str, default="..\\datasets\\triplets", help='Directory to export organized unannotated files for later processing')
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']
    export_directory = args['export_directory']

    # instantiate triplet generator class
    triplet_generator = Triplet_Generator(input_directory, export_directory)

    # run triplet generator
    export_directory = triplet_generator.generate_triplets()
    statics_csv_output = triplet_generator.statics_output_path_csv

    print(f'Triplets generated at {os.path.abspath(export_directory)}')
    print(f"Statics file generated at {os.path.abspath(statics_csv_output)}")
