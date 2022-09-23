
import os
import re
import tqdm
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

class Triplet_Generator:
    ''' This Class carries out all functions of the triplet generator.

        Inputs:
            batch_files_directory --> Path to directory where outputs from the batch annotator are kept
            Export directory --> Path to directory where outputs from the triplet generator should be kept

        Outputs:
            Triplets directory --> Directory with triplets generated for every patient-day csv provided in the import directory
            Statics File --> statics file with information on all patient-days provided in the import directory

        '''

    def __init__(self, batch_files_directory, triplet_export_directory="..\\intermediate_datasets\\triplets", statics_only = False):

        # # setup import and export directories
        self.batch_files_directory, self.triplet_export_directory, self.statics_output_path = self.setup_directories(batch_files_directory, triplet_export_directory)

        # if we want to only generate a statics file
        self.statics_only = statics_only

        # define constants
        self.time_col = 'TimeRel'
        self.breath_id_col = 'breath_id'
        self.dyssynchrony_mask_col = 'dyssynchrony_mask'
        self.min_length = 0.25
        self.max_length = 10
        self.min_breath_fraction = 0.75
        self.NUM_BREATHS = 3

        self.TA_CSV_SUFFIX = 'TriggersAndArtifacts.csv'
        self.DELINEATION_COLUMN = 't:Fsp'
        self.TRAINING_COLUMNS = ['Time', 'TimeRel', 's:Fsp', 's:Paw', 's:Pes']
        self.LABEL_CODE_COLUMNS = ['a:Fsp', 'a:Paw', 'a:Pes']
        self.TRIPLET_FILE_ID_COLUMNS = ['triplet_id', 'breath_id']

        self.DYSSYNCHRONY_IDS = {104: 'Autotrigger',
                            106: 'Ineffective Trigger',
                            107: 'Double Trigger',
                            108: 'Trigger Delay',
                            109: 'Flow Undershoot',
                            110: 'Delayed Termination',
                            111: 'Premature Termination',
                            112: 'Expiratory Asynchrony',
                            113: 'Other Asynchrony',
                            114: 'Reverse Trigger',
                            115: 'Flow Overshoot',
                            1: 'Other Asynchrony'}

        self.COOCCURRENCES_TO_CREATE_VIA_OR = {'Inadequate Support': ['Premature Termination', 'Flow Undershoot'],
                                          'General Inadequate Support': ['Premature Termination', 'Flow Undershoot',
                                                                         'Ineffective Trigger', 'Trigger Delay']}
        self.COOCCURRENCES_TO_CREATE_VIA_AND = {'Double Trigger Autotrigger': ['Double Trigger', 'Autotrigger'],
                                           'Double Trigger Reverse Trigger': ['Double Trigger', 'Reverse Trigger'],
                                           'Double Trigger Premature Termination': ['Double Trigger',
                                                                                    'Premature Termination'],
                                           'Double Trigger Flow Undershoot': ['Double Trigger', 'Flow Undershoot'],
                                           'Double Trigger Inadequate Support': ['Double Trigger',
                                                                                 'Inadequate Support']}

        self.KEEP_COLUMNS = self.TRIPLET_FILE_ID_COLUMNS + self.TRAINING_COLUMNS + [self.DELINEATION_COLUMN] + list(
            set(self.DYSSYNCHRONY_IDS.values())) + list(set(self.COOCCURRENCES_TO_CREATE_VIA_OR.keys())) + list(
            set(self.COOCCURRENCES_TO_CREATE_VIA_AND.keys())) + ['No Double Trigger', 'No Inadequate Support']


    def setup_directories(self, batch_files_directory, triplet_export_directory):
        # strip quotes
        batch_files_directory = batch_files_directory.replace('"', '').replace("'", '')
        triplet_export_directory = triplet_export_directory.replace('"', '').replace("'", '')

        # make export directory with timestamp
        triplet_export_directory = os.path.join(triplet_export_directory, str(datetime.now()).replace(':', '-').replace(' ', ','))
        os.makedirs(triplet_export_directory)

        # setup statics output path
        statics_output_path = os.path.join(triplet_export_directory, 'statics.csv')

        return batch_files_directory, triplet_export_directory, statics_output_path

    def get_patient_id(self, subdir_name):
        '''
        Extract the patient ID from a filename using regex.
        '''

        # Using this pattern scares me a little bit, because it basically will pick up any
        # number of digits following a "p" or "pt" and followed by an underscore.
        # It works for all of the current filenames, and the optional "t" is required because
        # some filenames only have "P###_" instead of "Pt###_", but this should be considered when
        # new files come in.
        patient_id_pattern = 'pt?(\d*)_'

        return re.findall(patient_id_pattern, subdir_name, flags=re.IGNORECASE)[0]

    def get_day_id(self, subdir_name):
        '''
        Extract the day ID from a filename using regex.
        '''

        # Again, a couple of weird cases (e.g. Pt219_Day2d_Asynchrony) force me to use
        # the "anything but a number" regex at the end of day_id_pattern.
        # This works for all current cases, but should be looked at carefully when more data
        # flows in (hopefully all filenames will be uniformized once this project takes off)
        day_id_pattern = 'day(\d*)[^\d]'

        return re.findall(day_id_pattern, subdir_name, flags=re.IGNORECASE)[0]


    def get_patient_day(self):

        # get the patient and day id
        self.patient_id = self.get_patient_id(self.subdir_name)
        self.day_id = self.get_day_id(self.subdir_name)


        self.patient_day_dir = os.path.join(self.batch_files_directory, self.subdir_name)
        self.patient_day_output_dir = os.path.join(self.triplet_export_directory, self.subdir_name)

        # setup patient_day directories
        if not os.path.exists(self.patient_day_output_dir):
            os.mkdir(self.patient_day_output_dir)

        csv_files = os.listdir(self.patient_day_dir)

        # Find which file is the TriggersAndArtifacts file we care about
        ta_file_index = np.where([self.TA_CSV_SUFFIX in csv for csv in csv_files])[0][0]

        self.patient_day = pd.read_csv(os.path.join(self.patient_day_dir, csv_files[ta_file_index]))

    def create_dyssynchrony_mask(self):

        # Create a mask to select only rows in which I should search for dyssynchronies
        # By creating column where rows between inspiration trigger (1) and expiration trigger (-1)
        # are equal to 1, which will be used for filtering
        self.patient_day['dyssynchrony_mask'] = self.patient_day[self.DELINEATION_COLUMN].cumsum()

        # If the first trigger in the patient day is an expiration (-1) then the cumsum will
        # need 1 added to the entire dyssynchrony_mask column
        if self.patient_day['dyssynchrony_mask'].min() == -1:
            self.patient_day['dyssynchrony_mask'] += 1

    def create_breath_ids(self):

        # Create breath ids by replacing expiration triggers
        self.patient_day['breath_id'] = self.patient_day[self.DELINEATION_COLUMN].replace(-1, 0)
        self.patient_day['breath_id'] = self.patient_day['breath_id'].cumsum()

    def create_cooccurrence_column_via_or(self, statics, cooccurrence_col, combine_col_list):
        '''
        Combines the two columns listed in combine_col_list into a new statics column
        named cooccurrence_col via boolean 'or'

        Args:
            statics (pd.DataFrame): The statics DataFrame to which a new column will be added
            cooccurrence_col (str): The name of the newly added column
            combine_col_list (list[str]): A list containing the two existing statics columns
                                          to be combined via boolean 'or'

        Returns:
            statics (pd.DataFrame): The statics DataFrame with a new column
        '''

        # DEPRECATED: We now have more than 2 columns to combine for generalized_inadequate_support
        # ~ statics[cooccurrence_col] = ((statics[combine_col_list[0]].fillna(0).astype(bool)) |
        # ~ (statics[combine_col_list[1]].fillna(0).astype(bool))).astype(int)

        # We'll take the max, which in this case acts as an
        # 'or' across the entire row of combine_col_list
        statics[cooccurrence_col] = statics[combine_col_list].max(axis=1).fillna(0)

        return statics

    def create_cooccurrence_column_via_and(self, statics, cooccurrence_col, combine_col_list):
        '''
        Combines the two columns listed in combine_col_list into a new statics column
        named cooccurrence_col via boolean 'and'

        Args:
            statics (pd.DataFrame): The statics DataFrame to which a new column will be added
            cooccurrence_col (str): The name of the newly added column
            combine_col_list (list[str]): A list containing the two existing statics columns
                                          to be combined via boolean 'and'

        Returns:
            statics (pd.DataFrame): The statics DataFrame with a new column
        '''

        # TODO: Generalize for n columns in combine_col_list (sum() == num elements in list)
        statics[cooccurrence_col] = ((statics[combine_col_list[0]].fillna(0).astype(bool)) &
                                     (statics[combine_col_list[1]].fillna(0).astype(bool))).astype(int)

        return statics

    def create_one_hot_dyssynchronies(self):
        '''
        Given the list of dyssynchrony codes, create a column for each and
        assign the entire breath 1 if the disynchrony's present and 0 if not.
        NOTE: It is assumed that patient_day_masked contains ONLY rows that should
        be searched for dyssynchrony codes (in the current implementation, this
        means rows between inspiration and expiration.)

        Args:
            min_breath_fraction (float): The minimum fraction of the time between a breath's
                                         inspiration and expiration that must contain the dyssynchrony
                                         code to qualify as that dyssynchrony type
        '''
        # We're only looking for dyssynchronies between inspiration and expiration, so we'll mask the df to start with
        patient_day_masked = self.patient_day[self.patient_day[self.dyssynchrony_mask_col] == 1]

        # Initialize a DataFrame for each of the columns that contain dyssynchrony codes
        one_hot_df_dict = {}
        for label_code_column in self.LABEL_CODE_COLUMNS:
            one_hot_df = pd.DataFrame(index=self.patient_day[self.breath_id_col].unique())

            for dissync_code in self.DYSSYNCHRONY_IDS.keys():
                # breath_contains_code = lambda breath: (breath[label_code_column] == dissync_code).max()

                # Create a column indicating whether or not the dissync is present
                dissync_present_df = patient_day_masked[[self.breath_id_col]].copy()
                dissync_present_df['dissync_present'] = (patient_day_masked[label_code_column] == dissync_code)
                dissync_present = dissync_present_df.groupby(self.breath_id_col)['dissync_present']
                # dissync_present = dissync_present.groupby(breath_id_col)

                one_hot_df[dissync_code] = ((dissync_present.sum() / dissync_present.size()) > self.min_breath_fraction) * 1

                # Compute the fraction of breath labeled with this dyssynchrony code and compare it with our min_breath_fraction
                # breath_contains_code = lambda breath: ((breath[label_code_column] == dissync_code).sum() / breath.shape[0]) > min_breath_fraction
                # Create a column with the same length as patient_day that contains a binary indicator of each dyssynchrony code
                # one_hot_df[dissync_code] = patient_day_masked.groupby(breath_id_col).apply(breath_contains_code) * 1

            # SPECIAL CASE: There are two codes for "Other Asynchrony" ([1, 113]) that must be turned into a single column
            one_hot_df[113] = one_hot_df[[1, 113]].max(axis=1)
            one_hot_df = one_hot_df.drop(columns=[1])

            one_hot_df = one_hot_df.rename(columns=self.DYSSYNCHRONY_IDS)

            one_hot_df_dict[label_code_column] = one_hot_df

        # TODO: Instead of dict, create a multi-index and do the below .max() natively in pandas across the proper axis
        full_one_hot_df = pd.DataFrame(data=np.stack([df for df in one_hot_df_dict.values()]).max(axis=0),
                                       columns=one_hot_df_dict[label_code_column].columns)

        # We'll create a column that's the inverse of "Double Trigger" called "No Double Trigger" to act as our final
        # multiclass label for the multiclass classification problem (so all rows have exactly one "1")
        # TODO: Speed comparison of the below line vs. 1 - full_one_hot_df['Double Trigger']
        full_one_hot_df['No Double Trigger'] = full_one_hot_df['Double Trigger'].map({0: 1, 1: 0})

        # !! WARNING !! - The OR operations MUST BE RUN before the AND operations OR ELSE DOOOOM.
        # AKA Inadequate Support will not be created and not be available in the AND operation

        # Combine columns like Double Trigger and Autotrigger into a single column for training
        # Start with the ones to combine using an 'or' operation
        for cooccurrence_col, combine_col_list in self.COOCCURRENCES_TO_CREATE_VIA_OR.items():
            full_one_hot_df = self.create_cooccurrence_column_via_or(full_one_hot_df, cooccurrence_col, combine_col_list)

        # Now create the columns combined via 'and'
        for cooccurrence_col, combine_col_list in self.COOCCURRENCES_TO_CREATE_VIA_AND.items():
            full_one_hot_df = self.create_cooccurrence_column_via_and(full_one_hot_df, cooccurrence_col, combine_col_list)

        # Since we'll train a multilabel model to predict the two underlying types of
        # Inadequate Support separately (as Ineffective Trigger and Flow Undershoot themselves)
        # we'll create an H0 column for that truthing scheme, which we can derive from the newly
        # created Inadequate Support column
        full_one_hot_df['No Inadequate Support'] = full_one_hot_df['Inadequate Support'].map({0: 1, 1: 0})

        return full_one_hot_df

    def create_breath_statics(self):
        '''
        Create a statics file for each breath that contains the following columns:
            - Breath ID
            - Start time
            - End time
            - Breath length
            - A binary column for each dyssynchrony/artifact type
        '''

        # Initialize an empty statics DataFrame
        statics = pd.DataFrame(index=self.patient_day[self.breath_id_col].unique())

        # Grab the first and last values of the time column as the start and end times for each breath
        statics['start_time'] = self.patient_day.groupby(self.breath_id_col).take([0])[self.time_col].droplevel(1)
        statics['expiration_time'] = self.patient_day[self.patient_day[self.DELINEATION_COLUMN] == -1].reset_index().set_index(self.breath_id_col)[self.time_col]
        statics['end_time'] = self.patient_day.groupby(self.breath_id_col).take([-1])[self.time_col].droplevel(1)

        # Calculate the length of each breath
        statics['length'] = statics['end_time'] - statics['start_time']
        statics['inspiration_length'] = statics['expiration_time'] - statics['start_time']
        statics['expiration_length'] = statics['end_time'] - statics['expiration_time']

        # Add in the one hot encoded dyssynchronies
        statics = pd.concat([statics, self.create_one_hot_dyssynchronies()], axis=1)

        statics.index.name = self.breath_id_col
        statics = statics.reset_index()
        return statics

    def build_statics_file(self):

        # Build the statics file, creating ID columns for the patient and day based on the filename
        self.patient_day_statics = self.create_breath_statics()

        self.patient_day_statics['patient_id'] = self.patient_id
        self.patient_day_statics['day_id'] = self.day_id
        self.patient_day_statics['original_subdirectory'] = self.subdir_name

        # Stack the statics files (they'll be merged once the loop ends)
        self.patient_day_statics_list.append(self.patient_day_statics)

    def _get_context_blackslist(self, blacklist, context_breaths):
        '''
        Given a blacklist, return the surrounding breaths that should also be blacklisted
        '''

        context_blacklist = []
        for blacklisted_id in blacklist:
            for i in range(1, context_breaths + 1):
                context_blacklist += [blacklisted_id + i,
                                      blacklisted_id - i]

        return context_blacklist

    def get_breath_length_blacklist(self, patient_day, num_breaths=3, min_length=0.25, max_length=10.0, time_col='TimeRel',
                                    breath_id_col='breath_id'):
        '''
        Identify breaths that are either too long or too short to be considered.

        params:
            patient_day (DataFrame): A patient's day worth of recorded breaths
            num_breaths (int): The number of breaths to include as context, including the breath itself (we're doing triplets now, so default is 3)
            min_length (float): Minimum length of a breath in seconds
            max_length (float): Maximum length of a breath in seconds
            time_col (str): Column name that contains datetime information in patient_day
        '''
        # TODO: Feed this function statics instead of patient_day
        context_breaths = int(np.floor(self.NUM_BREATHS / 2))

        min_length = pd.Timedelta(seconds=self.min_length)
        max_length = pd.Timedelta(seconds=self.max_length)

        breath_lengths = self.patient_day.groupby(self.breath_id_col).apply(
            lambda breath: breath[self.time_col].iloc[-1] - breath[self.time_col].iloc[0])

        breaths_too_short = breath_lengths[breath_lengths < min_length].index.values.tolist()
        breaths_too_long = breath_lengths[breath_lengths > max_length].index.values.tolist()

        too_short_context_blacklist = self._get_context_blackslist(breaths_too_short, context_breaths)
        too_long_context_blacklist = self._get_context_blackslist(breaths_too_long, context_breaths)

        return np.unique(breaths_too_short +
                         breaths_too_long +
                         too_short_context_blacklist +
                         too_long_context_blacklist).tolist()

    def build_triplet(self, breath_id, one_hot_dyssynchronies):

        # Take the breath before, current breath and breath after as a triplet
        triplet = self.patient_day[self.patient_day['breath_id'].isin([breath_id - 1,
                                                             breath_id,
                                                             breath_id + 1])].reset_index()

        # Create a triplet ID, which will just be the ID of the middle breath
        triplet['triplet_id'] = breath_id

        # Merge so that the one hot columns span the entire triplet
        triplet = triplet.merge(one_hot_dyssynchronies[one_hot_dyssynchronies.index == breath_id],
                                left_on='triplet_id',
                                right_index=True,
                                how='left')

        triplet = triplet[self.KEEP_COLUMNS]

        # Save to csv
        triplet.to_csv(os.path.join(self.patient_day_output_dir,
                                    'breath_{}.csv'.format(breath_id)),
                       index=False)

    def find_h0s_with_adjacent_h1(self, statics, truth_col, num_breaths=3, patient_id_col='patient_id', day_id_col='day_id'):
        '''

        '''
        # A lambda function to apply to the truth column that finds if current breath (middle breath)
        # is an h0 and either the previous or subsequent breath are h1
        is_h0_with_adjacent_h1 = lambda truth: 1 if truth[1] == 0 and (truth[0] == 1 or truth[2] == 1) else 0

        return statics.groupby(level=[patient_id_col, day_id_col])[truth_col].fillna(0).rolling(num_breaths,
                                                                                                center=True).apply(
            is_h0_with_adjacent_h1).fillna(0)

    def create_final_statics(self):

        self.all_patient_day_statics = pd.concat(self.patient_day_statics_list)
        self.all_patient_day_statics = self.all_patient_day_statics.set_index(['patient_id', 'day_id', 'breath_id'])

        # Now we'll produce the h0_with_adjacent_h1 column and merge
        self.all_patient_day_statics['h0_with_adjacent_h1'] = self.find_h0s_with_adjacent_h1(
            statics=self.all_patient_day_statics,
            truth_col='Double Trigger',
            num_breaths=self.NUM_BREATHS,
            patient_id_col='patient_id',
            day_id_col='day_id')

        # For mutliclass classification of the underlying double trigger dyssynchronies,
        # we need an indicator of where there is a double trigger without an underlying dyssynchrony
        # to filter these out in split_train_val_test.py.

        # This next line searches for any column that starts with "Double Trigger" (including "Double Trigger" itself)
        # and if only one column (which would have to be "Double Trigger", since the others are derived from it)
        # is present (== 1) then we know there is a Double Trigger with no cooccurring dyssynchrony
        self.all_patient_day_statics['no_cooccurrence_double_trigger'] = (
                self.all_patient_day_statics.filter(regex='^Double Trigger *', axis=1).sum(axis=1) == 1).astype(int)

        self.all_patient_day_statics.to_hdf(self.statics_output_path, key='statics')

    def create_triplets(self):
        # Grab all the subdirectories, which contain individual patient days, in the export directory
        subdir_names = os.listdir(self.batch_files_directory)

        # Here's where we'll accumulate all of the individual statics files within the loop
        self.patient_day_statics_list = []

        print(f'Creating CSVs from {len(subdir_names)} subdirectories of patient-days...')

        # Loop through each subdirectory
        for self.subdir_name in tqdm.tqdm(subdir_names):

            # get the patient day and set up patient_day directories
            self.get_patient_day()

            # create dyssynchrony mask
            self.create_dyssynchrony_mask()

            # create breath ids
            self.create_breath_ids()

            # Convert the time column to datetime format (be explicit with the format or it's slooowwwww)
            self.patient_day['TimeRel'] = pd.to_datetime(self.patient_day['TimeRel'], format='%H:%M:%S.%f')

            # setup statics file
            self.build_statics_file()

            # if we are not only creating a statics file
            if not self.statics_only:

                # then get blacklist of breaths that are too long/short so we don't do anything with them
                self.breath_length_blacklist = self.get_breath_length_blacklist(self.patient_day, self.NUM_BREATHS)

                self.breath_id_blacklist = np.unique(self.breath_length_blacklist)

                # Get dyssynchrony columns which, so that the one hot columns span the entire triplet,
                # we will merge within the next loop
                one_hot_dyssynchronies = self.create_one_hot_dyssynchronies()

                # Loop through each breath
                for breath_id in range(1, self.patient_day['breath_id'].max()):
                    if breath_id not in self.breath_id_blacklist:
                        # create triplet
                        self.build_triplet(breath_id, one_hot_dyssynchronies)

        # after looping through all triplet, create final statics file
        self.create_final_statics()

        return self.triplet_export_directory

# if running this file directly, only do do triplet processing
if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--export_directory', type=str, default="..\\intermediate_datasets\\triplets", help='Directory to export organized unannotated files for later processing')
    p.add_argument('--statics_only', type=str, default=False, help='If you only want the statics file with no triplets')
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']
    export_directory = args['export_directory']
    statics_only = args['statics_only']

    # instantiate batch annotator class
    triplet_generator = Triplet_Generator(input_directory, export_directory, statics_only)

    # run batch processor
    export_directory = triplet_generator.create_triplets()

    if not statics_only:
        print(f'Triplets generated at {export_directory}')
    print(f"Statics file generated at {os.path.join(export_directory, 'statics.csv')}")

