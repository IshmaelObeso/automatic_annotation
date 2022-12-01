import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from components.dataset_generation.utilities import utils, deltaPes_utils
from pqdm.processes import pqdm

class Triplet_Generator:
    ''' This Class carries out all functions of the triplet generator.

        Inputs:
            batch_files_directory --> Path to directory where outputs from the batch annotator are kept
            Export directory --> Path to directory where outputs from the triplet generator should be kept

        Outputs:
            Triplets directory --> Directory with triplets generated for every patient-day csv provided in the import directory
            Statics File --> statics file with information on all patient-days provided in the import directory(csv and hdf)

        '''

    def __init__(self, batch_files_directory: object, filter_file_info: object = None) -> object:
        """

        Args:
            batch_files_directory:
            filter_file_info:
        """
        # # setup import and export directories
        self.batch_files_directory, self.triplet_export_directory, self.statics_directory = self.setup_directories(batch_files_directory)
        # save filter file info
        self.filter_file_info = filter_file_info


    def setup_directories(self, batch_files_directory: object) -> object:
        """

        Args:
            batch_files_directory:

        Returns:

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


    def get_patient_day(self, subdir_name: object) -> object:
        """

        Args:
            subdir_name:

        Returns:

        """
        # get the patient and day id
        patient_id = utils.get_patient_id(subdir_name)
        day_id = utils.get_day_id(subdir_name)

        # get paths for patient day folder
        patient_day_dir = Path(self.batch_files_directory, subdir_name)
        patient_day_output_dir = Path(self.triplet_export_directory, subdir_name)

        patient_day_output_dir.mkdir(parents=True, exist_ok=True)

        # get all csv files in patient day folder
        csv_files = [x.name for x in patient_day_dir.iterdir()]

        # Find which file is the TriggersAndArtifacts file we care about
        ta_file_index = np.where([utils.TA_CSV_SUFFIX in csv for csv in csv_files])[0][0]

        patient_day = pd.read_csv(Path(patient_day_dir, csv_files[ta_file_index]))

        return patient_id, day_id, patient_day_dir, patient_day_output_dir, patient_day

    def create_dyssynchrony_mask(self, patient_day: object) -> object:
        """

        Args:
            patient_day:

        Returns:

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

    def create_breath_ids(self, patient_day: object) -> object:
        """

        Args:
            patient_day:

        Returns:

        """
        # Create breath ids by replacing expiration triggers
        patient_day['breath_id'] = patient_day[utils.DELINEATION_COLUMN].replace(-1, 0)
        patient_day['breath_id'] = patient_day['breath_id'].cumsum()

        return patient_day

    def build_statics_file(self, patient_day: object, patient_id: object, day_id: object, patient_day_statics_list: object, subdir_name: object) -> object:
        """

        Args:
            patient_day:
            patient_id:
            day_id:
            patient_day_statics_list:
            subdir_name:

        Returns:

        """
        # Build the statics file, creating ID columns for the patient and day based on the filename
        patient_day_statics = utils.create_breath_statics(patient_day)

        patient_day_statics['patient_id'] = patient_id
        patient_day_statics['day_id'] = day_id
        patient_day_statics['original_subdirectory'] = subdir_name

        # Stack the statics files (they'll be merged once the loop ends)
        patient_day_statics_list.append(patient_day_statics)

        return patient_day_statics, patient_day_statics_list

    def build_triplet(self, patient_day: object, patient_day_output_dir: object, breath_id: object, one_hot_dyssynchronies: object) -> object:
        """

        Args:
            patient_day:
            patient_day_output_dir:
            breath_id:
            one_hot_dyssynchronies:

        Returns:

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

        # Save to csv
        triplet.to_csv(triplet_csv_filename, index=False)

        return triplet, triplet_csv_filename

    def calculate_deltaPes(self, triplet: object, breath_id: object, subdir: object, deltaPes_list: object, triplet_csv_filename: object) -> object:
        """

        Args:
            triplet:
            breath_id:
            subdir:
            deltaPes_list:
            triplet_csv_filename:

        Returns:

        """
        try:
            deltaPes = deltaPes_utils.calculate_deltaPes(triplet_csv_filename)
        # if there is a problem calculating deltaPES, save deltaPES as NaN
        except:
            print(f'Error Calculating DeltaPES for pt {utils.get_patient_id(subdir)}, day {utils.get_day_id(subdir)}, breath {breath_id} saving deltaPES as NaN')
            deltaPes = np.nan

        deltaPes_list.append([deltaPes, subdir, breath_id])
        # get binary truth for deltaPES, 0 if <20, 1 if >= 20
        deltaPes_binary = int(deltaPes>=20)

        triplet['deltaPes'] = deltaPes
        triplet['deltaPes_binary'] = deltaPes_binary
        triplet['deltaPes_clipped'] = triplet['deltaPes'].clip(upper=40)

        return triplet, deltaPes_list

    def add_deltaPes_to_statics(self, all_patient_day_statics: object, deltaPes_list: object) -> object:
        """

        Args:
            all_patient_day_statics:
            deltaPes_list:
            has_deltaPes:

        Returns:

        """
        merge_columns = ['original_subdirectory', 'breath_id']

        # Save this to restore the index after the merge
        statics_index = all_patient_day_statics.index.names

        # merge deltaPes into statics
        deltaPes_df = pd.DataFrame(deltaPes_list, columns=['deltaPes'] + merge_columns)

        all_patient_day_statics = all_patient_day_statics.reset_index().merge(deltaPes_df,
                                                                              how='left',
                                                                              on=merge_columns).set_index(statics_index)

        # threshhold deltaPes, values above 40 will be clipped to 40
        all_patient_day_statics['deltaPes_clipped'] = all_patient_day_statics['deltaPes'].clip(upper=40)

        # binarize deltaPES in the statics file
        # 0 if deltaPes is <20, 1 if deltaPes >=20
        all_patient_day_statics['deltaPes_binary'] = (all_patient_day_statics['deltaPes'] >= 20).astype(int)

        return all_patient_day_statics

    def create_final_statics(self, patient_day_statics_list: object, deltaPes_list: object) -> object:
        """

        Args:
            patient_day_statics_list:
            deltaPes_list:
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

        # For mutliclass classification of the underlying double trigger dyssynchronies,
        # we need an indicator of where there is a double trigger without an underlying dyssynchrony
        # to filter these out in split_train_val_test.py.

        # This next line searches for any column that starts with "Double Trigger" (including "Double Trigger" itself)
        # and if only one column (which would have to be "Double Trigger", since the others are derived from it)
        # is present (== 1) then we know there is a Double Trigger with no cooccurring dyssynchrony
        all_patient_day_statics['no_cooccurrence_double_trigger'] = (
                all_patient_day_statics.filter(regex='^Double Trigger *', axis=1).sum(axis=1) == 1).astype(int)

        # put deltaPes information into statics file
        all_patient_day_statics = self.add_deltaPes_to_statics(all_patient_day_statics, deltaPes_list)

        # save out statics files to triplets export directory
        all_patient_day_statics.to_hdf(Path(self.triplet_export_directory, 'statics.hdf'), key='statics')
        all_patient_day_statics.to_csv(Path(self.triplet_export_directory, 'statics.csv'))

        # save out statics files to statics directory
        all_patient_day_statics.to_hdf(Path(self.statics_directory, 'statics.hdf'), key='statics')
        all_patient_day_statics.to_csv(Path(self.statics_directory, 'statics.csv'))

    def loop_through_triplets(self, subdir_name: object) -> object:
        """

        Args:
            subdir_name:

        Returns:

        """
        # Here's where we'll accumulate all the individual statics files within the loop
        patient_day_statics_list = []

        # a list to idenfiy which breaths actually had a deltaPes calculated
        has_deltaPes = []

        # a temporary list of detlaPes calculations per breath
        deltaPes_list = []

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
                triplet, deltaPes_list = self.calculate_deltaPes(triplet,
                                                                               breath_id,
                                                                               subdir_name,
                                                                               deltaPes_list,
                                                                               triplet_csv_filename)

        # return first element in these lists, with multithreading there will only be one item per list
        return patient_day_statics_list[0], deltaPes_list[0]

    def generate_triplets(self, multiprocessing: object = False) -> object:
        """

        Args:
            multiprocessing:

        Returns:

        """
        # Grab the triplet folders from their directories
        p = Path(self.batch_files_directory)
        subdir_names = [subdir.name for subdir in p.iterdir() if subdir.name not in utils.ERROR_DIRS]

        print(f'Creating CSVs from {len(subdir_names)} subdirectories of patient-days...')

        # find the number of workers to use
        n_workers = utils.num_workers(multiprocessing)
        # multiprocessing requires a list to loop over, a function object, and number of workers
        # for each subdir name in subdir names, get the results from each function call and append them to a list called results
        results = pqdm(subdir_names, self.loop_through_triplets, n_jobs=1, desc='Patient-Days of Triplets Generated')

        # initialize empty lists to build
        patient_day_statics_list = []
        deltaPes_list = []

        # put together all results
        for result in results:
            # each result has structure [patient_day_statics_list, delta_pes_list, has_deltaPes]
            patient_day_statics_list.append(result[0])
            deltaPes_list.append(result[1])

        # after looping through all triplet, create final statics file
        self.create_final_statics(patient_day_statics_list, deltaPes_list)

        return self.triplet_export_directory

# if running this file directly, only do do triplet generation
if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with batch files')
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']

    # instantiate triplet generator class
    triplet_generator = Triplet_Generator(input_directory)

    # run triplet generator
    export_directory = triplet_generator.generate_triplets()
    statics_csv_output = triplet_generator.statics_directory

    print(f'Triplets generated at {export_directory}')
    print(f"Statics file generated at {statics_csv_output}")
