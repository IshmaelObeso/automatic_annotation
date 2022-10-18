import pandas as pd
import shutil
from .utilities import utils
from pathlib import Path

class Annotated_Dataset_Generator:

    """ Class that generates a dataset of raw patient
     files with their corresponding annotations inside an artifact file"""


    def __init__(self, raw_files_directory, spectral_triplets_directory, predictions_export_directory, threshold=.804):

        # setup directories
        self.annotated_dataset_directory, self.raw_files_directory, self.spectral_triplets_directory, self.spectral_statics_filepath = self.setup_directories(raw_files_directory, spectral_triplets_directory)

        # load the predictions dataframe
        self.predictions_df = self.load_predictions_dataframe(predictions_export_directory)

        # set threshold for predictions
        self.threshold = threshold

    def setup_directories(self, raw_files_directory, spectral_triplets_directory):

        # define paths
        raw_files_directory = Path(raw_files_directory.replace('"', '').replace("'", ''))
        spectral_triplets_directory = Path(spectral_triplets_directory)

        # make annotated dataset directory
        spectral_triplets_parent_directory = spectral_triplets_directory.parents[0]
        annotated_dataset_directory = Path(spectral_triplets_parent_directory, 'annotated_dataset')

        annotated_dataset_directory.mkdir(parents=True, exist_ok=True)

        # get the spectral triplet statics file path
        spectral_statics_filepath = Path(spectral_triplets_directory, 'spectral_statics.csv')

        return annotated_dataset_directory.resolve(),\
               raw_files_directory.resolve(),\
               spectral_triplets_directory.resolve(),\
               spectral_statics_filepath.resolve()

    def load_predictions_dataframe(self, predictions_export_directory):
        """ load the predictions dataframe from the csv of predictions and sets up indexes"""

        predictions_csv_filepath = Path(predictions_export_directory, 'predictions.csv')

        predictions_df = pd.read_csv(predictions_csv_filepath)

        # convert columns to string
        predictions_df[['patient_id', 'day_id']] = predictions_df[['patient_id', 'day_id']].astype(str)
        # set index for statics
        predictions_df = predictions_df.set_index(['patient_id', 'day_id', 'breath_id'])

        return predictions_df

    def get_breath_times(self):

        """ gets start and expiration times of breaths with associated predictions """

        # load spectral statics file
        spectral_statics = pd.read_csv(self.spectral_statics_filepath)
        # convert columns to string
        spectral_statics[['patient_id', 'day_id']] = spectral_statics[['patient_id', 'day_id']].astype(str)
        # set index for statics
        spectral_statics = spectral_statics.set_index(['patient_id', 'day_id', 'breath_id'])

        # get rows from statics file that have corresponding uid in preds dataframe
        spectral_statics = spectral_statics.loc[self.predictions_df.index]

        # grab the start and expiration times of breaths in those rows
        spectral_statics = spectral_statics[['start_time', 'end_time']]

        # convert start time and expiration time columns to seconds
        spectral_statics['start_time'] = pd.to_datetime(spectral_statics['start_time'])
        spectral_statics['end_time'] = pd.to_datetime(spectral_statics['end_time'])

        spectral_statics['start_time'] = spectral_statics['start_time'].dt.second + (spectral_statics['start_time'].dt.minute * 60) + (spectral_statics['start_time'].dt.hour * 3600) + ((spectral_statics['start_time'].dt.day -1) * 86400)
        spectral_statics['end_time'] = spectral_statics['end_time'].dt.second + (spectral_statics['end_time'].dt.minute * 60) + (spectral_statics['end_time'].dt.hour * 3600) + ((spectral_statics['end_time'].dt.day - 1) * 86400)

        # merge with preds df
        self.predictions_df = pd.concat([self.predictions_df, spectral_statics], axis=1)


    def copy_raw_files(self):

        """ Copies raw files from raw files directory to annotated dataset directory,
         only copies raw files that we have predictions for  """

        # grab all csv files from raw directory
        raw_csv_files = list(self.raw_files_directory.glob('*.csv'))

        # get all patient_days that we have predictions for
        patient_days = self.predictions_df.index.droplevel(2)
        patient_days = patient_days.unique().tolist()

        # make list of patient day csv files that we have predictions for
        patient_day_files_with_predictions = []

        for patient_day_file in raw_csv_files:

            # get filename from path
            patient_day_filename = patient_day_file.name

            # get patient and day id of file
            patient_id = utils.get_patient_id(patient_day_filename)
            day_id = utils.get_day_id(patient_day_filename)
            pt_day = (patient_id, day_id)

            # see if that patient day exists in our predictions file
            if pt_day in patient_days:
                patient_day_files_with_predictions.append(patient_day_file)

        # copy all patient day files that we have predictions for into new annotated dataset directory
        for patient_day_file in patient_day_files_with_predictions:

            # define desired filepath
            new_patient_day_filepath = Path(self.annotated_dataset_directory, patient_day_file.name)
            # copy file
            shutil.copy(patient_day_file, new_patient_day_filepath)

        return patient_day_files_with_predictions

    def create_art_files(self):

        """ creates artifact files for every patient day we have predictions for """

        # get a list of all patient day files that we have predictions for and copy any raw patient day files we have
        # predictions for into the annotated dataset directory
        patient_day_files_with_predictions = self.copy_raw_files()

        # merge breath times into the predictions dataframe
        self.get_breath_times()

        # get a predictions df with thresholded preds
        threshold_df = self.threshold_predictions()



        # for every patient day we have predictions for
        for patient_day_file in patient_day_files_with_predictions:

            # get the patient, day of that file
            # get filename from path
            patient_day_filename = patient_day_file.name

            # get patient and day id of file
            patient_id = utils.get_patient_id(patient_day_filename)
            day_id = utils.get_day_id(patient_day_filename)
            pt_day = (patient_id, day_id)

            # get the predictions from that patient day
            pt_day_preds = threshold_df.loc[pt_day]

            # get rows where there was a dyssynchrony
            pt_day_preds_dyssynchrony_index =pt_day_preds[pt_day_preds['prediction'] == 1].index

            # get filepath to the new dataset directory with patient_day_filename
            patient_day_filepath = Path(self.annotated_dataset_directory, patient_day_file.name)

            # build an artifacts file for that patient day
            artifacts_filepath = patient_day_filepath.with_suffix('.art')

            with open(artifacts_filepath, 'w') as file:
                # write the column names
                file.write('begin,end,signal,code\n')

                # for every dyssynchrony for that patient day
                for index in pt_day_preds_dyssynchrony_index:

                    begin = pt_day_preds.loc[index]['start_time']
                    end = pt_day_preds.loc[index]['end_time']

                    # signal is AirwayPressure for Double Trigger
                    signal = 'AirwayPressure'

                    # code is 107 for Double Trigger
                    code = '107'

                    # also Ben wants to include Otherdyssynchrony in airway pressure channel whenever we log a Double Trigger
                    # signal is SpirometryFlow for 'Other'
                    other_signal = 'SpirometryFlow'
                    # code is 1 for 'Other'
                    other_code = '1'

                    # write the lines
                    double_trigger_line = f'{begin},{end},{signal},{code}\n'
                    other_signal_line = f'{begin},{end},{other_signal},{other_code}\n'
                    # first write the double trigger line
                    file.write(double_trigger_line)
                    # then write the other signal line
                    file.write(other_signal_line)

            # import pdb; pdb.set_trace()








