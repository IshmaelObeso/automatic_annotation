import pandas as pd
import shutil
from components.dataset_generation.utilities import utils
from pathlib import Path

class AnnotatedDatasetGenerator:

    """ Class that generates a dataset of raw patient
     files with their corresponding annotations inside an artifact file"""


    def __init__(self, raw_files_directory, spectral_triplets_directory):

        # setup directories
        self.annotated_dataset_directory, self.raw_files_directory, self.spectral_triplets_directory, self.spectral_statics_filepath = self.setup_directories(raw_files_directory, spectral_triplets_directory)

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

    def get_breath_times(self, predictions_df):

        """ gets start and expiration times of breaths with associated predictions """

        # load spectral statics file
        spectral_statics = pd.read_csv(self.spectral_statics_filepath)
        # convert columns to string
        spectral_statics[['patient_id', 'day_id']] = spectral_statics[['patient_id', 'day_id']].astype(str)
        # set index for statics
        spectral_statics = spectral_statics.set_index(['patient_id', 'day_id', 'breath_id'])

        # get rows from statics file that have corresponding uid in preds dataframe
        spectral_statics = spectral_statics.loc[predictions_df.index]

        # grab the start and expiration times of breaths in those rows
        spectral_statics = spectral_statics[['start_time', 'end_time']]

        # convert start time and expiration time columns to seconds
        spectral_statics['start_time'] = pd.to_datetime(spectral_statics['start_time'])
        spectral_statics['end_time'] = pd.to_datetime(spectral_statics['end_time'])

        spectral_statics['start_time'] = spectral_statics['start_time'].dt.second + (spectral_statics['start_time'].dt.minute * 60) + (spectral_statics['start_time'].dt.hour * 3600) + ((spectral_statics['start_time'].dt.day -1) * 86400)
        spectral_statics['end_time'] = spectral_statics['end_time'].dt.second + (spectral_statics['end_time'].dt.minute * 60) + (spectral_statics['end_time'].dt.hour * 3600) + ((spectral_statics['end_time'].dt.day - 1) * 86400)

        # merge with preds df
        predictions_df = pd.concat([predictions_df, spectral_statics], axis=1)

        return predictions_df


    def copy_raw_files(self, predictions_df):

        """ Copies raw files from raw files directory to annotated dataset directory,
         only copies raw files that we have predictions for  """

        # grab all csv files from raw directory
        raw_csv_files = list(self.raw_files_directory.glob('*.csv'))

        # get all patient_days that we have predictions for
        patient_days = predictions_df.index.droplevel(2)
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

    def create_art_files(self, binary_predictions_df, multitarget_predictions_df=None):

        """ creates artifact files for every patient day we have predictions for from the binary predictions, and adds multitarget predictions if there is a multitarget prediction file"""

        # get a list of all patient day files that we have predictions for and copy any raw patient day files we have
        # predictions for into the annotated dataset directory
        patient_day_files_with_predictions = self.copy_raw_files(binary_predictions_df)

        # merge breath times into the predictions dataframe
        predictions_df = self.get_breath_times(binary_predictions_df)

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
            pt_day_preds = predictions_df.loc[pt_day]

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

                    # if we have not included a multitarget preds dataframe
                    if multitarget_predictions_df is None:

                        begin = pt_day_preds.loc[index]['start_time']
                        end = pt_day_preds.loc[index]['end_time']

                        # signal is AirwayPressure for Double Trigger
                        airway_pressure_signal = 'AirwayPressure'

                        # code is 107 for Double Trigger
                        double_trigger_code = '107'

                        # also Ben wants to include Otherdyssynchrony in airway pressure channel whenever we log a Double Trigger
                        # signal is SpirometryFlow for 'Other'
                        flow_signal = 'SpirometryFlow'
                        # code is 1 for 'Other'
                        other_code = '1'

                        # write the lines
                        double_trigger_line = f'{begin},{end},{airway_pressure_signal},{double_trigger_code}\n'
                        other_signal_line = f'{begin},{end},{flow_signal},{other_code}\n'
                        # first write the double trigger line
                        file.write(double_trigger_line)
                        # then write the other signal line
                        file.write(other_signal_line)

                    # if we did include a multitarget preds dataframe
                    else:

                        begin = pt_day_preds.loc[index]['start_time']
                        end = pt_day_preds.loc[index]['end_time']

                        # signal is AirwayPressure for Double Trigger
                        airway_pressure_signal = 'AirwayPressure'

                        # code is 107 for Double Trigger
                        double_trigger_code = '107'

                        # signal is SpirometryFlow for reverse trigger, inadequate support, Other
                        flow_signal = 'SpirometryFlow'

                        # grab the patient day from the multitarget dataframe
                        multitarget_patient_day_preds = multitarget_predictions_df.loc[pt_day]

                        # get the reverse trigger prediction for the breath with the Double Trigger
                        reverse_trigger_prediction = multitarget_patient_day_preds.loc[index]['Double Trigger Reverse Trigger_pred']
                        reverse_trigger_prediction_thresholded = multitarget_patient_day_preds.loc[index]['Double Trigger Reverse Trigger_threshold']

                        # get the inadequate support prediction for the breath with the Double Trigger
                        premature_termination_prediction = multitarget_patient_day_preds.loc[index]['Double Trigger Premature Termination_pred']
                        premature_termination_prediction_thresholded = multitarget_patient_day_preds.loc[index]['Double Trigger Premature Termination_threshold']

                        # get the inadequate support prediction for the breath with the Double Trigger
                        flow_undershoot_prediction = multitarget_patient_day_preds.loc[index]['Double Trigger Flow Undershoot_pred']
                        flow_undershoot_prediction_thresholded = multitarget_patient_day_preds.loc[index]['Double Trigger Flow Undershoot_threshold']

                        # make dict with information about each column
                        multitarget_dict = {reverse_trigger_prediction_thresholded: {'prediction': reverse_trigger_prediction, 'code': '114'},
                        premature_termination_prediction_thresholded: {'prediction': premature_termination_prediction, 'code': '111'},
                        flow_undershoot_prediction_thresholded: {'prediction': flow_undershoot_prediction, 'code': '115'},
                        }

                        # find out which classes passed their thresholds for detection
                        prediction_thresholded_list = [reverse_trigger_prediction_thresholded, premature_termination_prediction_thresholded, flow_undershoot_prediction_thresholded]
                        detected = [item for i, item in enumerate(prediction_thresholded_list) if prediction_thresholded_list[i] == 1]

                        # if no class passed threshold, use 'Other' code
                        if len(detected) == 0:
                            other_code = '1'

                        # if just one class passed threshold, use that class code
                        if len(detected) == 1:
                            other_code = multitarget_dict[detected[0]]['code']

                        # if multiple classes passed threshold, find the class with the largest raw prediction and use that
                        if len(detected) > 1:

                            # variables to keep track of largest prediction and its asynchrony code
                            largest_pred = 0
                            other_code = '1'

                            # loop through the classes that passed detection threshold
                            for i, label in enumerate(detected):

                                raw_pred = multitarget_dict[detected[i]]['prediction']
                                class_code = multitarget_dict[detected[i]]['code']

                                if raw_pred > largest_pred:

                                    largest_pred = raw_pred
                                    other_code = class_code



                        # # find out which one is larger, if they are both the same, the annotation should be 'Other' because we don't know which of the two it is
                        # # if prediction is reverse trigger
                        # if reverse_trigger_prediction_thresholded > inadequate_support_prediction_thresholded:
                        #     # reverse trigger code is
                        #     other_code = '114'
                        #
                        # elif inadequate_support_prediction_thresholded > reverse_trigger_prediction_thresholded:

                        #     # inadequate support code doesn't exist :(
                        #     other_code = '113'
                        #
                        # # if they are both 1, use the larger non-thresholded prediction
                        # elif (reverse_trigger_prediction_thresholded == 1) and (inadequate_support_prediction_thresholded == 1):
                        #
                        #     # if reverse trigger prediction is larger, use that
                        #     if reverse_trigger_prediction > inadequate_support_prediction:
                        #         # reverse trigger code is
                        #         other_code = '114'
                        #     # if inadequate support prediction is larger, use that
                        #     else:
                        #         # inadequate support code doesn't exist :(
                        #         other_code = '113'
                        #
                        # # if they are both 0
                        # else:
                        #
                        #     # code is 1 for 'Other'
                        #     other_code = '1'

                        # write the lines
                        double_trigger_line = f'{begin},{end},{airway_pressure_signal},{double_trigger_code}\n'
                        other_signal_line = f'{begin},{end},{flow_signal},{other_code}\n'
                        # first write the double trigger line
                        file.write(double_trigger_line)
                        # then write the other signal line
                        file.write(other_signal_line)









