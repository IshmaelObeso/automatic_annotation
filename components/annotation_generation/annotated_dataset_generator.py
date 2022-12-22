import pandas as pd
import shutil
import itertools
from components.dataset_generation.utilities import utils
from pathlib import Path
from typing import Union


class AnnotatedDatasetGenerator:
    """
    Generates a dataset of raw patient files with their corresponding annotations inside an artifact file

     Attributes:



    """

    def __init__(self, raw_files_directory: Union[str, Path], spectral_triplets_directory: Union[str, Path]) -> None:
        """
        Sets initial class attributes

        Args:
            raw_files_directory (Union[str, Path]): Path to directory where raw patient-day files are stored
            spectral_triplets_directory (Union[str, Path]): Path to directory where spectral triplets are stored
        """

        # setup directories
        self._raw_files_directory = raw_files_directory
        self._spectral_triplets_directory = spectral_triplets_directory
        self._annotated_dataset_directory = None
        self._spectral_statics_filepath = None

    @staticmethod
    def _setup_directories(raw_files_directory: Union[str, Path], spectral_triplets_directory: Union[str, Path]) -> tuple[Path, Path, Path]:
        """
        Sets up annotated_dataset_directory, raw_files_directory, and spectral_statics_filepath

        Args:
            raw_files_directory (Union[str, Path]): Path to directory where raw patient-day files are stored
            spectral_triplets_directory (Union[str, Path]): Path to directory where spectral triplets are stored

        Returns:
            annotated_dataset_directory (Path): Path to directory where we will save our .art files
            raw_files_directory (Path): Path to directory where raw patient-day files are stored
            spectral_statics_filepath (Path): Path to spectral statics file
        """

        # define paths
        raw_files_directory = Path(raw_files_directory.replace('"', '').replace("'", ''))
        spectral_triplets_directory = Path(spectral_triplets_directory)

        # make annotated dataset directory
        spectral_triplets_parent_directory = spectral_triplets_directory.parents[0]
        annotated_dataset_directory = Path(spectral_triplets_parent_directory, 'annotated_dataset')

        annotated_dataset_directory.mkdir(parents=True, exist_ok=True)

        # get the spectral triplet statics file path
        spectral_statics_filepath = Path(spectral_triplets_directory, 'spectral_statics.csv')

        return annotated_dataset_directory.resolve(), \
               raw_files_directory.resolve(), \
               spectral_statics_filepath.resolve()

    @staticmethod
    def _get_breath_times(predictions_df: pd.DataFrame, spectral_statics_filepath: Path) -> pd.DataFrame:
        """
        Gets start and expiration times of breaths with associated predictions

        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions from model
            spectral_statics_filepath (Path): Path to spectral statics file

        Returns:
            predictions_df (pd.DataFrame): DataFrame with predictions from model with breath timings added
        """

        # load spectral statics file
        spectral_statics = pd.read_csv(spectral_statics_filepath)
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

        spectral_statics['start_time'] = spectral_statics['start_time'].dt.second + (
                    spectral_statics['start_time'].dt.minute * 60) + (spectral_statics['start_time'].dt.hour * 3600) + (
                                                     (spectral_statics['start_time'].dt.day - 1) * 86400)
        spectral_statics['end_time'] = spectral_statics['end_time'].dt.second + (
                    spectral_statics['end_time'].dt.minute * 60) + (spectral_statics['end_time'].dt.hour * 3600) + (
                                                   (spectral_statics['end_time'].dt.day - 1) * 86400)

        # merge with preds df
        predictions_df = pd.concat([predictions_df, spectral_statics], axis=1)

        return predictions_df

    @staticmethod
    def _copy_raw_files(predictions_df: pd.DataFrame, raw_files_directory: Path, annotated_dataset_directory: Path) -> list[Union[str, Path]]:
        """
        Copies raw files from raw files directory to annotated dataset directory,
        only copies raw files that we have predictions for

        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions from model with breath timings added
            raw_files_directory (Path): Path to directory where raw patient-day files are stored
            annotated_dataset_directory (Path): Path to directory where we will save our .art files

        Returns:
            patient_day_files_with_predictions (list[Union[str, Path]]): List of patient-day files we have predictions for

        """

        # grab all csv files from raw directory
        raw_csv_files = list(raw_files_directory.glob('*.csv'))

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
            new_patient_day_filepath = Path(annotated_dataset_directory, patient_day_file.name)

            # copy file
            shutil.copy(patient_day_file, new_patient_day_filepath)

        return patient_day_files_with_predictions

    def create_art_files(self, predictions_df: pd.DataFrame, models_dict: dict) -> None:
        """
        Creates artifact files for every patient day we have predictions for from the binary predictions,
        and adds multitarget predictions if there is a multitarget prediction file

        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions from model
            models_dict (dict): Dictionary with information about models is stored
        """

        # setup directories
        annotated_dataset_directory, \
            raw_files_directory, \
            spectral_statics_filepath = AnnotatedDatasetGenerator._setup_directories(self._raw_files_directory,
                                                                                     self._spectral_triplets_directory)

        # get a list of all patient day files that we have predictions for and copy any raw patient day files we have
        # predictions for into the annotated dataset directory
        patient_day_files_with_predictions = AnnotatedDatasetGenerator._copy_raw_files(predictions_df, raw_files_directory,
                                                                  annotated_dataset_directory)

        # get a list of all output column names, if model is being used
        output_columns = []
        for name, parameters in models_dict.items():
            if parameters['use']:
                output_columns.append(parameters['output_columns'])
            else:
                pass
        # flatten list
        output_columns = list(itertools.chain(*output_columns))

        # merge breath times into the predictions dataframe
        predictions_df = AnnotatedDatasetGenerator._get_breath_times(predictions_df, spectral_statics_filepath)

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

            # get filepath to the new dataset directory with patient_day_filename
            patient_day_filepath = Path(annotated_dataset_directory, patient_day_file.name)

            # build an artifacts file for that patient day
            artifacts_filepath = patient_day_filepath.with_suffix('.art')

            # grab all rows from that day that have detections in columns
            threshold_columns = [f'{column}_threshold' for column in output_columns]
            # find rows where there are detections for every dyssynchrony we are using
            pt_day_preds_dyssynchrony_index = []
            # for every threshold column
            for threshold_column in threshold_columns:
                # grab indexes of rows where there are detections in the threshold column,
                # turn index to list, and add it to the existing list of rows where there are detections
                pt_day_preds_dyssynchrony_index.extend(pt_day_preds[pt_day_preds[threshold_column] == 1].index.tolist())

            # turn list of indexes to one index, drop duplicates, and sort
            pt_day_preds_dyssynchrony_index = pd.Index(pt_day_preds_dyssynchrony_index).drop_duplicates().sort_values()

            with open(artifacts_filepath, 'w') as file:
                # write the column names
                file.write('begin,end,signal,code\n')

                # for every dyssynchrony for that patient day
                for index in pt_day_preds_dyssynchrony_index:

                    begin = pt_day_preds.loc[index]['start_time']
                    end = pt_day_preds.loc[index]['end_time']

                    # grab the columns and row that correspond to this breath
                    detected_classes = pt_day_preds.loc[index, threshold_columns]
                    # find which of the classes we are using were detected if any
                    detected_classes = detected_classes.loc[(detected_classes == 1)].index.to_list()

                    ## Now we write the logic to annotate the breaths based on the detected classes for that breath

                    ### TODO: THIS LOGIC ONLY WORKS IF ALL MODELS ARE BINARY, IF WE DECIDE TO ADD IN A MULTI-TARGET MODEL LATER, THIS NEEDS TO CHANGE

                    ## For double trigger
                    # check to see if double trigger is detected, if it is, annotate it, then remove double trigger from detected classes list
                    for detected_class in detected_classes:

                        detected_class_name = detected_classes[0].split('_')[0]

                        if detected_class_name == 'Double Trigger':

                            # if double trigger is the only detected thing, add ??? label to the breath in the spirometry flow channel
                            if len(detected_classes) == 1:

                                dyssynchrony_code = models_dict[detected_class_name]['dyssynch_code']
                                channel_code = models_dict[detected_class_name]['channel']
                                other_code = '1'
                                other_channel = 'SpirometryFlow'
                                # write the lines
                                line = f'{begin},{end},{channel_code},{dyssynchrony_code}\n'
                                other_line = f'{begin},{end},{other_channel},{other_code}\n'
                                # first write the double trigger line
                                file.write(line)
                                # then write 'other' line
                                file.write(other_line)
                                # then remove double trigger from list of detected classes
                                detected_classes.remove(detected_class)

                            # else if other classes are detected, let them get annotated later in the script
                            else:
                                dyssynchrony_code = models_dict[detected_class_name]['dyssynch_code']
                                channel_code = models_dict[detected_class_name]['channel']
                                # write the lines
                                line = f'{begin},{end},{channel_code},{dyssynchrony_code}\n'
                                # first write the double trigger line
                                file.write(line)
                                # then remove double trigger from list of detected classes
                                detected_classes.remove(detected_class)

                    # for every other class
                    # if just one class passed threshold, use that class code
                    if len(detected_classes) == 1:
                        detected_class_name = detected_classes[0].split('_')[0]
                        dyssynchrony_code = models_dict[detected_class_name]['dyssynch_code']
                        channel_code = models_dict[detected_class_name]['channel']

                        # write the lines
                        line = f'{begin},{end},{channel_code},{dyssynchrony_code}\n'
                        # first write the double trigger line
                        file.write(line)

                    # if multiple classes passed threshold, find the class with the largest raw prediction and use that
                    if len(detected_classes) > 1:

                        # variables to keep track of the largest prediction and its asynchrony code
                        largest_pred = 0
                        dyssynchrony_code = '0'
                        channel_code = '0'

                        # loop through the classes that passed detection threshold
                        for detected_class in detected_classes:

                            detected_class_name = detected_class.split('_')[0]
                            raw_pred = pt_day_preds.loc[index, detected_class]
                            current_dyssynchrony_code = models_dict[detected_class_name]['dyssynch_code']
                            current_channel_code = models_dict[detected_class_name]['channel']

                            if raw_pred > largest_pred:
                                largest_pred = raw_pred
                                dyssynchrony_code = current_dyssynchrony_code
                                channel_code = current_channel_code

                        # write the lines
                        line = f'{begin},{end},{channel_code},{dyssynchrony_code}\n'
                        # first write the double trigger line
                        file.write(line)
