import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from components.annotation_generation import prediction_data_pipe


class BinaryPredictionGenerator:
    ''' Generates Prediction dataframe for binary model predictions '''

    def __init__(self, spectral_triplet_directory):

        # setup directories
        self.spectral_triplet_directory, self.predictions_export_directory, self.predictions_output_path_csv, self.predictions_output_path_hdf = self.setup_directories(spectral_triplet_directory)

        #instantiate dataset
        self.data_class = prediction_data_pipe.Data_Pipe(spectral_triplet_directory, output_cols=['Double Trigger'])

    def setup_directories(self, spectral_triplet_directory):

        # define paths
        spectral_triplet_directory = Path(spectral_triplet_directory)

        # put preds df in the parent directory of the spectral triplets directory
        spectral_triplets_parent_directory = spectral_triplet_directory.parents[0]
        predictions_export_directory = Path(spectral_triplets_parent_directory, 'predictions')

        predictions_export_directory.mkdir(parents=True, exist_ok=True)

        # setup predictions dataframe filepath
        predictions_output_path_csv = Path(predictions_export_directory, 'binary_predictions.csv')
        predictions_output_path_hdf = Path(predictions_export_directory, 'binary_predictions.hdf')

        return spectral_triplet_directory.resolve(),\
               predictions_export_directory.resolve(),\
               predictions_output_path_csv.resolve(),\
               predictions_output_path_hdf.resolve()

    def threshold_predictions(self, predictions_df, threshold):

        """ Use a threshold to get binarized predictions for every breath"""

        threshold_df = predictions_df
        threshold_df['prediction'] = (threshold_df['prediction'] >= threshold).astype(int)

        return threshold_df

    def save_predictions(self, predictions_df):
        """ saves the raw predictions dataframe into csv and hdf """

        # Write the statics file
        predictions_df.to_hdf(self.predictions_output_path_hdf, key='statics')
        predictions_df.to_csv(self.predictions_output_path_csv)

    def get_predictions(self, model, threshold):
        """ get predictions for every spectral triplet in dataset, predictions will be output as hdf file"""

        # get model attributes
        input_name, input_shape, output_name = model.get_model_attributes()
        # set batch_size to 1
        input_shape[0] = 1


        # test getitem for every uid
        preds_list = []
        truths_list = []
        patient_id_list = []
        day_id_list = []
        breath_id_list = []

        # get all uids in dataset
        all_uids = self.data_class.get_uids()

        # for every spectrogram in dataset
        for uid in tqdm.tqdm(all_uids, desc='Generating Binary Predictions'):

            # get spectrograms for every uid
            xs, ys, ts, uid = self.data_class.__getitem__(uid)

            # reshape
            xs = np.reshape(xs, input_shape)

            # get prediction
            pred = model.session.run([output_name], {input_name: xs.astype(np.float32)})
            # get prediction as a singular array
            pred = pred[0][0]

            # get the max of truth ie. was there a dyssynchrony in this triplet or not
            ys = np.nan_to_num(ys).max(axis=0)

            # append to list to make into df later
            preds_list.append(pred)
            truths_list.append(ys)
            patient_id_list.append(uid[0])
            day_id_list.append(uid[1])
            breath_id_list.append(uid[2])

        # stack preds and truths into array
        preds = np.stack(preds_list)
        truths = np.stack(truths_list)

        # make predictions dataframe
        df = pd.DataFrame({'patient_id': patient_id_list,
                           'day_id': day_id_list,
                           'breath_id': breath_id_list})

        preds_df = pd.DataFrame({'prediction': preds[:, 0], 'truth': truths[:, 0]})

        preds_df = pd.concat([df, preds_df], axis=1)

        preds_df = preds_df.set_index(['patient_id', 'day_id', 'breath_id'])

        # threshold predictions df
        self.threshold_predictions(preds_df, threshold)

        # save the predictions dataframe
        self.save_predictions(preds_df)

        return preds_df

class MultitargetPredictionGenerator:
    ''' Generates Prediction dataframe for multitarget model predictions'''
    def __init__(self, spectral_triplet_directory):
        # setup directories
        self.spectral_triplet_directory, self.predictions_export_directory, self.predictions_output_path_csv, self.predictions_output_path_hdf = self.setup_directories(
            spectral_triplet_directory)

        # instantiate dataset
        self.data_class = prediction_data_pipe.Data_Pipe(spectral_triplet_directory, output_cols=['Double Trigger Reverse Trigger', 'Double Trigger Inadequate Support'])

    def setup_directories(self, spectral_triplet_directory):
        # define paths
        spectral_triplet_directory = Path(spectral_triplet_directory)

        # put preds df in the parent directory of the spectral triplets directory
        spectral_triplets_parent_directory = spectral_triplet_directory.parents[0]
        predictions_export_directory = Path(spectral_triplets_parent_directory, 'predictions')

        predictions_export_directory.mkdir(parents=True, exist_ok=True)

        # setup predictions dataframe filepath
        predictions_output_path_csv = Path(predictions_export_directory, 'multitarget_predictions.csv')
        predictions_output_path_hdf = Path(predictions_export_directory, 'multitarget_predictions.hdf')

        return spectral_triplet_directory.resolve(), \
               predictions_export_directory.resolve(), \
               predictions_output_path_csv.resolve(), \
               predictions_output_path_hdf.resolve()

    def threshold_predictions(self, predictions_df, thresholds):
        """ Use a threshold to get binarized predictions for every breath """

        # threshold Reverse Trigger Predictions
        predictions_df['Double Trigger Reverse Trigger_pred'] = (predictions_df['Double Trigger Reverse Trigger_pred'] >= thresholds[0]).astype(int)

        # threshold Inadequate Support Predictions
        predictions_df['Double Trigger Inadequate Support_pred'] = (predictions_df['Double Trigger Inadequate Support_pred'] >= thresholds[1]).astype(int)

        return predictions_df

    def save_predictions(self, predictions_df):
        """ saves the raw predictions dataframe into csv and hdf """

        # Write the statics file
        predictions_df.to_hdf(self.predictions_output_path_hdf, key='statics')
        predictions_df.to_csv(self.predictions_output_path_csv)

    def get_predictions(self, model, threshold):
        """ get predictions for every spectral triplet in dataset, predictions will be output as hdf file"""

        # get model attributes
        input_name, input_shape, output_name = model.get_model_attributes()
        # set batch_size to 1
        input_shape[0] = 1

        # test getitem for every uid
        preds_list = []
        truths_list = []
        patient_id_list = []
        day_id_list = []
        breath_id_list = []

        # get all uids in dataset
        all_uids = self.data_class.get_uids()

        # for every spectrogram in dataset
        for uid in tqdm.tqdm(all_uids, desc='Generating Multitarget Predictions'):

            # get spectrograms for every uid
            xs, ys, ts, uid = self.data_class.__getitem__(uid)

            # reshape
            xs = np.reshape(xs, input_shape)

            # get prediction
            pred = model.session.run([output_name], {input_name: xs.astype(np.float32)})
            # get prediction as a singular array
            pred = pred[0][0]

            # get the max of truth ie. was there a dyssynchrony in this triplet or not
            ys = np.nan_to_num(ys).max(axis=0)

            # append to list to make into df later
            preds_list.append(pred)
            truths_list.append(ys)
            patient_id_list.append(uid[0])
            day_id_list.append(uid[1])
            breath_id_list.append(uid[2])

        # stack preds and truths into array
        preds = np.stack(preds_list)
        truths = np.stack(truths_list)

        # make predictions dataframe
        df = pd.DataFrame({'patient_id': patient_id_list,
                           'day_id': day_id_list,
                           'breath_id': breath_id_list})

        preds_df = pd.DataFrame({'Double Trigger Reverse Trigger_truth': truths[:, 0], 'Double Trigger Reverse Trigger_pred': preds[:, 0],
                                 'Double Trigger Inadequate Support_truth': truths[:, 1], 'Double Trigger Inadequate Support_pred': preds[:, 1]})

        preds_df = pd.concat([df, preds_df], axis=1)

        preds_df = preds_df.set_index(['patient_id', 'day_id', 'breath_id'])

        # threshold predictions df
        self.threshold_predictions(preds_df, threshold)

        # save the predictions dataframe
        self.save_predictions(preds_df)

        return preds_df

