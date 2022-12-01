import pandas as pd
import numpy as np
from pathlib import Path
from components.annotation_generation import prediction_data_pipe
from pqdm.processes import pqdm


class PredictionAggregator:

    '''
    Warps around a prediction generator, allows for multiple model predictions to be output and concantenated into a predictions dataframe
    '''


    def __init__(self, spectral_triplets_directory: object) -> object:

        ''' initiate by setting up directory paths 

        Args:
            spectral_triplets_directory: 
        '''

        # setup directories
        self.spectral_triplets_directory,\
        self.predictions_output_path_csv,\
        self.predictions_output_path_hdf = self.setup_directories(spectral_triplets_directory=spectral_triplets_directory)

    def setup_directories(self, spectral_triplets_directory: object) -> object:
        """

        Args:
            spectral_triplets_directory: 

        Returns:

        """
        spectral_triplets_directory = Path(spectral_triplets_directory)

        # put preds df in the parent directory of the spectral triplets directory
        spectral_triplets_parent_directory = spectral_triplets_directory.parents[0]
        predictions_export_directory = Path(spectral_triplets_parent_directory, 'predictions')

        predictions_export_directory.mkdir(parents=True, exist_ok=True)

        # setup predictions dataframe filepath
        predictions_output_path_csv = Path(predictions_export_directory, 'predictions.csv')
        predictions_output_path_hdf = Path(predictions_export_directory, 'predictions.hdf')

        return spectral_triplets_directory.resolve(),\
               predictions_output_path_csv.resolve(),\
               predictions_output_path_hdf.resolve()


    def save_predictions(self, predictions_df: object) -> object:
        """ saves the raw predictions dataframe into csv and hdf 

        Args:
            predictions_df: 
        """

        # Write the statics file
        predictions_df.to_hdf(self.predictions_output_path_hdf, key='statics')
        predictions_df.to_csv(self.predictions_output_path_csv)

    def generate_all_predictions(self, models_dict: object) -> object:

        ''' get dictionary with information about models, use this to generate predictions 

        Args:
            models_dict: 
        '''

        # make list of predictions dataframes
        prediction_dfs = []

        # get predictions for every model and aggregate predictions into one dataframe
        for model_name, parameters in models_dict.items():

            if parameters['use']:

                # instantiate prediction generator
                prediction_generator = PredictionGenerator(
                    spectral_triplet_directory=self.spectral_triplets_directory,
                    output_cols=parameters['output_columns'])

                # get predictions file for model
                predictions_df = prediction_generator.get_predictions(model_name, parameters)

                # append predictions df to prediction dfs list
                prediction_dfs.append(predictions_df)

        # make large df from list of smaller dfs
        predictions_df = pd.concat(prediction_dfs, axis=1)

        # save predictiojns
        self.save_predictions(predictions_df)

        return predictions_df
class PredictionGenerator:
    ''' Generates Prediction dataframe for binary model predictions '''

    def __init__(self, spectral_triplet_directory: object, output_cols: object = ['Double Trigger']) -> object:
        """

        Args:
            spectral_triplet_directory: 
            output_cols: 
        """
        #instantiate dataset
        self.data_class = prediction_data_pipe.Data_Pipe(spectral_triplet_directory, output_cols=output_cols)

        self.output_cols = output_cols

    def threshold_predictions(self, predictions_df: object, threshold_dict: object) -> object:

        """ Use a threshold to get binarized predictions for every breath

        Args:
            predictions_df: 
            threshold_dict: 
        """

        # for every dyssynchrony threshold predictions and make new column
        for output_col in self.output_cols:

            # get the column names
            pred_column_name = f'{output_col}_preds'
            threshold_column_name = f'{output_col}_threshold'

            # get the threshold
            threshold = threshold_dict[output_col]

            # make threshold column
            predictions_df[threshold_column_name] = (predictions_df[pred_column_name] >= threshold).astype(int)

        return predictions_df


    def generate_predictions(self, uid: object) -> object:
        """

        Args:
            uid: 

        Returns:

        """
        # get spectrograms for every uid
        xs, ys, ts, uid = self.data_class.__getitem__(uid)

        # reshape
        xs = np.reshape(xs, self.input_shape)

        # get prediction
        pred = self.model.session.run([self.output_name], {self.input_name: xs.astype(np.float32)})
        # get prediction as a singular array
        pred = pred[0][0]

        # get the max of truth ie. was there a dyssynchrony in this triplet or not
        ys = np.nan_to_num(ys).max(axis=0)

        # save truth and preds as dataframes
        uid_y0s_df = pd.DataFrame([pred], columns=[f'{col}_preds' for col in self.output_cols])
        uid_ys_df = pd.DataFrame([ys], columns=[f'{col}_truth' for col in self.output_cols])

        # concat truth and preds
        uid_df = pd.concat([uid_ys_df, uid_y0s_df], axis=1)

        # add index columns
        uid_df['patient_id'] = uid[0]
        uid_df['day_id'] = uid[1]
        uid_df['breath_id'] = uid[2]

        return uid_df

    def get_predictions(self, model_name: object, parameters: object) -> object:
        """ get predictions for every spectral triplet in dataset, predictions will be output as hdf file

        Args:
            model_name: 
            parameters: 
        """

        # get model parameters
        threshold_dict = parameters['threshold']
        self.model = parameters['model_object']

        # get model attributes
        self.input_name, self.input_shape, self.output_name = self.model.get_model_attributes()
        # set batch_size to 1
        self.input_shape[0] = 1

        # get all uids in dataset
        all_uids = self.data_class.get_uids()

        # multiprocessing requires a list to loop over, a function object, and number of workers
        # MULTIPROCESSING ONNX MAKES IT SLOWER, SET N_JOBS TO 1 :(
        # will keep the pqdm call though because it looks nicer than a loop with tqdm
        # results will be a list of uid_dfs that we will concantenate together

        results = pqdm(all_uids, self.generate_predictions, n_jobs=1, desc=f'Generating {model_name} Predictions')

        # concat dfs
        preds_df = pd.concat(results)

        # set index
        preds_df = preds_df.set_index(['patient_id', 'day_id', 'breath_id'])

        # threshold predictions df
        self.threshold_predictions(preds_df, threshold_dict)

        return preds_df

