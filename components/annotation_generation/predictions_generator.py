import pandas as pd
import numpy as np
from pathlib import Path
from components.annotation_generation import dyssynchrony_dataloader
from pqdm.processes import pqdm
from typing import Union


class PredictionAggregator:
    """
    Warps around a prediction generator, allows for multiple model predictions to be output and concantenated into a predictions dataframe

    Attributes:
        _spectral_triplets_directory (Path): Path to the directory where spectral triplets are stored
        _predictions_output_path_csv (Path): Path to where predictions csv will be saved
        _predictions_output_path_hdf (Path): Path to where predictions hdf will be saved
    """
    def __init__(self, spectral_triplets_directory: Union[str, Path]) -> None:
        """
        Sets initial class attributes

        Args:
            spectral_triplets_directory (Union[str, Path]): Path to the directory where spectral triplets are stored

        Returns:
            None:
        """

        # setup attributes
        self._spectral_triplets_directory = spectral_triplets_directory
        self._predictions_output_path_csv = None
        self._predictions_output_path_hdf = None

    @staticmethod
    def _setup_directories(spectral_triplets_directory: Union[str, Path]) -> tuple[Path, Path, Path]:
        """
        Sets up spectral_triplets_directory, predictions_output_path_csv, and predictions_output_path_hdf
        to use spectral triplets for predictions and save the predictions to csv and hdf

        Args:
            spectral_triplets_directory (Union[str, Path]):

        Returns:
            spectral_triplets_directory (Path): Path to the directory where spectral triplets are stored
            predictions_output_path_csv (Path): Path to where predictions csv will be saved
            predictions_output_path_hdf (Path): Path to where predictions hdf will be saved
        """
        spectral_triplets_directory = Path(spectral_triplets_directory)

        # put preds df in the parent directory of the spectral triplets directory
        spectral_triplets_parent_directory = spectral_triplets_directory.parents[0]
        predictions_export_directory = Path(spectral_triplets_parent_directory, 'predictions')

        predictions_export_directory.mkdir(parents=True, exist_ok=True)

        # setup predictions dataframe filepath
        predictions_output_path_csv = Path(predictions_export_directory, 'predictions.csv')
        predictions_output_path_hdf = Path(predictions_export_directory, 'predictions.hdf')

        return spectral_triplets_directory.resolve(), predictions_output_path_csv.resolve(), predictions_output_path_hdf.resolve()


    def _save_predictions(self, predictions_df: pd.DataFrame) -> None:
        """
        Saves the raw predictions dataframe into csv and hdf

        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions from model

        Returns:
            None:

        """

        # Write the statics file
        predictions_df.to_hdf(self._predictions_output_path_hdf, key='statics')
        predictions_df.to_csv(self._predictions_output_path_csv)

    def generate_all_predictions(self, models_dict: dict) -> pd.DataFrame:
        """
        Get dictionary with information about models, use this to generate predictions

        Args:
            models_dict (dict): Dictionary with information about models is stored

        Returns:
            predictions_df (pd.DataFrame): DataFrame with predictions from model

        """

        # setup directories
        self._spectral_triplets_directory,\
            self._predictions_output_path_csv,\
            self._predictions_output_path_hdf = PredictionAggregator._setup_directories(spectral_triplets_directory=self._spectral_triplets_directory)

        # make list of predictions dataframes
        prediction_dfs = []

        # get predictions for every model and aggregate predictions into one dataframe
        for model_name, parameters in models_dict.items():

            if parameters['use']:

                # instantiate prediction generator
                prediction_generator = PredictionGenerator(
                    spectral_triplet_directory=self._spectral_triplets_directory,
                    output_cols=parameters['output_columns'])

                # get predictions file for model
                predictions_df = prediction_generator.get_predictions(model_name, parameters)

                # append predictions df to prediction dfs list
                prediction_dfs.append(predictions_df)

        # make large df from list of smaller dfs
        predictions_df = pd.concat(prediction_dfs, axis=1)

        # save predictions
        self._save_predictions(predictions_df)

        return predictions_df


class PredictionGenerator:
    """
    Generates Prediction dataframe for binary model predictions

    Attributes:
        _data_class (dyssynchrony_dataloader.DyssynchronyDataLoader):
        _output_cols (list): List of the output columns of model
        _output_name (str): Name of model outputs
        _input_shape (np.ndarray): Shape of model inputs
        _input_name (str): Name of model inputs
        _model (annotation_model.AnnotationModel): Initialized onnx model that takes inputs and returns predictions
    """
    def __init__(self, spectral_triplet_directory: Path, output_cols: list[str] = None) -> None:
        """
        Sets initial class attributes

        Args:
            spectral_triplet_directory (Path): Path to the directory where spectral triplets are stored
            output_cols (list[str]): List of the output columns of model

        Returns:
            None:
        """

        # setup attributes
        self._data_class = dyssynchrony_dataloader.DyssynchronyDataLoader(spectral_triplet_directory, output_cols=output_cols)
        self._output_cols = output_cols
        self._output_name = None
        self._input_shape = None
        self._input_name = None
        self._model = None

    def _threshold_predictions(self, predictions_df: pd.DataFrame, threshold_dict: dict) -> pd.DataFrame:
        """
        Use a threshold to get binarized predictions for every breath

        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions from model
            threshold_dict (dict): Dictionary with information about threshold for this model

        Returns:
            predictions_df (pd.DataFrame): DataFrame with predictions from model and thresholded predictions
        """

        # for every dyssynchrony threshold predictions and make new column
        for output_col in self._output_cols:

            # get the column names
            pred_column_name = f'{output_col}_preds'
            threshold_column_name = f'{output_col}_threshold'

            # get the threshold
            threshold = threshold_dict[output_col]

            # make threshold column
            predictions_df[threshold_column_name] = (predictions_df[pred_column_name] >= threshold).astype(int)

        return predictions_df

    def _generate_predictions(self, uid: tuple[str, str, int]) -> pd.DataFrame:
        """
        Generates predictions from onnx model

        Args:
            uid (tuple[str, str, int]): Uid to get predictions of

        Returns:
            uid_df (pd.DataFrame): DataFrame with predictions from uid

        """
        # get spectrograms for every uid
        xs, ys, ts, uid = self._data_class.__getitem__(uid)

        # reshape
        xs = np.reshape(xs, self._input_shape)

        # get prediction
        pred = self._model.run([self._output_name], {self._input_name: xs.astype(np.float32)})
        # get prediction as a singular array
        pred = pred[0][0]

        # get the max of truth i.e. was there a dyssynchrony in this triplet or not
        ys = np.nan_to_num(ys).max(axis=0, initial=None)

        # save truth and preds as dataframes
        uid_y0s_df = pd.DataFrame([pred], columns=[f'{col}_preds' for col in self._output_cols])
        uid_ys_df = pd.DataFrame([ys], columns=[f'{col}_truth' for col in self._output_cols])

        # concat truth and preds
        uid_df = pd.concat([uid_ys_df, uid_y0s_df], axis=1)

        # add index columns
        uid_df['patient_id'] = uid[0]
        uid_df['day_id'] = uid[1]
        uid_df['breath_id'] = uid[2]

        return uid_df

    def get_predictions(self, model_name: str, parameters: dict) -> pd.DataFrame:
        """
        Get predictions for every spectral triplet in dataset, predictions will be output as hdf file

        Args:
            model_name (str): The name of the model to get predictions from (used only for print statements)
            parameters (dict): Parameters of the model, e.g. thresholds and the model object

        Returns:
            preds_df (pd.DataFrame): DataFrame with predictions from model
        """

        # get model parameters
        threshold_dict = parameters['threshold']
        self._model = parameters['model_object']

        # get model attributes
        self._input_name = self._model.input_name
        self._input_shape = self._model.input_shape
        self._output_name = self._model.output_name

        # set batch_size to 1
        self._input_shape[0] = 1

        # get all uids in dataset
        all_uids = self._data_class.all_uids

        # multiprocessing requires a list to loop over, a function object, and number of workers
        # MULTIPROCESSING ONNX MAKES IT SLOWER, SET N_JOBS TO 1 :(
        # will keep the pqdm call though because it looks nicer than a loop with tqdm
        # results will be a list of uid_dfs that we will concantenate together

        results = pqdm(all_uids, self._generate_predictions, n_jobs=1, desc=f'Generating {model_name} Predictions')

        # concat dfs
        preds_df = pd.concat(results)

        # set index
        preds_df = preds_df.set_index(['patient_id', 'day_id', 'breath_id'])

        # threshold predictions df
        self._threshold_predictions(preds_df, threshold_dict)

        return preds_df

