import onnxruntime as rt
import numpy as np

from pathlib import Path
from typing import Union, TypedDict


class AnnotationModel:
    """
    Creates an onnx model to annotate spectral triplets. onnx is framework-agnostic so you can turn any type of model
    into an onnx model and use it here (Pytorch, Tensorflow, Jax, etc.)

    Attributes
        _model_path (Union[str, Path]): path to the saved model we want to use
        _session (rt.InferenceSession): initialized inference session object


    """

    def __init__(self, model_path: Union[str, Path]) -> None:
        """
        Sets initial class attributes

        Args:
            model_path (Union[str, Path]): path to the saved model we want to use
        Returns:
            None
        """

        # instantiate model
        self._model_path = str(Path(model_path))

        # load path as string, because onnx does not accept pathlib paths
        self._session = AnnotationModel._init_session(self._model_path)

    def run(self, *args: dict) -> list[np.ndarray]:
        """
        Runs an inference session on the instantiated model object. In other words give it inputs,
        it will return predictions

        Args:
            *args: arguments, usually input and output names, and input data

        Returns:
            np.ndarray: Returns numpy array of predictions on input data
        """

        # return predictions
        return self._session.run(*args)

    @staticmethod
    def _init_session(model_path: Union[str, Path]) -> rt.InferenceSession:
        """
        Initializes an inference session in the model object, allows us to put in input data and get predictions

        Args:
            model_path (Path): Path to saved .onnx model

        Returns:
            sess (rt.InferenceSession): Returns initialized inference session object
        """

        # set the providers list to only CPU, GPU causes problems with .exe file
        providers_list = ['CPUExecutionProvider']
        # initialize inference session
        sess = rt.InferenceSession(model_path, providers=providers_list)
        return sess

    def __setstate__(self, values: dict) -> None:
        """
        allows setting state of object, important for making onnx model picklable

        Args:
            values (dict): dictionary of values we want to set (we only have model_path do anything right now)
        Returns:
            None
        """

        # set values
        self._model_path = values['model_path']
        self._session = AnnotationModel._init_session(self._model_path)

    @property
    def __getstate__(self) -> TypedDict('Model_Path', {'model_path': Union[Path, str]}):
        """
        Gets the model path from the model object
        Important for making onnx model picklable

        Returns:
            model_path (TypedDict('Model_Path', {'model_path': Union[Path, str]})): path to saved .onnx model

        """

        # return the model path
        return {'model_path': self._model_path}

    @property
    def input_name(self) -> str:
        """

        Returns:
            input_name (str): name of model inputs

        """
        return self._session.get_inputs()[0].name

    @property
    def input_shape(self) -> np.ndarray:
        """

        Returns:
           input_shape (np.ndarray): shape of model inputs

        """
        return self._session.get_inputs()[0].shape

    @property
    def output_name(self) -> str:
        """

        Returns:
           output_name (str): name of model outputs
        """
        return self._session.get_outputs()[0].name

    @property
    def output_shape(self) -> np.ndarray:
        """

        Returns:
           output_shape (np.ndarray): shape of model outputs
        """
        return self._session.get_outputs()[0].shape
