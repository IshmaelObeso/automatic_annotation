import onnxruntime as rt
import os
from pathlib import Path

class Annotation_Model:
    """
    Creates a model to annotate spectral triplets

    """

    def __init__(self, model_path):

        # instantiate model
        model_path = Path(model_path)

        # load path as string, because onnx does not accept pathlib paths
        self.session = rt.InferenceSession(str(model_path))

    def get_model_attributes(self):
        """ get attributes like input name, input shape, and output name from model """

        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_name = self.session.get_outputs()[0].name

        return input_name, input_shape, output_name

