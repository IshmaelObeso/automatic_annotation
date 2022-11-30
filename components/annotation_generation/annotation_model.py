import onnxruntime as rt
import os
from pathlib import Path


def init_session(model_path):
    # always run on CPU for safe
    sess = rt.InferenceSession(model_path)
    return sess

class Annotation_Model:
    """
    Creates a model to annotate spectral triplets

    """

    def __init__(self, model_path):

        # instantiate model
        self.model_path = str(Path(model_path))

        # load path as string, because onnx does not accept pathlib paths
        self.session = init_session(self.model_path)

    def run(self, *args):
        return self.session.run(*args)

    def get_model_attributes(self):
        """ get attributes like input name, input shape, and output name from model """

        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_name = self.session.get_outputs()[0].name

        return input_name, input_shape, output_name

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.session = init_session(self.model_path)

