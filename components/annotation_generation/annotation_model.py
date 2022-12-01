import onnxruntime as rt
from pathlib import Path

class Annotation_Model:
    """
    Creates a model to annotate spectral triplets

    """

    def __init__(self, model_path: object) -> object:
        """

        Args:
            model_path:
        """
        # instantiate model
        self.model_path = str(Path(model_path))

        # load path as string, because onnx does not accept pathlib paths
        self.session = self.init_session(self.model_path)

    def run(self, *args: object) -> object:
        """

        Args:
            *args:

        Returns:

        """
        return self.session.run(*args)

    def get_model_attributes(self) -> object:
        """ get attributes like input name, input shape, and output name from model """

        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_name = self.session.get_outputs()[0].name

        return input_name, input_shape, output_name

    def init_session(model_path):
        providers_list = ['CPUExecutionProvider']
        sess = rt.InferenceSession(model_path, providers=providers_list)
        return sess

    def __getstate__(self) -> object:
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.session = self.init_session(self.model_path)
