import onnxruntime as rt
import os

class Annotation_Model:
    """
    Creates a model to annotate spectral triplets

    """

    def __init__(self, model_directory='..\\models'):

        # set up paths to model directory
        self.model_directory = self.setup_directories(model_directory)
    def setup_directories(self, model_directory):

        # strip quotes
        model_directory = model_directory.replace('"', '').replace("'", '')

        return model_directory

    def load_model(self):
        """ loads the model into ort session for prediction """

        # we will use the dc model for now, later update class to take arguments to use either dc or multitarget model
        model_path = os.path.join(self.model_directory, 'dc_model.onnx')

        self.session = rt.InferenceSession(model_path)


    def get_model_attributes(self):
        """ get attributes like input name, input shape, and output name from model """

        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_name = self.session.get_outputs()[0].name

        return input_name, input_shape, output_name

