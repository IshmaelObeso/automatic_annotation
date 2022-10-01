import tqdm
import pandas as pd
import numpy as np
from components import prediction_data_pipe, annotation_model


class Prediction_Generator:

    def __init__(self, spectral_dataset_directory, model_directory='..\\models' ):

        #instantiate dataset and model classes
        self.data_class = prediction_data_pipe.Data_Pipe(spectral_dataset_directory)

        self.model = annotation_model.Annotation_Model(model_directory)

    def get_predictions(self):
        """ get predictions for every spectral triplet in dataset, predictions will be output as hdf file"""

        # load model session
        self.model.load_model()

        # get model attributes
        input_name, input_shape, output_name = self.model.get_model_attributes()

        # test getitem for every uid
        preds_list = []
        truths_list = []
        patient_id_list = []
        day_id_list = []
        breath_id_list = []

        # get all uids in dataset
        all_uids = self.data_class.get_uids()

        # for every spectrogram in dataset
        for uid in tqdm.tqdm(all_uids, desc='Generating Predictions'):

            # get spectrograms for every uid
            xs, ys, ts, uid = self.data_class.__getitem__(uid)

            # reshape
            xs = np.reshape(xs, input_shape)

            # get prediction
            pred = self.model.session.run([output_name], {input_name: xs.astype(np.float32)})
            # get prediction as a singular array
            pred = pred[0][0]

            # get the max of truth ie. was there a dyssynchrony in this triplet or not
            ys = np.nan_to_num(ys).max()

            # append to list to make into df later
            preds_list.append(pred)
            truths_list.append(ys)
            patient_id_list.append(uid[0])
            day_id_list.append(uid[1])
            breath_id_list.append(uid[2])

        # stack preds and truths into array
        preds = np.stack(preds_list).flatten()
        truths = np.stack(truths_list)

        # make predictions dataframe
        df = pd.DataFrame({'patient_id': patient_id_list,
                           'day_id': day_id_list,
                           'breath_id': breath_id_list})

        preds_df = pd.DataFrame({'prediction': preds, 'truth': truths})

        preds_df = pd.concat([df, preds_df], axis=1)

        preds_df = preds_df.set_index(['patient_id', 'day_id', 'breath_id'])

        return preds_df