from components.annotation_generation import annotated_dataset_generator, predictions_generator

raw_files_directory = r"C:\Users\gobes\Documents\raw_files_only"

dataset = r"C:\Users\gobes\Documents\Unannotated_Raw_Exports\2022-09-30,15-12-50.135596\spectral_triplets"

model_directory = r"C:\Users\gobes\PycharmProjects\automatic_annotation\models"

# instantiate prediction generator
predictions_generator = predictions_generator.Prediction_Generator(dataset, model_directory)

# get predictions
preds = predictions_generator.get_predictions()

# try to use annotation generator
annotation_generator = annotated_dataset_generator.Annotated_Dataset_Generator(raw_files_directory, dataset, preds)

annotation_generator.create_art_files()

import pdb; pdb.set_trace()