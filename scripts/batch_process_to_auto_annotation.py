import argparse
import os
import sys

sys.path.append('..')

from components import batch_annotation_generator, triplets_generator, spectral_triplets_generator, predictions_generator, annotation_model, annotated_dataset_generator

# This script creates annotations from raw REDVENT files
# This script will run the batch annotator on raw patient-day files, organize them into output directories,
# Then it will run the triplet generator on the organized batch outputs, generate triplet directories and a statics file
# Then it will run the spectral triplet generator on the triplets directory, generate spectral triplets directories and a spectral triplets statics file
# Then it will predict on every spectral triplet generated
# Then it will output those predictions to a directory with the raw files

def main(input_directory, dataset_directory='\\datasets', vent_annotator_filepath='.\\batch_annotator\RipVent.BatchProcessor.exe', threshold=.804):

    # instantiate batch annotator class
    batch_annotator = batch_annotation_generator.Batch_Annotator(input_directory, dataset_directory, vent_annotator_filepath)

    # run batch annotator, save the directory it exports the batch annotations to
    export_directory = batch_annotator.batch_process()

    # instantiate triplet generator class
    triplet_generator = triplets_generator.Triplet_Generator(export_directory)

    # run triplet generator
    export_directory = triplet_generator.generate_triplets()
    statics_csv_output = triplet_generator.statics_output_path_csv

    print(f'Triplets generated at {os.path.abspath(export_directory)}')
    print(f"Statics file generated at {os.path.abspath(statics_csv_output)}")

    # instantiate spectral triplet generator class
    spectral_triplet_generator = spectral_triplets_generator.Spectral_Triplet_Generator(export_directory)

    # run spectral triplet generator
    spectra_triplets_directory = spectral_triplet_generator.generate_spectral_triplets()
    statics_csv_output = spectral_triplet_generator.statics_output_path_csv

    print(f'Spectral Triplets generated at {os.path.abspath(spectra_triplets_directory)}')
    print(f"Spectral Statics file generated at {os.path.abspath(statics_csv_output)}")


    # instantiate prediction generator
    prediction_generator = predictions_generator.Prediction_Generator(spectra_triplets_directory)

    # instantiate dc model
    dc_model_path = '.\\models\\dc_model.onnx'
    dc_model = annotation_model.Annotation_Model(dc_model_path)

    # instantiate multitarget model
    multitarget_model_path = '.\\models\\mt_model.onnx'
    multitarget_model = annotation_model.Annotation_Model(multitarget_model_path)

    # get predictions for dc model
    predictions_export_directory = prediction_generator.get_predictions(dc_model, threshold)

    # get predictions for multitarget model


    # try to use annotation generator
    annotation_generator = annotated_dataset_generator.Annotated_Dataset_Generator(input_directory, spectra_triplets_directory, predictions_export_directory)

    annotation_generator.create_art_files()

    print(f'Annotated Dataset Created at {annotation_generator.annotated_dataset_directory}')

if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--dataset_directory', type=str, default='\\datasets',
                   help='Directory to export datasets to')
    p.add_argument('--batch_processor_exe_filepath', type=str, default='.\\batch_annotator\RipVent.BatchProcessor.exe',
                   help='Path to vent annotator')
    p.add_argument('--threshold', type=float, default=.804)
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']
    dataset_directory = args['dataset_directory']
    vent_annotator_filepath = args['batch_processor_exe_filepath']
    threshold = args['threshold']

    # run main
    main(input_directory, dataset_directory, vent_annotator_filepath, threshold)

