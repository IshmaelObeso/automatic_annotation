import argparse
import os
import sys

sys.path.append('..')

from components.dataset_generation import batch_annotation_generator, spectral_triplets_generator, triplets_generator
from components.annotation_generation import annotated_dataset_generator, annotation_model, predictions_generator


# This script creates annotations from raw REDVENT files
# This script will run the batch annotator on raw patient-day files, organize them into output directories,
# Then it will run the triplet generator on the organized batch outputs, generate triplet directories and a statics file
# Then it will run the spectral triplet generator on the triplets directory, generate spectral triplets directories and a spectral triplets statics file
# Then it will predict on every spectral triplet generated
# Then it will output those predictions to a directory with the raw files

def main(
         input_directory, dataset_directory='\\datasets',
         vent_annotator_filepath='.\\batch_annotator\RipVent.BatchProcessor.exe',
         binary_threshold=.804,
         multitarget_thresholds=[.001, 1.14e-05],
         generate_triplets_and_statics=True,
         generate_annotations=True
         ):

    # instantiate batch annotator class
    batch_annotator = batch_annotation_generator.Batch_Annotator(input_directory, dataset_directory, vent_annotator_filepath)

    # run batch annotator, save the directory it exports the batch annotations to
    export_directory = batch_annotator.batch_process()

    if generate_triplets_and_statics:

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
        spectral_triplets_directory = spectral_triplet_generator.generate_spectral_triplets()
        statics_csv_output = spectral_triplet_generator.statics_output_path_csv

        print(f'Spectral Triplets generated at {os.path.abspath(spectral_triplets_directory)}')
        print(f"Spectral Statics file generated at {os.path.abspath(statics_csv_output)}")

        if generate_annotations:

            # instantiate binary prediction generator
            binary_prediction_generator = predictions_generator.BinaryPredictionGenerator(spectral_triplets_directory)

            # instantiate multitarget prediction generator
            multitarget_prediction_generator = predictions_generator.MultitargetPredictionGenerator(spectral_triplets_directory)

            # instantiate dc model
            dc_model_path = '.\\models\\dc_model.onnx'
            dc_model = annotation_model.Annotation_Model(dc_model_path)

            # instantiate multitarget model
            multitarget_model_path = '.\\models\\mt_model.onnx'
            multitarget_model = annotation_model.Annotation_Model(multitarget_model_path)

            # get predictions for dc model
            binary_preds_df = binary_prediction_generator.get_predictions(dc_model, binary_threshold)

            # get predictions for multitarget model
            multitarget_preds_df = multitarget_prediction_generator.get_predictions(multitarget_model, multitarget_thresholds)

            # instantiate annotated dataset generator
            annotation_generator = annotated_dataset_generator.AnnotatedDatasetGenerator(raw_files_directory=input_directory, spectral_triplets_directory=spectral_triplets_directory)

            # create artifact file from binary predictions
            annotation_generator.create_art_files(binary_preds_df, multitarget_preds_df)

            # add multitarget predictions to artifact files

            print(f'Annotated Dataset Created at {annotation_generator.annotated_dataset_directory}')

if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--dataset_directory', type=str, default='\\datasets',
                   help='Directory to export datasets to')
    p.add_argument('--batch_processor_exe_filepath', type=str, default='.\\batch_annotator\RipVent.BatchProcessor.exe',
                   help='Path to vent annotator')
    p.add_argument('--generate_triplets_and_statics', type=bool, default=True)
    p.add_argument('--generate_annotations', type=bool, default=True)
    p.add_argument('--binary_threshold', type=float, default=.804)
    p.add_argument('--multitarget_thresholds', help='[reverse_trigger_threshold, inadequate_support_threshold]', type=list, default=[.25, 3.3e-01])
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']
    dataset_directory = args['dataset_directory']
    vent_annotator_filepath = args['batch_processor_exe_filepath']
    binary_threshold = args['binary_threshold']
    multitarget_thresholds = args['multitarget_thresholds']

    generate_triplets_and_statics = args['generate_triplets_and_statics']
    generate_annotations = args['generate_annotations']

    # if generate_annotations is true, then generate_triplets_and_statics must also be true
    if generate_annotations:
        generate_triplets_and_statics = True

    # run main
    main(
        input_directory,
        dataset_directory,
        vent_annotator_filepath,
        binary_threshold,
        multitarget_thresholds,
        generate_triplets_and_statics,
        generate_annotations
    )

