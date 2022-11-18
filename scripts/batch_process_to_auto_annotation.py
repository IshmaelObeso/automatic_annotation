import argparse
import os
import sys
import pandas as pd

sys.path.append('..')

from components.dataset_generation import batch_annotation_generator, spectral_triplets_generator, triplets_generator
from components.annotation_generation import annotated_dataset_generator, annotation_model, predictions_generator
from components.annotation_generation.utilities.model_settings import MODELS_DICT


# This script creates annotations from raw REDVENT files
# This script will run the batch annotator on raw patient-day files, organize them into output directories,
# Then it will run the triplet generator on the organized batch outputs, generate triplet directories and a statics file
# Then it will run the spectral triplet generator on the triplets directory, generate spectral triplets directories and a spectral triplets statics file
# Then it will predict on every spectral triplet generated
# Then it will output those predictions to a directory with the raw files

def main(
        import_directory,
        export_directory ='\\datasets',
        vent_annotator_filepath='.\\batch_annotator\RipVent.BatchProcessor.exe',
        binary_threshold=None,
        multitarget_thresholds=None,
        generate_triplets_and_statics=True,
        generate_annotations=True,
        filter_file_info=None,
         ):

    # if generate_annotations is true, then generate_triplets_and_statics must also be true
    if (generate_annotations is True) and (generate_triplets_and_statics is False):

        generate_triplets_and_statics = True

        print('Generating triplets and statics must happen before generating Annotations. \n Generate Annotations set to True.')

    assert import_directory is not None, 'Import Directory must be provided '

    # instantiate batch annotator class
    batch_annotator = batch_annotation_generator.Batch_Annotator(import_directory, export_directory, vent_annotator_filepath)

    # run batch annotator, save the directory it exports the batch annotations to
    export_directory = batch_annotator.batch_process()

    if generate_triplets_and_statics:


        # instantiate triplet generator class, and include a filter filepath if given one
        triplet_generator = triplets_generator.Triplet_Generator(export_directory)

        # run triplet generator
        export_directory = triplet_generator.generate_triplets()
        statics_csv_output = triplet_generator.statics_output_path_csv

        print(f'Triplets generated at {os.path.abspath(export_directory)}')
        print(f"Statics file generated at {os.path.abspath(statics_csv_output)}")



        # instantiate spectral triplet generator class
        spectral_triplet_generator = spectral_triplets_generator.Spectral_Triplet_Generator(export_directory, filter_file_info=filter_file_info)

        # run spectral triplet generator
        spectral_triplets_directory = spectral_triplet_generator.generate_spectral_triplets()
        statics_csv_output = spectral_triplet_generator.statics_output_path_csv

        print(f'Spectral Triplets generated at {os.path.abspath(spectral_triplets_directory)}')
        print(f"Spectral Statics file generated at {os.path.abspath(statics_csv_output)}")

        if generate_annotations:

            # instantiate models and save model objects to dict
            for model_name, parameters in MODELS_DICT.items():

                # grab model path from dict and instantiate
                model_object = annotation_model.Annotation_Model(parameters['path'])

                # save model object in dict
                MODELS_DICT[model_name]['model_object'] = model_object
            # save thresholds to models from inputs
            MODELS_DICT['Binary Double Trigger']['thresholds'] = binary_threshold
            MODELS_DICT['Multi-Target']['thresholds'] = multitarget_thresholds

            # instantiate predictions wrapper
            predictions_wrapper = predictions_generator.PredictionAggregator(spectral_triplets_directory=spectral_triplets_directory)

            # generate all predictions and output as predictions dataframe
            predictions_df = predictions_wrapper.generate_all_predictions(models_dict=MODELS_DICT)

            # instantiate annotated dataset generator
            annotation_generator = annotated_dataset_generator.AnnotatedDatasetGenerator(raw_files_directory=import_directory, spectral_triplets_directory=spectral_triplets_directory)

            # create artifact file from binary predictions
            annotation_generator.create_art_files(predictions_df)

            # add multitarget predictions to artifact files
            print(f'Annotated Dataset Created at {annotation_generator.annotated_dataset_directory}')

            print('---------Done!---------')

if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--import_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--export_directory', type=str, default='\\datasets',
                   help='Directory to export datasets to')
    p.add_argument('--batch_processor_exe_filepath', type=str, default='.\\batch_annotator\RipVent.BatchProcessor.exe',
                   help='Path to vent annotator')
    p.add_argument('--generate_triplets_and_statics', type=bool, default=True)
    p.add_argument('--generate_annotations', type=bool, default=True)
    p.add_argument('--binary_threshold', type=dict[str, int], default={'Double Trigger': .804})
    p.add_argument('--multitarget_thresholds',
                   help='[reverse_trigger_threshold, premature_termination_threshold, flow_undershoot_threshold]',
                   type=dict[str, int],
                   default={'Double Trigger Reverse Trigger': 4.8e-02,
                            'Double Trigger Premature Termination': 3.2e-02,
                            'Double Trigger Flow Undershoot': 0.71})
    p.add_argument('--use_filter_file', type=bool, default=False)
    p.add_argument('--filter_filepath', type=str, default="C:\\Users\\iobeso\\Documents\\Asynchrony_Project_file__master.xlsx")
    p.add_argument('--exclude_columns_and_values',
                   type=dict[str, str],
                   default={'Reviewed by:': 'NaN', 'analysis exclusion': 'not NaN'})

    args = vars(p.parse_args())

    # define args
    import_directory = args['import_directory']
    export_directory = args['export_directory']
    vent_annotator_filepath = args['batch_processor_exe_filepath']
    binary_threshold = args['binary_threshold']
    multitarget_thresholds = args['multitarget_thresholds']

    use_filter_file = args['use_filter_file']
    filter_filepath = args['filter_filepath']
    exclude_columns_and_values = args['exclude_columns_and_values']

    generate_triplets_and_statics = args['generate_triplets_and_statics']
    generate_annotations = args['generate_annotations']

    # TESTING
    filter_file_info = {}
    filter_file_info['use'] = use_filter_file
    filter_file_info['filepath'] = filter_filepath
    filter_file_info['exclude_columns_and_values'] = exclude_columns_and_values

    # run main
    main(
        import_directory,
        export_directory,
        vent_annotator_filepath,
        binary_threshold,
        multitarget_thresholds,
        generate_triplets_and_statics,
        generate_annotations,
        filter_file_info
    )

