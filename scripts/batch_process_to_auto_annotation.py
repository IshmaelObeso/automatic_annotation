import argparse
import os
import sys
import shutil

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

# define some complex types
ThresholdsDict = dict[str, float]
FilterFileInfo = dict[dict[str, bool], dict[str, str], dict[dict[str, str], dict[str, str]]]

def main(
        raw_files_directory: str,
        export_directory: str = '\\datasets',
        batch_processor_exe_filepath: str = '.\\batch_annotator\RipVent.BatchProcessor.exe',
        thresholds_dict: ThresholdsDict = None,
        generate_triplets_and_statics: bool = True,
        generate_annotations: bool = True,
        filter_file_info: FilterFileInfo = None,
        delete_triplets_and_spectral_triplets: bool = False,
        multiprocessing: bool = False
         ) -> None:
    """
        This function will take raw REDVENT patient-day files and process them, users can flag how much processing should be done.
        Basic functionality is as follows:
        raw_patient_day_files --> batch_process raw files --> generate triplets --> generate spectral triplets --
        --> generate predictions --> generate .art files (files that contain predictions) --> generate annotated dataset



    Args:
        raw_files_directory (str): directory where raw patient-day files are stored
        export_directory (str): directory to store outputs
        batch_processor_exe_filepath (str): filepath to the batch processor .exe
        thresholds_dict (Dict[str, Dict[str, float]]): dictionary with information about thresholds for models
        generate_triplets_and_statics (bool): bool that controls whether to generate triplets and statics files
        generate_annotations (bool): bool that controls whether to generate annotations
        filter_file_info (Dict[str, bool]): dict that contains information about the filter file,
                                            i.e. which columns to filter by
        delete_triplets_and_spectral_triplets (bool): bool that controls whether to delete the triplets and spectral
                                                    triplets directory after the program is finished (to save space)
        multiprocessing (bool): bool that controls whether to multiprocess or not

    Returns:
        None:
    """

    ## TODO: Implement input checker, to validate whether the inputs to main are valid,
    ## this allows us to get rid of some scattered assert statements

    # # save thresholds to models from inputs
    for model, threshold in thresholds_dict.items():

        MODELS_DICT[model]['threshold'] = threshold

    # if generate_annotations is true, then generate_triplets_and_statics must also be true
    if (generate_annotations is True) and (generate_triplets_and_statics is False):

        generate_triplets_and_statics = True

        print('Generating triplets and statics must happen before generating Annotations. \n Generate Annotations set to True.')

    assert raw_files_directory is not None, 'Import Directory must be provided '

    # instantiate batch annotator class
    batch_annotator = batch_annotation_generator.Batch_Annotator(raw_files_directory, export_directory, batch_processor_exe_filepath)

    # run batch annotator, save the directory it exports the batch annotations to
    batch_export_directory = batch_annotator.batch_process_and_validate()

    if generate_triplets_and_statics:

        # instantiate triplet generator class, and include a filter filepath if given one
        triplet_generator = triplets_generator.TripletGenerator(batch_export_directory)

        # run triplet generator
        triplet_export_directory, triplet_statics_directory = triplet_generator.generate_triplets(multiprocessing)

        print(f'Triplets generated at {os.path.abspath(triplet_export_directory)}')
        print(f"Statics file generated at {os.path.abspath(triplet_statics_directory)}")

        # instantiate spectral triplet generator class
        spectral_triplet_generator = spectral_triplets_generator.SpectralTripletGenerator(triplet_export_directory, filter_file_info=filter_file_info)

        # run spectral triplet generator
        spectral_triplets_export_directory, spectral_statics_directory = spectral_triplet_generator.generate_spectral_triplets(multiprocessing)

        print(f'Spectral Triplets generated at {os.path.abspath(spectral_triplets_export_directory)}')
        print(f"Spectral Statics file generated at {os.path.abspath(spectral_statics_directory)}")

        # if we want to generate annotations
        if generate_annotations:

            # instantiate models and save model objects to dict
            for model_name, parameters in MODELS_DICT.items():

                # grab model path from dict and instantiate
                model_object = annotation_model.AnnotationModel(parameters['path'])

                # save model object in dict
                MODELS_DICT[model_name]['model_object'] = model_object

            # instantiate predictions wrapper
            predictions_wrapper = predictions_generator.PredictionAggregator(spectral_triplets_directory=spectral_triplets_export_directory)

            # generate all predictions and output as predictions dataframe
            predictions_df = predictions_wrapper.generate_all_predictions(models_dict=MODELS_DICT)

            # instantiate annotated dataset generator
            annotation_generator = annotated_dataset_generator.AnnotatedDatasetGenerator(raw_files_directory=raw_files_directory, spectral_triplets_directory=spectral_triplets_export_directory)

            # create artifact file from binary predictions and info from model settings
            annotation_generator.create_art_files(predictions_df, models_dict=MODELS_DICT)

            # add multitarget predictions to artifact files
            print(f'Annotated Dataset Created at {annotation_generator._annotated_dataset_directory}')

    # finally delete triplets and spectral triplets directory if cleanup is selected
    if generate_triplets_and_statics and delete_triplets_and_spectral_triplets:
        print('Deleting Triplets and Spectral Triplets Folders')

        # if triplets directory exists, delete it and its contents
        if triplet_export_directory.is_dir():
            shutil.rmtree(triplet_export_directory)
            print('Triplets Folder Deleted')

        # if spectral triplets directory exists, delete it and its contents
        if spectral_triplets_export_directory.is_dir():
            shutil.rmtree(spectral_triplets_export_directory)
            print('Spectral Triplets Folder Deleted')

    print('----DONE----')

if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--raw_files_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--export_directory', type=str, default='\\datasets',
                   help='Directory to export datasets to')
    p.add_argument('--batch_processor_exe_filepath', type=str, default='.\\batch_annotator\RipVent.BatchProcessor.exe',
                   help='Path to vent annotator')
    p.add_argument('--generate_triplets_and_statics', type=bool, default=True)
    p.add_argument('--generate_annotations', type=bool, default=True)
    p.add_argument('--delete_triplets_and_spectral_triplets', type=bool, default=False)
    p.add_argument('--double_trigger_threshold', type=float, default=.9)
    p.add_argument('--auto_trigger_threshold', type=float, default=.9)
    p.add_argument('--delayed_termination_threshold', type=float, default=.9)
    p.add_argument('--flow_undershoot_threshold', type=float, default=.9)
    p.add_argument('--premature_termination_threshold', type=float, default=.9)
    p.add_argument('--reverse_trigger_threshold', type=float, default=.9)
    p.add_argument('--use_filter_file', type=bool, default=False)
    p.add_argument('--filter_filepath', type=str, default=None)
    p.add_argument('--exclude_columns_and_values',
                   type=dict[str, str],
                   default={'Reviewed by:': 'NaN', 'analysis exclusion': 'not NaN'})
    p.add_argument('--multiprocessing', type=bool, default=False)

    args = vars(p.parse_args())

    # define args
    raw_files_directory = args['raw_files_directory']
    export_directory = args['export_directory']
    batch_processor_exe_filepath = args['batch_processor_exe_filepath']
    use_filter_file = args['use_filter_file']
    filter_filepath = args['filter_filepath']
    exclude_columns_and_values = args['exclude_columns_and_values']
    generate_triplets_and_statics = args['generate_triplets_and_statics']
    generate_annotations = args['generate_annotations']
    delete_triplets_and_spectral_triplets = args['delete_triplets_and_spectral_triplets']
    multiprocessing=args['multiprocessing']

    # threshold args
    double_trigger_threshold = args['double_trigger_threshold']
    auto_trigger_threshold = args['auto_trigger_threshold']
    delayed_termination_threshold = args['delayed_termination_threshold']
    flow_undershoot_threshold = args['flow_undershoot_threshold']
    premature_termination_threshold = args['premature_termination_threshold']
    reverse_trigger_threshold = args['reverse_trigger_threshold']

    # define filter file info dict
    filter_file_info = {}
    filter_file_info['use'] = use_filter_file
    filter_file_info['filepath'] = filter_filepath
    filter_file_info['exclude_columns_and_values'] = exclude_columns_and_values

    # define threshold dict
    threshold_dict = {}
    threshold_dict['Double Trigger'] = {'Double Trigger': double_trigger_threshold}
    threshold_dict['Autotrigger'] = {'Autotrigger': auto_trigger_threshold}
    threshold_dict['Delayed Termination'] = {'Delayed Termination': delayed_termination_threshold}
    threshold_dict['Flow Undershoot'] = {'Flow Undershoot': flow_undershoot_threshold}
    threshold_dict['Premature Termination'] = {'Premature Termination': premature_termination_threshold}
    threshold_dict['Reverse Trigger'] = {'Reverse Trigger': reverse_trigger_threshold}

    # run main
    main(
        raw_files_directory,
        export_directory,
        batch_processor_exe_filepath,
        threshold_dict,
        generate_triplets_and_statics,
        generate_annotations,
        filter_file_info,
        delete_triplets_and_spectral_triplets,
        multiprocessing
    )

