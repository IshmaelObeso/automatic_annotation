import argparse
import os
import sys

sys.path.append('..')

from components import batch_annotation_generator, triplets_generator

# This script will run the batch annotator on raw patient-day files, organize them into output directories,
# Then it will run the triplet generator on the organized batch outputs, generate triplet directories and a statics file

def main(input_directory, dataset_directory='\\datasets', vent_annotator_filepath='.\\batch_annotator\RipVent.BatchProcessor.exe'):

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


# if running this file directly, only do batch processing
if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--dataset_directory', type=str, default='\\datasets',
                   help='Directory to export datasets to')
    p.add_argument('--batch_processor_exe_filepath', type=str, default='.\\batch_annotator\RipVent.BatchProcessor.exe',
                   help='Path to vent annotator')
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']
    dataset_directory = args['dataset_directory']
    vent_annotator_filepath = args['batch_processor_exe_filepath']

    # run main
    main(input_directory, dataset_directory, vent_annotator_filepath)
