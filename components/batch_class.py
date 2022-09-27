import os
from shutil import move
import glob
import csv
import tqdm
from datetime import datetime
import argparse



class Batch_Annotator:
    ''' This Class carries out all functions of the batch annotator.

    Inputs:
        Import directory --> String of path to directory where raw files to be annotated are placed
        Export directory --> String of path to directory where structured files will be placed after going through batch processor.
                            These files should be unannotated, they will be annotated by the model later in the pipe

    Outputs:
        Batch Annotations --> Directory with annotation files for every patient-day file provided in the import directory

    '''

    def __init__(self, import_directory, dataset_directory='..\\datasets', RipVentBatchAnnotator_filepath = '..\\batch_annotator\RipVent.BatchProcessor.exe'):

        # # setup import and export directories
        self.import_directory, self.export_directory = self.setup_directories(import_directory, dataset_directory)
        self.RipVentBatchAnnotator_filepath = RipVentBatchAnnotator_filepath
    def setup_directories(self, import_directory, dataset_directory):

        # strip quotes
        import_directory = import_directory.replace('"', '').replace("'", '')
        dataset_directory = dataset_directory.replace('"', '').replace("'", '')

        # make export directory with timestamp
        export_directory = os.path.join(dataset_directory, 'batch_outputs', str(datetime.now()).replace(':', '-').replace(' ', ','))
        os.makedirs(export_directory)

        return import_directory, export_directory

    def create_batch_csv(self):

        ## create a csv file which will tell the batchprocessor to run across all files
        # save a list of each filename in the entry directory
        batch_csv = os.listdir(self.import_directory)
        # batch annotator expects a blank space in the beginning of csv file so one is created
        batch_csv.insert(0, '')
        # write csv file to current directory for batch annotator to use
        batch_csv_filepath = os.path.join(self.export_directory, 'batch_csv.csv')
        with open(batch_csv_filepath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(batch_csv))

        # save batch csv to attribute
        self.batch_csv = batch_csv
        self.batch_csv_filepath = batch_csv_filepath

    def run_batch_processor(self):

        # import pdb; pdb.set_trace()
        # run batch annotator and add read to allow for it to finish before moving on
        os.popen(self.RipVentBatchAnnotator_filepath + ' ' + self.import_directory + ' ' + self.import_directory + ' ' + self.batch_csv_filepath).read()

    def delete_csv(self):

        # if we created a csv file we will remove it
        os.remove(self.batch_csv_filepath)

    def move_files_to_export_dir(self):

        # list we will save files to export in
        export_files = []
        # list of strings all of our export files contain
        exports = ['Abd', 'Pes', 'RC', 'RIP', 'Spiro', 'TriggersAndArtifacts']

        # loop through different export file names and append to a list
        for export in exports:
            export_files.extend(
                [fn for fn in glob.glob(self.import_directory.strip() + "\\*") if os.path.basename(fn).endswith(export + '.csv')])
        # all files that end in a .fit are export files
        export_files.extend([fn for fn in glob.glob(self.import_directory.strip() + "\\*.fit")])

        # move these saved files to the export directory
        for export_file in tqdm.tqdm(export_files):
            move(export_file, self.export_directory.strip() + "\\")

    def organize_export_dir(self):

        # Grab the file names from the export directory
        unstructured_file_names = os.listdir(self.export_directory)

        # Get the unique set of file prefixes
        new_subdir_names = set([name.split('.')[0] for name in unstructured_file_names])

        for new_subdir in tqdm.tqdm(new_subdir_names):
            # Make the new subdirectory for this patient day
            new_subdir_absolute_path = os.path.join(self.export_directory, new_subdir)

            os.mkdir(new_subdir_absolute_path)

            # Move the associated files into the subdirectory
            for unstructured_file_name in unstructured_file_names:
                if new_subdir == unstructured_file_name.split('.')[0]:
                    move(os.path.join(self.export_directory, unstructured_file_name),
                         os.path.join(new_subdir_absolute_path, unstructured_file_name))

    def batch_process(self):

        print('Batch Processing Starting!')

        # put methods together to run batch processor
        self.create_batch_csv()
        print('Creating Batch Files')
        self.run_batch_processor()
        print('Batch File Creation Finished')
        self.delete_csv()
        print('Moving Batch Files to Export dir')
        self.move_files_to_export_dir()
        print('Batch Files Moved')
        print('Organizing Batch Files')
        self.organize_export_dir()
        print('Batch Files Organized')
        print('Batch Processing Done!')

        # return export directory for ease of use
        return self.export_directory

# if running this file directly, only do batch processing
if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--dataset_directory', type=str, default='..\\datasets', help='Directory to export organized unannotated files for later processing')
    p.add_argument('--batch_annotator_filepath', type=str, default='..\\batch_annotator\RipVent.BatchProcessor.exe', help='Path to vent annotator')
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']
    dataset_directory = args['dataset_directory']
    batch_annotator_filepath = args['batch_annotator_filepath']

    # instantiate batch annotator class
    batch_annotator = Batch_Annotator(input_directory, dataset_directory, batch_annotator_filepath)

    # run batch processor
    export_directory = batch_annotator.batch_process()

    print(f'Batch outputs generated at {os.path.abspath(export_directory)}')
