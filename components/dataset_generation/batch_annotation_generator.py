import os
import csv
import tqdm
from datetime import datetime
from pathlib import Path
import argparse
from typing import NoReturn
from components.dataset_generation.data_cleaner import Data_Cleaner


class Batch_Annotator:

    ''' This Class carries out all functions of the batch annotator.

    Inputs:
        Import directory --> String of path to directory where raw files to be annotated are placed
        Export directory --> String of path to directory where structured files will be placed after going through batch processor.
                            These files should be unannotated, they will be annotated by the model later in the pipe

    Outputs:
        Batch Annotations --> Directory with annotation files for every patient-day file provided in the import directory

    '''

    def __init__(self, import_directory: object, export_directory: object = '..\\datasets',
                 RipVentBatchAnnotator_filepath: object = '..\\batch_annotator\RipVent.BatchProcessor.exe') -> object:
        """

        Args:
            import_directory:
            export_directory:
            RipVentBatchAnnotator_filepath:
        """
        # setup directories
        self.import_directory = import_directory
        self.export_directory = export_directory
        self.RipVentBatchAnnotator_filepath = RipVentBatchAnnotator_filepath

    def setup_directories(self, import_directory: object, export_directory: object, exe_path: object) -> object:
        """

        Args:
            import_directory:
            export_directory:
            exe_path:

        Returns:

        """
        # define paths
        import_directory = Path(import_directory.replace('"', '').replace("'", ''))
        export_directory = Path(export_directory.replace('"', '').replace("'", ''))
        exe_path = Path(exe_path.replace('"', '').replace("'", ''))

        # make export directory with timestamp
        export_directory = Path(export_directory, str(datetime.now()).replace(':', '-').replace(' ', '_'), 'batch_outputs')
        export_directory.mkdir(parents=True, exist_ok=True)

        return import_directory.resolve(), export_directory.resolve(), exe_path.resolve()

    def create_batch_csv(self, import_directory: object, export_directory: object) -> object:
        """

        Args:
            import_directory:
            export_directory:

        Returns:

        """
        ## create a csv file which will tell the batchprocessor to run across all files
        # save a list of each filename in the entry directory
        batch_csv = [x.name for x in import_directory.iterdir()]

        # batch annotator expects a blank space in the beginning of csv file so one is created
        batch_csv.insert(0, '')
        # write csv file to current directory for batch annotator to use
        batch_csv_filepath = Path(export_directory, 'batch_csv.csv')
        with open(batch_csv_filepath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(batch_csv))

        # return batch csv filepath
        batch_csv_filepath = batch_csv_filepath

        return batch_csv_filepath


    def run_batch_processor(self, import_directory: object, batch_csv_filepath: object, RipVentBatchAnnotator_filepath: object) -> object:
        """

        Args:
            import_directory:
            batch_csv_filepath:
            RipVentBatchAnnotator_filepath:
        """
        # run batch annotator and add read to allow for it to finish before moving on
        command = str(RipVentBatchAnnotator_filepath) + ' ' + str(import_directory) + ' ' + str(import_directory) + ' ' + str(batch_csv_filepath)
        os.popen(command).read()


    def delete_csv(self, batch_csv_filepath: object) -> NoReturn:
        """

        Args:
            batch_csv_filepath:
        """
        # if we created a csv file we will remove it
        batch_csv_filepath.unlink()

    def move_files_to_export_dir(self, import_directory: object, export_directory: object) -> object:
        """

        Args:
            import_directory:
            export_directory:
        """
        # list we will save files to export in
        export_files = []
        # list of strings all of our export files contain
        exports = ['Abd', 'Pes', 'RC', 'RIP', 'Spiro', 'TriggersAndArtifacts']

        for export in exports:
            export_files.extend(
                list(import_directory.glob('*.'+export+'.csv')))

        # all files that end in a .fit are export files
        export_files.extend(list(import_directory.glob('*.fit')))

        # move these saved files to the export directory
        for export_file in tqdm.tqdm(export_files, desc='Files Moved'):

            # define current export filepath
            export_filepath = Path(export_file)
            # define desired export filepath
            new_export_filepath = Path(export_directory, export_filepath.name)
            # move file
            export_filepath.rename(new_export_filepath)


    def organize_export_dir(self, export_directory: Path) -> NoReturn:
        """

        Args:
            export_directory:
        """
        # Grab the file names from the export directory
        unstructured_file_names = [x.name for x in export_directory.iterdir() if x.is_file()]

        # Get the unique set of file prefixes
        new_subdir_names = set([name.split('.')[0] for name in unstructured_file_names])

        for new_subdir in tqdm.tqdm(new_subdir_names, desc='Files Organized'):

            # define new subdir path
            new_subdir = Path(export_directory, new_subdir)
            # make subdir
            new_subdir.mkdir()

            for unstructured_file_name in unstructured_file_names:
                # import pdb; pdb.set_trace()
                if new_subdir.name == unstructured_file_name.split('.')[0]:
                    # define current unstructured_file_name path
                    unstructured_file_name_path = Path(export_directory, unstructured_file_name)
                    # define desired export filepath
                    export_filepath = Path(new_subdir, unstructured_file_name)
                    # move file
                    unstructured_file_name_path.rename(export_filepath)

    def batch_process(self, raw_import_directory: object, raw_export_directory: object, raw_RipVentBatchAnnotator_filepath: object) -> object:
        """

        Args:
            raw_import_directory:
            raw_export_directory:
            raw_RipVentBatchAnnotator_filepath:

        Returns:

        """
        print('Batch Processing Starting!')

        # # setup import and export directories
        import_directory, export_directory, RipVentBatchAnnotator_filepath = self.setup_directories(raw_import_directory, raw_export_directory, raw_RipVentBatchAnnotator_filepath)

        # put methods together to run batch processor
        batch_csv_filepath = self.create_batch_csv(import_directory, export_directory)
        print('Creating Batch Files')
        self.run_batch_processor(import_directory, batch_csv_filepath, RipVentBatchAnnotator_filepath)
        print('Batch File Creation Finished')
        self.delete_csv(batch_csv_filepath)
        print('Moving Batch Files to Export dir')
        self.move_files_to_export_dir(import_directory, export_directory)
        print('Batch Files Moved')
        print('Organizing Batch Files')
        self.organize_export_dir(export_directory)
        print('Batch Files Organized')
        print('Batch Processing Done!')

        # return export directory for ease of use
        return export_directory

    def batch_process_and_validate(self) -> object:
        """

        Returns:

        """
        # instantiate data cleaner
        data_cleaner = Data_Cleaner()

        # batch process files
        export_directory = self.batch_process(self.import_directory, self.export_directory, self.RipVentBatchAnnotator_filepath)

        ## check the batch files directory for errors

        # check for duplicate patient days
        data_cleaner.check_for_duplicate_pt_days(export_directory)

        # check for invalid directories
        num_invalid, invalid_dir = data_cleaner.check_for_invalid_subdirs(export_directory)

        # while the number of invalid files is greater than 0, run the batch processor over the invalid files to fix them
        while num_invalid > 0:

            # batch process files
            self.batch_process(invalid_dir, self.export_directory, self.RipVentBatchAnnotator_filepath)

            # check the num invalid again
            num_invalid, invalid_dir = data_cleaner.check_for_invalid_subdirs(export_directory)

        # return export directory for other functions
        return export_directory


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

    print(f'Batch outputs generated at {export_directory}')
