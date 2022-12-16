import os
import csv
import tqdm
from datetime import datetime
from pathlib import Path
import argparse
from components.dataset_generation.data_cleaner import Data_Cleaner


class Batch_Annotator:

    ''' This Class turns a directory of raw REDVENT patient-day files into an
        organized directory of batch processed patient-day files

    Attributes:
        raw_files_directory (str): directory where raw patient-day files are stored
        export_directory (str): directory to store outputs
        batch_processor_filepath (str): filepath to the batch processor .exe

    '''

    def __init__(self, raw_files_directory: str, export_directory: str = '..\\datasets',
                 batch_processor_filepath: str = '..\\batch_annotator\RipVent.BatchProcessor.exe') -> None:
        """
        Sets initial class attributes

        Args:
            raw_files_directory (str): directory where raw patient-day files are stored
            export_directory (str): directory to store outputs
            batch_processor_filepath (str): filepath to the batch processor .exe

        Returns:
            None:
        """
        # save attributes
        self.raw_files_directory = raw_files_directory
        self.export_directory = export_directory
        self.batch_processor_filepath = batch_processor_filepath

    def _setup_directories(self, import_directory: str, export_directory: str, exe_path: str) -> tuple[Path, Path, Path]:
        """

        This method takes import_directory, export_directory, and exe_paths as strings, creates the export directory
        if it does not exist, and then returns these path strings as Path objects for ease-of-use later. It also changes
        the export directory path to point to the batch_outputs folder in the export directory

        Args:
            import_directory (str): directory where raw patient-day files are stored
            export_directory (str): directory to store outputs
            exe_path (str): filepath to the batch processor .exe

        Returns:
            tuple[Path, Path, Path]: returns Paths of import_directory, export_directory, exe_path as Path objects for
                                    easy manipulation

        """
        # define paths
        import_directory = Path(import_directory.replace('"', '').replace("'", ''))
        export_directory = Path(export_directory.replace('"', '').replace("'", ''))
        exe_path = Path(exe_path.replace('"', '').replace("'", ''))

        # make export directory with timestamp
        export_directory = Path(export_directory, str(datetime.now()).replace(':', '-').replace(' ', '_'), 'batch_outputs')
        export_directory.mkdir(parents=True, exist_ok=True)

        return import_directory.resolve(), export_directory.resolve(), exe_path.resolve()

    def _create_batch_csv(self, import_directory: Path, export_directory: Path) -> Path:
        """
        This method creates a csv file (batch csv) which will tell the batch processor which files to run on
        (all the ones in the import directory)

        Args:
            import_directory (Path): Path object that stores directory where raw patient-day files are stored
            export_directory (Path): Path object that stores directory to store outputs

        Returns:
            batch_csv_filepath(Path): Path object that stores filepath to batch csv

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


    def _run_batch_processor(self, import_directory: Path, batch_csv_filepath: Path,
                             RipVentBatchAnnotator_filepath: Path) -> None:
        """
        This method uses popen to run the batch processor .exe over files in the import_directory and the batch_csv_file

        Args:
            import_directory (Path): Path object that stores directory where raw patient-day files are stored
            batch_csv_filepath(Path): Path object that stores filepath to batch csv
            RipVentBatchAnnotator_filepath (Path): Path object that stores filepath to the batch processor .exe

        Returns:
            None:
        """
        # run batch annotator and add read to allow for it to finish before moving on
        command = str(RipVentBatchAnnotator_filepath) + ' ' + str(import_directory) + ' ' + str(import_directory) + ' ' + str(batch_csv_filepath)
        os.popen(command).read()


    def _delete_csv(self, batch_csv_filepath: Path) -> None:
        """
        This method deletes the batch csv that was generated as an input for the batch processor.
        After the batch processor is run this file is not needed, so we delete it

        Args:
            batch_csv_filepath (Path): Path object that stores filepath to batch csv

        Returns:
            None:
        """
        # if we created a csv file we will remove it
        batch_csv_filepath.unlink()

    def _move_files_to_export_dir(self, import_directory: Path, export_directory: Path) -> None:
        """
        This method moves all files generated by the batch processor from the import directory to the batch_outputs
        folder in the export directory

        Args:
            import_directory (Path): Path object that stores directory where raw patient-day files are stored
            export_directory (Path): Path object that stores directory to store outputs

        Returns:
            None:
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


    def _organize_export_dir(self, export_directory: Path) -> None:
        """
        This method organizes batch processed files in the export directory. It creates directories for every patient-day
        and places the batch processed filed into their corresponding directories

        Args:
            export_directory (Path): Path object that stores directory to store outputs

        Returns:
            None:
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

    def _batch_process(self, raw_files_directory: str, export_directory: str, batch_processor_filepath: str) -> Path:
        """
        This method runs the batch processing pipeline, creates batch csv, runs the batch processor, delete batch csv,
        move files to export directory, and organize files.


        Args:
            raw_files_directory (str): directory where raw patient-day files are stored
            export_directory (str): directory to store outputs
            batch_processor_filepath (str): filepath to the batch processor .exe

        Returns:
            export_directory (Path): Path object that stores directory to store outputs

        """
        print('Batch Processing Starting!')

        # # setup import and export directories
        import_directory, export_directory, RipVentBatchAnnotator_filepath = self._setup_directories(raw_files_directory, export_directory, batch_processor_filepath)

        # put methods together to run batch processor
        batch_csv_filepath = self._create_batch_csv(import_directory, export_directory)
        print('Creating Batch Files')
        self._run_batch_processor(import_directory, batch_csv_filepath, RipVentBatchAnnotator_filepath)
        print('Batch File Creation Finished')
        self._delete_csv(batch_csv_filepath)
        print('Moving Batch Files to Export dir')
        self._move_files_to_export_dir(import_directory, export_directory)
        print('Batch Files Moved')
        print('Organizing Batch Files')
        self._organize_export_dir(export_directory)
        print('Batch Files Organized')
        print('Batch Processing Done!')

        # return export directory for ease of use
        return export_directory

    def batch_process_and_validate(self) -> Path:
        """
        This method runs the batch processing pipeline then checks to see if batch processing worked for all patient-days
        if there are patient days that batch processing failed, rerun those patient-days until they are processed properly
        This happens because the batch processor .exe fails randomly sometimes

        Returns:
            export_directory (Path): Path object that stores directory to store outputs
        """
        # instantiate data cleaner
        data_cleaner = Data_Cleaner()

        # batch process files
        export_directory = self._batch_process(self.raw_files_directory, self.export_directory, self.batch_processor_filepath)

        ## check the batch files directory for errors

        # check for duplicate patient days
        data_cleaner.check_for_duplicate_pt_days(export_directory)

        # check for invalid directories
        num_invalid, invalid_dir = data_cleaner.check_for_invalid_subdirs(export_directory)

        # while the number of invalid files is greater than 0, run the batch processor over the invalid files to fix them
        while num_invalid > 0:

            # batch process files
            self._batch_process(invalid_dir, self.export_directory, self.batch_processor_filepath)

            # check the num invalid again
            num_invalid, invalid_dir = data_cleaner.check_for_invalid_subdirs(export_directory)

        # return export directory for other functions
        return export_directory


# if running this file directly, only do batch processing
if __name__ == "__main__":

    # Command Line Arguments
    p = argparse.ArgumentParser()
    p.add_argument('--raw_files_directory', type=str, default=None, help='Directory with raw unannotated files')
    p.add_argument('--export_directory', type=str, default='..\\datasets', help='Directory to export organized unannotated files for later processing')
    p.add_argument('--batch_annotator_filepath', type=str, default='..\\batch_annotator\RipVent.BatchProcessor.exe', help='Path to vent annotator')
    args = vars(p.parse_args())

    # define args
    input_directory = args['input_directory']
    export_directory = args['dataset_directory']
    batch_annotator_filepath = args['batch_annotator_filepath']

    # instantiate batch annotator class
    batch_annotator = Batch_Annotator(input_directory, export_directory, batch_annotator_filepath)

    # run batch processor
    export_directory = batch_annotator.batch_process_and_validate()

    print(f'Batch outputs generated at {export_directory}')
