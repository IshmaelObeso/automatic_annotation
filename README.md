# automatic_annotation
The Automatic Annotator is a program designed to use a CNN model trained on annotated datasets of ventilator dyssynchronies to automatically annotate unannotated
datasets of ventilator dyssynchronies.

USAGE:
-- EXE FILE

You can double click the Bennotator.exe file and run the program that way

-- SCRIPTS
There are two types of scripts in this program, 
        - batch process to triplets scripts produce only triplets and a statics file from patient-day files
        - batch process to auto annotation scripts produce annotated datasets from unannotated patient-day files
You can run these scripts one of two ways

  - GUI-Based (RECOMMENDED): If you want to run these scripts from the command line, but input arguments from a GUI, run the scripts like this
             ``` run gui/batch_process_to_auto_annotation ```
               Then use the Gui to define the paths to the import directory and dataset directories

  - Non-GUI-Based: If you want to run these scripts from the command line, and input arguments from the command line, run the scripts like this
             ``` run scripts/name_of_script_to_run --import directory PATH_TO_UNANNOTATED_FILES --dataset_directory PATH_TO_DIRECTORY_TO_STORE_OUTPUT_DATASETS ```

INSTALLING REQUIRED PACKAGES

-- Installation with Pip
    ``` pip install -r requirements.txt ```
    
-- Installation with Conda
    ``` conda env create -f environment.yml```


INSTALLING REQUIRED PACKAGES

-- Installation with Pip
    ``` pip install -r requirements.txt ```
    
-- Installation with Conda
    ``` conda env create -f environment.yml```