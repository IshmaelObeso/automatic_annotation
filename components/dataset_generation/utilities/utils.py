### Utilities for the dyssynchrony project

import numpy as np
import pandas as pd
import re
import multiprocessing as mp

dyssynchrony = {104: 'Autotrigger',
                106: 'Ineffective Trigger',
                107: 'Double Trigger',
                108: 'Trigger Delay',
                109: 'Flow Undershoot',
                110: 'Delayed Termination',
                111: 'Premature Termination',
                112: 'Expiratory Asynchrony',
                113: 'Other Asynchrony',
                114: 'Reverse Trigger',
                115: 'Flow Overshoot',
                1: 'Other Asynchrony'}

ARTIFACTS = {101: 'Disconnect',
             102: 'Agitation',
             103: 'Secretions',
             105: 'PS Cutoff Bump',
             201: 'Manometry disconnect',
             202: 'Manometry agitation',
             203: 'Manometry peristalsis',
             204: 'Manometry cough',
             205: 'Manometry cry',
             206: 'Manometry trigger inadequate',
             207: 'Manometry cardiac artifact induced trigger',
             208: 'Manometry cardiac artifact lack of trigger',
             209: 'Manometry double trigger',
             210: 'Manometry passive'}

# There are certain patient days with multiple annotation files
# Ben has picked out the ones that we should use, so I'm listing the exclusions here
# And will filter them in create_triplets.py
# !! WARNING !! These files are manually listed and need to be updated with new data pulls
DUPLICATE_DIRECTORIES_TO_IGNORE = ['Pt230_Day4_AsynchronyCont',
                                   'Pt233_Day3_Asynchrony',
                                   '20190804_135150.TPP-NIF.Pt273_Day9_Asynchronycsv',
                                   'Pt299_Day1-_Asynchrony',
                                   'Pt370_Day12_Asynchrony_SIMV_RATE26',
                                   'P370_Day18_asynchronyPS20_10',
                                   'Pt372_Day0_asynchrony2',
                                   'pt380_day2_asynchrony2_lowerrate']

# TODO: Move inline comments to line above
DELINEATION_COLUMN = 't:Fsp'  # Marks breaths {1: Inspriation, -1: Expiration}
LABEL_CODE_COLUMNS = ['a:Fsp', 'a:Paw', 'a:Pes']  # These hold the dyssynchrony and artifact codes
TRAINING_COLUMNS = ['Time', 'TimeRel', 's:Fsp', 's:Paw', 's:Pes']  # Columns to retain for the training data
WAVEFORM_COLUMNS = ['s:Fsp', 's:Paw', 's:Pes']

# We want certain truth columns to be combined into a single truth column
# For example, double trigger and autotrigger
# The keys of these dictionaries will be the resulting column name and the values will be
# lists that contain the two column names that will be combined

# !! WARNING !! - The OR operations MUST BE RUN before the AND operations OR ELSE DOOOOM.
# AKA Inadequate Support will not be created and not be available in the AND operation

# The first dictionary should have its values combined by boolean 'or'
# Inadequate Support is a dyssynchrony type that Ben named because
# Premature Termination and Flow Undershoot exist on the same continuous spectrum
COOCCURRENCES_TO_CREATE_VIA_OR = {'Inadequate Support': ['Premature Termination', 'Flow Undershoot'],
                                  'General Inadequate Support': ['Premature Termination', 'Flow Undershoot',
                                                                 'Ineffective Trigger', 'Trigger Delay']}

# The second dictionary should have its values combined by boolean 'and'
COOCCURRENCES_TO_CREATE_VIA_AND = {'Double Trigger Autotrigger': ['Double Trigger', 'Autotrigger'],
                                   'Double Trigger Reverse Trigger': ['Double Trigger', 'Reverse Trigger'],
                                   'Double Trigger Premature Termination': ['Double Trigger', 'Premature Termination'],
                                   'Double Trigger Flow Undershoot': ['Double Trigger', 'Flow Undershoot'],
                                   'Double Trigger Inadequate Support': ['Double Trigger', 'Inadequate Support']}

# constants for triplet generation
TA_CSV_SUFFIX = 'TriggersAndArtifacts.csv'
NUM_BREATHS = 3
TRIPLET_FILE_ID_COLUMNS = ['triplet_id', 'breath_id']
KEEP_COLUMNS = TRIPLET_FILE_ID_COLUMNS + TRAINING_COLUMNS + [DELINEATION_COLUMN] + list(
    set(dyssynchrony.values())) + list(set(COOCCURRENCES_TO_CREATE_VIA_OR.keys())) + list(
    set(COOCCURRENCES_TO_CREATE_VIA_AND.keys())) + ['No Double Trigger', 'No Inadequate Support']

MERGE_COLUMNS = ['original_subdirectory', 'breath_id']

ERROR_DIRS = ['duplicates', 'invalid']

# constants for spectral triplet generation
# Sampling frequency (this is from the ventilators themselves, sampling 200 times per second)
FS = 200
# The window to use in the Fourier transform
WINDOW = ('tukey', 0.1)
# The size of the window
NPERSEG = 64
# Define the stride or step size
STRIDE = 2
# Defining the overlap between consecutive windows as the window size minus the step size
NOVERLAP = NPERSEG - STRIDE

# Other parameters I will admit I'm not as familiar with but I'll research more if they become more relevant
# (these are the scipy defaults, just trying to be explicit)

# This has to do with 0 padding which I'm not too worried about with our small step size
NFFT = None
DETREND = 'constant'
RETURN_ONESIDED = True
SCALING = 'density'

# The number of frequency bins to extract from the spectrogram
FREQ_BINS = 16

# We want both the power spectrum density and angle Fourier transforms
MODES = ['psd', 'angle']


def get_context_blackslist(blacklist: list, context_breaths: list) -> list:
    """
    Given a blacklist, return the surrounding breaths that should also be blacklisted

    Args:
        blacklist (list): list of breaths to not include in analysis
        context_breaths (list): list of breaths surrounding blacklisted breaths that should also not be included

    Returns:
        list: return context_breaths list of breaths surrounding blacklisted breaths that should also not be included

    """

    context_blacklist = []
    for blacklisted_id in blacklist:
        for i in range(1, context_breaths + 1):
            context_blacklist += [blacklisted_id + i,
                                  blacklisted_id - i]

    return context_blacklist


def get_artifact_blacklist(patient_day: pd.DataFrame, num_breaths: int = 3, breath_id_col: str = 'breath_id') -> list:
    """
    Identify the breaths with artifacts and return the breath_ids.

    Args:
        patient_day (DataFrame): A patient's day worth of recorded breaths
        num_breaths (int): The number of breaths to include as context, including the breath itself (we're doing triplets now, so default is 3)
        breath_id_col (str): The column name containing breath ids
    Returns:
        list: Returns a list that includes all breaths that have artifacts to exclude

    """

    # TODO: Use statics as an input instead of patient_day
    # First figure out how many breaths on either side we need to include as context
    context_breaths = int(np.floor(num_breaths / 2))

    # Initialize a blacklist for ids that contain artifacts
    artifact_blacklist = []

    # For each column that may contain an artifact code, identify unique breath ids and append them to the blacklist
    for label_code_column in LABEL_CODE_COLUMNS:
        artifact_blacklist += patient_day[patient_day[label_code_column].isin(ARTIFACTS)][
            breath_id_col].unique().tolist()

    # Clean up duplicates if the same code existed in multiple columns
    artifact_blacklist = np.unique(artifact_blacklist).tolist()

    # Mark the context breaths as blacklisted
    context_blacklist = get_context_blackslist(artifact_blacklist, context_breaths)

    return np.unique(artifact_blacklist + context_blacklist).tolist()


def get_breath_length_blacklist(patient_day: pd.DataFrame, num_breaths: int = 3, min_length: float = 0.25, max_length: float = 10.0,
                                time_col: str = 'TimeRel',
                                breath_id_col: str = 'breath_id') -> np.ndarray:
    '''
    Identify breaths that are either too long or too short to be considered.

    Args:
        patient_day (DataFrame): A patient's day worth of recorded breaths
        num_breaths (int): The number of breaths to include as context, including the breath itself (we're doing triplets now, so default is 3)
        min_length (float): Minimum length of a breath in seconds
        max_length (float): Maximum length of a breath in seconds
        time_col (str): Column name that contains datetime information in patient_day
    Returns:
        np.ndarray: Returns array of breaths that are either too long or too short to be included in dataset
    '''

    # TODO: Feed this function statics instead of patient_day
    context_breaths = int(np.floor(num_breaths / 2))

    min_length = pd.Timedelta(seconds=min_length)
    max_length = pd.Timedelta(seconds=max_length)

    breath_lengths = patient_day.groupby(breath_id_col).apply(
        lambda breath: breath[time_col].iloc[-1] - breath[time_col].iloc[0])

    breaths_too_short = breath_lengths[breath_lengths < min_length].index.values.tolist()
    breaths_too_long = breath_lengths[breath_lengths > max_length].index.values.tolist()

    too_short_context_blacklist = get_context_blackslist(breaths_too_short, context_breaths)
    too_long_context_blacklist = get_context_blackslist(breaths_too_long, context_breaths)

    return np.unique(breaths_too_short +
                     breaths_too_long +
                     too_short_context_blacklist +
                     too_long_context_blacklist).tolist()


def one_hot_dyssynchronies(patient_day: pd.DataFrame, breath_id_col: str = 'breath_id', dyssynchrony_mask_col: str = 'dyssynchrony_mask',
                           min_breath_fraction: float = 0.75) -> pd.DataFrame:
    """
    Given the list of dyssynchrony codes, create a column for each and
    assign the entire breath 1 if the disynchrony's present and 0 if not.
    NOTE: It is assumed that patient_day_masked contains ONLY rows that should
    be searched for dyssynchrony codes (in the current implementation, this
    means rows between inspiration and expiration.)

    Args:
        patient_day (pd.DataFrame): DataFrame of patient-day
        breath_id_col (str): column that stores information about breath ids
        dyssynchrony_mask_col (str): colun that stores information about dyssynchrony masks
        min_breath_fraction (float): The minimum fraction of the time between a breath's
                             inspiration and expiration that must contain the dyssynchrony
                             code to qualify as that dyssynchrony type

    Returns:
        pd.DataFrame: Returns dataframe with dyssynchronies one-hot encoded

    """
    # We're only looking for dyssynchronies between inspiration and expiration, so we'll mask the df to start with
    patient_day_masked = patient_day[patient_day[dyssynchrony_mask_col] == 1]

    # Initialize a DataFrame for each of the columns that contain dyssynchrony codes
    one_hot_df_dict = {}
    for label_code_column in LABEL_CODE_COLUMNS:
        one_hot_df = pd.DataFrame(index=patient_day[breath_id_col].unique())

        for dissync_code in dyssynchrony.keys():
            ## TODO: This lambda function is what needs to be updated when you talk with Dave about the exact
            ## paramters that constitute a dyssynchrony (Full breath (current implementation)? From inhale to exhale? Percentage based?)
            # full_breath_contains_code = lambda breath: (breath[label_code_column].min() == dissync_code) and (breath[label_code_column].max() == dissync_code)
            # breath_contains_code = lambda breath: (breath[label_code_column] == dissync_code).max()

            # Create a column indicating whether or not the dissync is present
            dissync_present_df = patient_day_masked[[breath_id_col]].copy()
            dissync_present_df['dissync_present'] = (patient_day_masked[label_code_column] == dissync_code)
            dissync_present = dissync_present_df.groupby(breath_id_col)['dissync_present']
            # dissync_present = dissync_present.groupby(breath_id_col)

            # TODO: Change to .mean()
            one_hot_df[dissync_code] = ((dissync_present.sum() / dissync_present.size()) > min_breath_fraction) * 1

            # Compute the fraction of breath labeled with this dyssynchrony code and compare it with our min_breath_fraction
            # breath_contains_code = lambda breath: ((breath[label_code_column] == dissync_code).sum() / breath.shape[0]) > min_breath_fraction
            # Create a column with the same length as patient_day that contains a binary indicator of each dyssynchrony code
            # one_hot_df[dissync_code] = patient_day_masked.groupby(breath_id_col).apply(breath_contains_code) * 1

        # SPECIAL CASE: There are two codes for "Other Asynchrony" ([1, 113]) that must be turned into a single column
        one_hot_df[113] = one_hot_df[[1, 113]].max(axis=1)
        one_hot_df = one_hot_df.drop(columns=[1])

        one_hot_df = one_hot_df.rename(columns=dyssynchrony)

        one_hot_df_dict[label_code_column] = one_hot_df

    # TODO: Instead of dict, create a multi-index and do the below .max() natively in pandas across the proper axis
    full_one_hot_df = pd.DataFrame(data=np.stack([df for df in one_hot_df_dict.values()]).max(axis=0),
                                   columns=one_hot_df_dict[label_code_column].columns)

    # We'll create a column that's the inverse of "Double Trigger" called "No Double Trigger" to act as our final
    # multiclass label for the multiclass classification problem (so all rows have exactly one "1")
    # TODO: Speed comparison of the below line vs. 1 - full_one_hot_df['Double Trigger']
    full_one_hot_df['No Double Trigger'] = full_one_hot_df['Double Trigger'].map({0: 1, 1: 0})

    # !! WARNING !! - The OR operations MUST BE RUN before the AND operations OR ELSE DOOOOM.
    # AKA Inadequate Support will not be created and not be available in the AND operation

    # Combine columns like Double Trigger and Autotrigger into a single column for training
    # Start with the ones to combine using an 'or' operation
    for cooccurrence_col, combine_col_list in COOCCURRENCES_TO_CREATE_VIA_OR.items():
        full_one_hot_df = create_cooccurrence_column_via_or(full_one_hot_df,
                                                            cooccurrence_col,
                                                            combine_col_list)

    # Now create the columns combined via 'and'
    for cooccurrence_col, combine_col_list in COOCCURRENCES_TO_CREATE_VIA_AND.items():
        full_one_hot_df = create_cooccurrence_column_via_and(full_one_hot_df,
                                                             cooccurrence_col,
                                                             combine_col_list)

    # Since we'll train a multilabel model to predict the two underlying types of
    # Inadequate Support separately (as Ineffective Trigger and Flow Undershoot themselves)
    # we'll create an H0 column for that truthing scheme, which we can derive from the newly
    # created Inadequate Support column
    full_one_hot_df['No Inadequate Support'] = full_one_hot_df['Inadequate Support'].map({0: 1, 1: 0})

    return full_one_hot_df


def one_hot_artifacts(patient_day: pd.DataFrame, breath_id_col: str = 'breath_id') -> pd.DataFrame:
    """
    Given the list of artifact codes, create a column for each and
    assign the entire breath 1 if the artifact's present and 0 if not.

    Args:
        patient_day (pd.DataFrame): DataFrame of patient-day
        breath_id_col (str): column that stores information about breath ids

    Returns:
        pd.DataFrame: Returns DataFrame with artifacts one-hot encoded

    """
    # Initialize a DataFrame for each of the columns that contain dyssynchrony codes
    one_hot_df_dict = {}
    for label_code_column in LABEL_CODE_COLUMNS:
        one_hot_df = pd.DataFrame(index=patient_day[breath_id_col].unique())

        for artifact_code in ARTIFACTS.keys():
            # If any row in the breath contains the artifact code, the entire breath is labeled an artifact
            breath_contains_code = lambda breath: (breath[label_code_column] == artifact_code).max()

            # Create a column with the same length as patient_day that contains a binary indicator of each artifact code
            one_hot_df[artifact_code] = patient_day.groupby(breath_id_col).apply(breath_contains_code) * 1

        one_hot_df_dict[label_code_column] = one_hot_df

    full_one_hot_df = pd.DataFrame(data=np.stack([df for df in one_hot_df_dict.values()]).max(axis=0),
                                   columns=ARTIFACTS.values())

    return full_one_hot_df


def create_breath_statics(patient_day: pd.DataFrame, num_breaths: int = 3, min_length: float = 0.25, max_length: float = 10.0, time_col: str = 'TimeRel',
                          breath_id_col: str = 'breath_id') -> pd.DataFrame:
    """
    Creates a statics file for each breath that contains the following columns:
        - Breath ID
        - Start time
        - End time
        - Breath length
        - A binary column for each dyssynchrony/artifact type

    Args:
        patient_day (pd.DataFrame): DataFrame of patient-day
        num_breaths (int): number of breaths to include, triplet (3) is the default
        min_length (float): minimum length of breath to include
        max_length (float): maximim length of breath to include
        time_col (str): string that contains information about which time column to use
        breath_id_col (str): string that contains information about which column contains breath ids

    Returns:
        pd.DataFrame: Returns dataframe that contains statics about triplets in patient-day

    """
    # Initialize an empty statics DataFrame
    statics = pd.DataFrame(index=patient_day[breath_id_col].unique())

    # Initialize a one hot encoded DataFrame for the artifacts (will fill it in using the artifact blacklist)
    artifact_one_hot = pd.DataFrame(columns=ARTIFACTS.values(), index=patient_day[breath_id_col].unique()).fillna(0)

    # Grab the first and last values of the time column as the start and end times for each breath
    statics['start_time'] = patient_day.groupby(breath_id_col).take([0])[time_col].droplevel(1)
    statics['expiration_time'] = \
    patient_day[patient_day[DELINEATION_COLUMN] == -1].reset_index().set_index(breath_id_col)[time_col]
    statics['end_time'] = patient_day.groupby(breath_id_col).take([-1])[time_col].droplevel(1)

    # Calculate the length of each breath
    statics['length'] = statics['end_time'] - statics['start_time']
    statics['inspiration_length'] = statics['expiration_time'] - statics['start_time']
    statics['expiration_length'] = statics['end_time'] - statics['expiration_time']

    # Add in the one hot encoded dyssynchronies
    statics = pd.concat([statics, one_hot_dyssynchronies(patient_day, breath_id_col=breath_id_col,
                                                         dyssynchrony_mask_col='dyssynchrony_mask')], axis=1)

    # Fill in 1s for the breaths that contain artifacts
    statics = pd.concat([statics, one_hot_artifacts(patient_day, breath_id_col=breath_id_col)], axis=1)

    statics.index.name = breath_id_col
    statics = statics.reset_index()

    return statics


def get_patient_id(filename: str) -> str:
    """
    Extract the patient ID from a filename using regex.
    Args:
        filename (str): filename

    Returns:
        str: Returns string of patient_id

    """

    # Using this pattern scares me a bit, because it basically will pick up any
    # number of digits following a "p" or "pt" and followed by an underscore.
    # It works for all the current filenames, and the optional "t" is required because
    # some filenames only have "P###_" instead of "Pt###_", but this should be considered when
    # new files come in.
    patient_id_pattern = 'pt?(\d*)_'

    return re.findall(patient_id_pattern, filename, flags=re.IGNORECASE)[0]


def get_day_id(filename: str) -> str:
    """
    Extract the day ID from a filename using regex.
    Args:
        filename (str): filename

    Returns:
        str: Returns string of day_id

    """

    # Again, a couple of weird cases (e.g. Pt219_Day2d_Asynchrony) force me to use
    # the "anything but a number" regex at the end of day_id_pattern.
    # This works for all current cases, but should be looked at carefully when more data
    # flows in (hopefully all filenames will be uniformized once this project takes off)
    day_id_pattern = 'day(\d*)'

    return re.findall(day_id_pattern, filename, flags=re.IGNORECASE)[0]

def find_h0s_with_adjacent_h1(statics: pd.DataFrame, truth_col: str, num_breaths: int = 3, patient_id_col: str = 'patient_id',
                              day_id_col: str = 'day_id') -> pd.DataFrame:
    """
    Finds h0 with h1's in adjacent breaths

    Args:
        statics (pd.DataFrame): DataFrame with summary information about patient-day
        truth_col (str): column with truth values
        num_breaths (int): number of breaths (default=3) i.e. triplet, to look around the h0
        patient_id_col (str): column wtih patient_ids
        day_id_col (str): column with day_ids

    Returns:
        pd.DataFrame: returns column with information on whether a given breath is  h0 with h1 in the surrounding breaths

    """
    # A lambda function to apply to the truth column that finds if current breath (middle breath)
    # is an h0 and either the previous or subsequent breath are h1
    is_h0_with_adjacent_h1 = lambda truth: 1 if truth[1] == 0 and (truth[0] == 1 or truth[2] == 1) else 0

    # rolling function that looks at each breath in sequence, if the breath is a h0 (0), and either of the surrounding breaths
    # are an h1 (1), then label that row as 1 (h0 with h1 adjacent)
    return statics.groupby(level=[patient_id_col, day_id_col])[truth_col].fillna(0).rolling(num_breaths,
                                                                                            center=True).apply(
        is_h0_with_adjacent_h1).fillna(0)


def create_cooccurrence_column_via_or(statics: pd.DataFrame, cooccurrence_col: str, combine_col_list: list) -> pd.DataFrame:
    '''
    Combines the two columns listed in combine_col_list into a new statics column
    named cooccurrence_col via boolean 'or'

    Args:
        statics (pd.DataFrame): The statics DataFrame to which a new column will be added
        cooccurrence_col (str): The name of the newly added column
        combine_col_list (list[str]): A list containing the two existing statics columns
                                      to be combined via boolean 'or'

    Returns:
        statics (pd.DataFrame): The statics DataFrame with a new column
    '''

    # We'll take the max, which in this case acts as an
    # 'or' across the entire row of combine_col_list
    statics[cooccurrence_col] = statics[combine_col_list].max(axis=1).fillna(0)

    return statics


def create_cooccurrence_column_via_and(statics: pd.DataFrame, cooccurrence_col: str, combine_col_list: list) -> pd.DataFrame:
    '''
    Combines the two columns listed in combine_col_list into a new statics column
    named cooccurrence_col via boolean 'and'

    Args:
        statics (pd.DataFrame): The statics DataFrame to which a new column will be added
        cooccurrence_col (str): The name of the newly added column
        combine_col_list (list[str]): A list containing the two existing statics columns
                                      to be combined via boolean 'and'

    Returns:
        statics (pd.DataFrame): The statics DataFrame with a new column
    '''

    # TODO: Generalize for n columns in combine_col_list (sum() == num elements in list)
    statics[cooccurrence_col] = ((statics[combine_col_list[0]].fillna(0).astype(bool)) &
                                 (statics[combine_col_list[1]].fillna(0).astype(bool))).astype(int)

    return statics

def num_workers(multiprocessing: bool = False) -> int:
    """
    Calculates how many processes to assign based on whether we should multiprocess or not

    Args:
        multiprocessing: bool that determines if we should multiprocess or not

    Returns:
        int: if multiprocessing is False, always return 1, if multiprocessing is True, return number_of_cores-1

    """

    if not multiprocessing:
        return 1
    elif mp.cpu_count() == 1:
        return 1
    else:
        return mp.cpu_count()-1

