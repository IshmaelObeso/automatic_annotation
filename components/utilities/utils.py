### Utilities for the dyssynchrony project

import numpy as np
import pandas as pd
import re

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


def _get_context_blackslist(blacklist, context_breaths):
    '''
    Given a blacklist, return the surrounding breaths that should also be blacklisted
    '''

    context_blacklist = []
    for blacklisted_id in blacklist:
        for i in range(1, context_breaths + 1):
            context_blacklist += [blacklisted_id + i,
                                  blacklisted_id - i]

    return context_blacklist


def get_artifact_blacklist(patient_day, num_breaths=3, breath_id_col='breath_id'):
    '''
    Identify the breaths with artifacts and return the breath_ids.

    params:
    patient_day (DataFrame): A patient's day worth of recorded breaths
    num_breaths (int): The number of breaths to include as context, including the breath itself (we're doing triplets now, so default is 3)
    breath_id_col (str): The column name containing breath ids
    '''
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
    context_blacklist = _get_context_blackslist(artifact_blacklist, context_breaths)

    return np.unique(artifact_blacklist + context_blacklist).tolist()


def get_breath_length_blacklist(patient_day, num_breaths=3, min_length=0.25, max_length=10.0, time_col='TimeRel',
                                breath_id_col='breath_id'):
    '''
    Identify breaths that are either too long or too short to be considered.

    params:
        patient_day (DataFrame): A patient's day worth of recorded breaths
        num_breaths (int): The number of breaths to include as context, including the breath itself (we're doing triplets now, so default is 3)
        min_length (float): Minimum length of a breath in seconds
        max_length (float): Maximum length of a breath in seconds
        time_col (str): Column name that contains datetime information in patient_day
    '''
    # TODO: Feed this function statics instead of patient_day
    context_breaths = int(np.floor(num_breaths / 2))

    min_length = pd.Timedelta(seconds=min_length)
    max_length = pd.Timedelta(seconds=max_length)

    breath_lengths = patient_day.groupby(breath_id_col).apply(
        lambda breath: breath[time_col].iloc[-1] - breath[time_col].iloc[0])

    breaths_too_short = breath_lengths[breath_lengths < min_length].index.values.tolist()
    breaths_too_long = breath_lengths[breath_lengths > max_length].index.values.tolist()

    too_short_context_blacklist = _get_context_blackslist(breaths_too_short, context_breaths)
    too_long_context_blacklist = _get_context_blackslist(breaths_too_long, context_breaths)

    return np.unique(breaths_too_short +
                     breaths_too_long +
                     too_short_context_blacklist +
                     too_long_context_blacklist).tolist()


def one_hot_dyssynchronies(patient_day, breath_id_col='breath_id', dyssynchrony_mask_col='dyssynchrony_mask',
                           min_breath_fraction=0.75):
    '''
    Given the list of dyssynchrony codes, create a column for each and
    assign the entire breath 1 if the disynchrony's present and 0 if not.
    NOTE: It is assumed that patient_day_masked contains ONLY rows that should
    be searched for dyssynchrony codes (in the current implementation, this
    means rows between inspiration and expiration.)

    Args:
        min_breath_fraction (float): The minimum fraction of the time between a breath's
                                     inspiration and expiration that must contain the dyssynchrony
                                     code to qualify as that dyssynchrony type
    '''
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


def one_hot_artifacts(patient_day, breath_id_col='breath_id'):
    '''
    Given the list of artifact codes, create a column for each and
    assign the entire breath 1 if the artifact's present and 0 if not.
    '''

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


def create_breath_statics(patient_day, num_breaths=3, min_length=0.25, max_length=10.0, time_col='TimeRel',
                          breath_id_col='breath_id'):
    '''
    Create a statics file for each breath that contains the following columns:
        - Breath ID
        - Start time
        - End time
        - Breath length
        - A binary column for each dyssynchrony/artifact type
    '''

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


def get_patient_id(filename):
    '''
    Extract the patient ID from a filename using regex.
    '''

    # Using this pattern scares me a little bit, because it basically will pick up any
    # number of digits following a "p" or "pt" and followed by an underscore.
    # It works for all of the current filenames, and the optional "t" is required because
    # some filenames only have "P###_" instead of "Pt###_", but this should be considered when
    # new files come in.
    patient_id_pattern = 'pt?(\d*)_'

    return re.findall(patient_id_pattern, filename, flags=re.IGNORECASE)[0]


def get_day_id(filename):
    '''
    Extract the day ID from a filename using regex.
    '''

    # Again, a couple of weird cases (e.g. Pt219_Day2d_Asynchrony) force me to use
    # the "anything but a number" regex at the end of day_id_pattern.
    # This works for all current cases, but should be looked at carefully when more data
    # flows in (hopefully all filenames will be uniformized once this project takes off)
    day_id_pattern = 'day(\d*)[^\d]'

    return re.findall(day_id_pattern, filename, flags=re.IGNORECASE)[0]


def create_patient_day_verification(annotation_progress, pseudo_test=False):
    '''
    Create a dataframe that indicates which patient day directories have been reviewed
    by both Ben and Tatsu using the annotation progress Excel document Ben provided.

    Args:
        annotation_progress (pd.DataFrame): Excel sheet provided by Ben in DataFrame format
        pseudo_test (bool): If False, produce data based only on the '1file:pt' column.
                            If True, use data that is does NOT have '1file:pt' but is
                            validated by both Tatsu and Ben in the 'Reviewed By:' column.

    Returns:
        patient_day_verification (pd.DataFrame): The patient_id, day_id and a binary indicator of
        whether or not both Ben and Tatsu have reviewed the file
    '''
    # Initialize empty DataFrame
    patient_day_verification = pd.DataFrame()

    # !! WARNING !! This assumes a lot about the file Ben provided
    # We anticipate the format of this file to remain constant, but be weary any time data is pulled
    # that these column names may change

    # Extract the patient_id from the 'File' column
    patient_day_verification['patient_id'] = annotation_progress['File'].apply(
        lambda filename: get_patient_id(filename))

    # Extract the day_id from the 'File' column (hacky fix: use '_' along with get_day_id because these are structured
    # slightly differently than the directory names
    patient_day_verification['day_id'] = annotation_progress['File'].apply(lambda filename: get_day_id(filename + '_'))

    # If the 'Reviewed by:' contains some form of 'Ben, Tatsu', 'Tatsu, Ben', etc., it has been double reviewed and is
    # safe to use in model building
    patient_day_verification['double_reviewed_orig'] = (annotation_progress['Reviewed by:'].str.lower().str.contains(
        'ben') &
                                                        annotation_progress['Reviewed by:'].str.lower().str.contains(
                                                            'tatsu')) * 1

    # The above double_reviewed column is deprecated.
    # Ben let us know that we should be using 1file:pt as a column to indicate whether or not
    # a patient day is ready for use in model building.
    patient_day_verification['double_reviewed'] = annotation_progress['1file:pt'].fillna(0).astype(int)

    # If we're creating a pseudo test set we'll find which files are not included in our train/test/val
    # split but have been validated on the first round by both Ben and Tatsu
    if pseudo_test:
        patient_day_verification['double_reviewed'] = ((patient_day_verification['double_reviewed'] == 0) &
                                                       (patient_day_verification['double_reviewed_orig'] == 1)) * 1

    patient_day_verification = patient_day_verification.set_index(['patient_id', 'day_id']).sort_index()

    return patient_day_verification


def find_h0s_with_adjacent_h1(statics, truth_col, num_breaths=3, patient_id_col='patient_id', day_id_col='day_id'):
    '''
    ## TODO: Document
    '''
    # A lambda function to apply to the truth column that finds if current breath (middle breath)
    # is an h0 and either the previous or subsequent breath are h1
    is_h0_with_adjacent_h1 = lambda truth: 1 if truth[1] == 0 and (truth[0] == 1 or truth[2] == 1) else 0

    # TODO: Document
    return statics.groupby(level=[patient_id_col, day_id_col])[truth_col].fillna(0).rolling(num_breaths,
                                                                                            center=True).apply(
        is_h0_with_adjacent_h1).fillna(0)


def create_cooccurrence_column_via_or(statics, cooccurrence_col, combine_col_list):
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

    # DEPRECATED: We now have more than 2 columns to combine for generalized_inadequate_support
    # ~ statics[cooccurrence_col] = ((statics[combine_col_list[0]].fillna(0).astype(bool)) |
    # ~ (statics[combine_col_list[1]].fillna(0).astype(bool))).astype(int)

    # We'll take the max, which in this case acts as an
    # 'or' across the entire row of combine_col_list
    statics[cooccurrence_col] = statics[combine_col_list].max(axis=1).fillna(0)

    return statics


def create_cooccurrence_column_via_and(statics, cooccurrence_col, combine_col_list):
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
