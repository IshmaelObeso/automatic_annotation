import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import re


def extract_breath_info(breath_filepath: object) -> object:
    """

    Args:
        breath_filepath:

    Returns:

    """
    patient_id_pattern = 'pt?(\d*)_'
    day_id_pattern = 'day(\d*)[^\d]'
    breath_id_pattern = 'breath_?(\d*)'


    patient = re.findall(patient_id_pattern, breath_filepath, flags=re.IGNORECASE)[0]
    day = re.findall(day_id_pattern, breath_filepath, flags=re.IGNORECASE)[0]
    breath = re.findall(breath_id_pattern, breath_filepath, flags=re.IGNORECASE)[0]

    return patient, day, breath


def get_inspiration_expiration_triggers(triplet: object, deliniation_column: object) -> object:
    """

    Args:
        triplet:
        deliniation_column:

    Returns:

    """
    # get index of inspiration and expiration triggers
    inspiration_triggers = triplet[triplet[deliniation_column] == 1].index
    expiration_triggers = triplet[triplet[deliniation_column] == -1].index
    all_triggers = triplet[(triplet[deliniation_column] == 1) | (triplet[deliniation_column] == -1)].index

    return inspiration_triggers, expiration_triggers, all_triggers


def left_and_right_current_inspiration_idxs(triplet: object, inspiration_triggers: object, inspiration_surround_s: object) -> object:
    """

    Args:
        triplet:
        inspiration_triggers:
        inspiration_surround_s:

    Returns:

    """
    # get index 100ms before second inspiration trigger
    timesteps_lessthan_or_equalto_100ms_before_inspiration = triplet[
        triplet['TimeRel'] <= triplet['TimeRel'].loc[inspiration_triggers[1]] - dt.timedelta(
            seconds=inspiration_surround_s)]
    current_inspiration_left_index = timesteps_lessthan_or_equalto_100ms_before_inspiration.index[-1]

    # get index 100ms after second inspiration trigger
    timesteps_greaterthan_or_equalto_100ms_after_inspiration = triplet[
        triplet['TimeRel'] >= triplet['TimeRel'].loc[inspiration_triggers[1]] + dt.timedelta(
            seconds=inspiration_surround_s)]
    current_inspiration_right_index = timesteps_greaterthan_or_equalto_100ms_after_inspiration.index[1]

    return current_inspiration_left_index, current_inspiration_right_index


def get_ymax(triplet: object, esophogeal_manometry_column: object, current_inspiration_left_index: object,
             current_inspiration_right_index: object) -> object:
    """

    Args:
        triplet:
        esophogeal_manometry_column:
        current_inspiration_left_index:
        current_inspiration_right_index:

    Returns:

    """
    # find ymax in the range 100ms around second inspiration trigger
    current_inspiration_ymax_range = triplet.loc[current_inspiration_left_index:current_inspiration_right_index]
    current_inspiration_ymax = current_inspiration_ymax_range[esophogeal_manometry_column].max()
    current_inspiration_ymax_idx = current_inspiration_ymax_range[esophogeal_manometry_column].idxmax()

    return current_inspiration_ymax, current_inspiration_ymax_idx


def left_and_right_current_expiration_idxs(triplet: object, inspiration_triggers: object, expiration_triggers: object, ymax_idx: object,
                                           expiration_surround_s: object) -> object:
    """

    Args:
        triplet:
        inspiration_triggers:
        expiration_triggers:
        ymax_idx:
        expiration_surround_s:

    Returns:

    """
    # get index 100ms before second expiration trigger
    timesteps_lessthan_or_equalto_100ms_before_expiration = triplet[
        triplet['TimeRel'] <= triplet['TimeRel'].loc[expiration_triggers[1]] - dt.timedelta(
            seconds=expiration_surround_s)]

    # the leftmost expiration index should never be before the deltaPes baseline
    expiration_left_index = inspiration_triggers[1]

    current_expiration_left_index = pd.Index([expiration_left_index, ymax_idx]).max()

    # get index 100ms after second inspiration trigger
    timesteps_greaterthan_or_equalto_100ms_after_expiration = triplet[
        triplet['TimeRel'] >= triplet['TimeRel'].loc[expiration_triggers[1]] + dt.timedelta(
            seconds=expiration_surround_s)]
    try:
        # if rightmost expiration index falls within triplet, grab index
        current_expiration_right_index = timesteps_greaterthan_or_equalto_100ms_after_expiration.index[1]
    except:
        # if rightmost expiration index falls outside triplet, grab last index in triplet
        current_expiration_right_index = triplet.index[-1]

    return current_expiration_left_index, current_expiration_right_index


def get_ymin(triplet: object, esophogeal_manometry_column: object, current_expiration_left_index: object,
             current_expiration_right_index: object) -> object:
    """

    Args:
        triplet:
        esophogeal_manometry_column:
        current_expiration_left_index:
        current_expiration_right_index:

    Returns:

    """
    # find ymin in the range from current inspiration trigger to 100ms after current expiration trigger
    current_inspiration_ymax_range = triplet.loc[current_expiration_left_index:current_expiration_right_index][
        esophogeal_manometry_column]
    current_expiration_ymin = current_inspiration_ymax_range.min()
    current_expiration_ymin_idx = current_inspiration_ymax_range.idxmin()

    return current_expiration_ymin, current_expiration_ymin_idx


def calculate_deltaPes(triplet: object, inspiration_surround_s: object = .1, expiration_surround_s: object = .1) -> object:
    """

    Args:
        breath_filepath:
        inspiration_surround_s:
        expiration_surround_s:

    Returns:

    """

    # set some variables for ease
    deliniation_column = 't:Fsp'
    esophogeal_manometry_column = 's:Pes'

    # convert timerel column to datetime for calculations
    triplet['TimeRel'] = pd.to_datetime(triplet['TimeRel'])

    # get index of inspiration and expiration triggers
    inspiration_triggers, expiration_triggers, all_triggers = get_inspiration_expiration_triggers(triplet,
                                                                                                  deliniation_column)

    # look at a region 100ms around the second inspiration trigger (the current breath inspiration trigger)
    current_inspiration_left_index, current_inspiration_right_index = left_and_right_current_inspiration_idxs(triplet,
                                                                                                              inspiration_triggers,
                                                                                                              inspiration_surround_s)

    # find ymax in the range 100ms around second inspiration trigger
    current_inspiration_ymax, current_inspiration_ymax_idx = get_ymax(triplet, esophogeal_manometry_column,
                                                                      current_inspiration_left_index,
                                                                      current_inspiration_right_index)

    # look at a region 100ms around the second expiration trigger (the current breath expiration trigger)
    current_expiration_left_index, current_expiration_right_index = left_and_right_current_expiration_idxs(triplet,
                                                                                                           inspiration_triggers,
                                                                                                           expiration_triggers,
                                                                                                           current_inspiration_ymax_idx,
                                                                                                           expiration_surround_s)

    # find ymin in the range from current inspiration trigger to 100ms after current expiration trigger
    current_expiration_ymin, current_expiration_ymin_idx = get_ymin(triplet, esophogeal_manometry_column,
                                                                    current_expiration_left_index,
                                                                    current_expiration_right_index)

    # calculate deltaPes
    deltaPes = current_inspiration_ymax - current_expiration_ymin

    return deltaPes


