import pandas as pd
import datetime as dt


def get_inspiration_expiration_triggers(triplet: pd.DataFrame, deliniation_column: str) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Gets the inspiration and expiration triggers of a triplet

    Args:
        triplet (pd.DataFrame): DataFrame of a triplet
        deliniation_column (str): column that contains information on triggers

    Returns:
        inspiration_triggers (pd.Index): Indexes of inspiration triggers
        expiration_triggers (pd.Index) Indexes of expiration triggers
        all_triggers (pd.Index) Indexes of all triggers
    """

    # get index of inspiration and expiration triggers
    inspiration_triggers = triplet[triplet[deliniation_column] == 1].index
    expiration_triggers = triplet[triplet[deliniation_column] == -1].index
    all_triggers = triplet[(triplet[deliniation_column] == 1) | (triplet[deliniation_column] == -1)].index

    return inspiration_triggers, expiration_triggers, all_triggers


def left_and_right_current_inspiration_idxs(triplet: pd.DataFrame, inspiration_triggers: pd.Index, inspiration_surround_s: float) -> tuple[pd.Index, pd.Index]:
    """
    Get indexes of timesteps corresponding to +-inspiration_surround from the current (second) breath inspiration trigger

    Args:
        triplet (pd.DataFrame): DataFrame of a triplet
        inspiration_triggers (pd.Index): Indexes of triplet inspiration triggers
        inspiration_surround_s (float): time in seconds around the inspiration trigger to get index of

    Returns:
        current_inspiration_left_index (pd.Index): Index corresponding to inspiration_trigger-inspiration surround
        current_inspiration_right_index (pd.Index): Index corresponding to inspiration_trigger+inspiration_surround
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


def get_ymax(triplet: pd.DataFrame, esophogeal_manometry_column: str, current_inspiration_left_index: pd.Index,
             current_inspiration_right_index: pd.Index) -> tuple[float, pd.Index]:
    """
    Gets the value and index of the highest value on the esophageal manometry channel between the current_inspiration_left_index
    and current_inspiration_right_index

    Args:
        triplet (pd.DataFrame): DataFrame of a triplet
        esophogeal_manometry_column (str): column that corresponds to the esophageal manometry channel
        current_inspiration_left_index (pd.Index): index corresponding to inspiration_trigger-inspiration surround
        current_inspiration_right_index (pd.Index): index corresponding to inspiration_trigger+inspiration_surround

    Returns:
        current_inspiration_ymax (float): Value of the highest value of the esophageal manometry channel between the
                                          current_inspiration_left_index and current_inspiration_right_index
        current_inspiration_ymax_idx (pd.Index): Index of the highest value of the esophageal manometry channel between
                                                 the current_inspiration_left_index and current_inspiration_right_index
    """

    # find ymax in the range 100ms around second inspiration trigger
    current_inspiration_ymax_range = triplet.loc[current_inspiration_left_index:current_inspiration_right_index]
    current_inspiration_ymax = current_inspiration_ymax_range[esophogeal_manometry_column].max()
    current_inspiration_ymax_idx = current_inspiration_ymax_range[esophogeal_manometry_column].idxmax()

    return current_inspiration_ymax, current_inspiration_ymax_idx


def left_and_right_current_expiration_idxs(triplet: pd.DataFrame, inspiration_triggers: pd.Index, expiration_triggers: pd.Index, ymax_idx: pd.Index,
                                           expiration_surround_s: float) -> tuple[pd.Index, pd.Index]:
    """
    Get indexes of timesteps corresponding to +-expiration from the current (second) breath expiration trigger

    Args:
        triplet (pd.DataFrame): DataFrame of a triplet
        inspiration_triggers (pd.Index): Indexes of inspiration triggers
        expiration_triggers (pd.Index): Indexes of expiration triggers
        ymax_idx (pd.Index): index of the ymax around the second inspiration trigger
        expiration_surround_s (float): how many seconds around the expiration trigger to save the index

    Returns:
        current_expiration_left_index (pd.Index): Index corresponding to expiration_trigger-expiration_surround_s
        current_expiration_right_index (pd.Index): Index corresponding to expiration_trigger+expiration_surround_s
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


def get_ymin(triplet: pd.DataFrame, esophogeal_manometry_column: str, current_expiration_left_index: pd.Index,
             current_expiration_right_index: pd.Index) -> tuple[float, pd.Index]:
    """
    Calculates the ymin in the range from current inspiration trigger to expiration_surround_s after current expiration trigger

    Args:
        triplet (pd.DataFrame): DataFrame of a triplet
        esophogeal_manometry_column (str): column that corresponds to the esophageal manometry channel
        current_expiration_left_index (pd.Index): index corresponding to expiration_trigger-expiration_surround_s surround
        current_expiration_right_index (pd.Index): index corresponding to expiration_trigger+expiration_surround_s

    Returns:
        current_expiration_ymin (float): Value of the lowest value of the esophageal manometry channel in the range
                            from current inspiration trigger to expiration_surround_s after current expiration trigger
        current_expiration_ymin_idx (pd.Index): Index of the lowest value of the esophageal manometry channel in the
                        range from current inspiration trigger to expiration_surround_s after current expiration trigger
    """

    # find ymin in the range from current inspiration trigger to 100ms after current expiration trigger
    current_inspiration_ymax_range = triplet.loc[current_expiration_left_index:current_expiration_right_index][
        esophogeal_manometry_column]
    current_expiration_ymin = current_inspiration_ymax_range.min()
    current_expiration_ymin_idx = current_inspiration_ymax_range.idxmin()

    return current_expiration_ymin, current_expiration_ymin_idx


def calculate_deltaPes(triplet: pd.DataFrame, inspiration_surround_s: float = .1, expiration_surround_s: float = .1) -> float:
    """
    Calculates the deltaPes of the central breath of a triplet

    Args:
        triplet (pd.DataFrame): DataFrame of a triplet
        inspiration_surround_s (float): how many seconds around the inspiration trigger to include in calculations
        expiration_surround_s (float): how many seconds around the expiration trigger to include in calculations

    Returns:
        deltaPes (float): Returns float of calculated deltaPes value
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


