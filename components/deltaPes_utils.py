import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import re


def extract_breath_info(breath_filepath):

    patient_id_pattern = 'pt?(\d*)_'
    day_id_pattern = 'day(\d*)[^\d]'
    breath_id_pattern = 'breath_?(\d*)'


    patient = re.findall(patient_id_pattern, breath_filepath, flags=re.IGNORECASE)[0]
    day = re.findall(day_id_pattern, breath_filepath, flags=re.IGNORECASE)[0]
    breath = re.findall(breath_id_pattern, breath_filepath, flags=re.IGNORECASE)[0]

    return patient, day, breath


def get_inspiration_expiration_triggers(triplet, deliniation_column):
    # get index of inspiration and expiration triggers
    inspiration_triggers = triplet[triplet[deliniation_column] == 1].index
    expiration_triggers = triplet[triplet[deliniation_column] == -1].index
    all_triggers = triplet[(triplet[deliniation_column] == 1) | (triplet[deliniation_column] == -1)].index

    return inspiration_triggers, expiration_triggers, all_triggers


def left_and_right_current_inspiration_idxs(triplet, inspiration_triggers, inspiration_surround_s):
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


def get_ymax(triplet, esophogeal_manometry_column, current_inspiration_left_index, current_inspiration_right_index):
    # find ymax in the range 100ms around second inspiration trigger
    current_inspiration_ymax_range = triplet.loc[current_inspiration_left_index:current_inspiration_right_index]
    current_inspiration_ymax = current_inspiration_ymax_range[esophogeal_manometry_column].max()
    current_inspiration_ymax_idx = current_inspiration_ymax_range[esophogeal_manometry_column].idxmax()

    return current_inspiration_ymax, current_inspiration_ymax_idx


def left_and_right_current_expiration_idxs(triplet, inspiration_triggers, expiration_triggers, ymax_idx,
                                           expiration_surround_s):
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


def get_ymin(triplet, esophogeal_manometry_column, current_expiration_left_index, current_expiration_right_index):
    # find ymin in the range from current inspiration trigger to 100ms after current expiration trigger
    current_inspiration_ymax_range = triplet.loc[current_expiration_left_index:current_expiration_right_index][
        esophogeal_manometry_column]
    current_expiration_ymin = current_inspiration_ymax_range.min()
    current_expiration_ymin_idx = current_inspiration_ymax_range.idxmin()

    return current_expiration_ymin, current_expiration_ymin_idx


def calculate_deltaPes(breath_filepath, inspiration_surround_s=.1, expiration_surround_s=.1):

    # get breath file
    triplet = pd.read_csv(breath_filepath)

    # set some variables for ease
    deliniation_column = 't:Fsp'
    esophogeal_manometry_column = 's:Pes'

    # extract breath info
    patient, day, breath = extract_breath_info(breath_filepath)

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


def graph_deltaPes(breath_filepath, figsize=(20, 10), inspiration_surround_s=.1, expiration_surround_s=.1):
    # get breath file
    triplet = pd.read_csv(breath_filepath)

    # set some variables for ease
    deliniation_column = 't:Fsp'
    esophogeal_manometry_column = 's:Pes'

    # extract breath info
    patient, day, breath = extract_breath_info(breath_filepath)

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

    # make figure object to draw on
    fig = plt.figure(figsize=figsize)
    font_size = 25

    # get ymin and ymax of esophogeal manometry channel
    ymin = triplet[esophogeal_manometry_column].min()
    ymax = triplet[esophogeal_manometry_column].max()
    delta_y = ymax - ymin

    # plot Pes of the triplet
    plt.plot(triplet['Time'], triplet[esophogeal_manometry_column])

    # plot area 100ms before and after current breath inspiration
    current_inspiration_trigger_y = triplet[esophogeal_manometry_column].loc[inspiration_triggers[1]]

    plt.hlines(y=current_inspiration_trigger_y,
               xmin=triplet['Time'].loc[current_inspiration_left_index],
               xmax=triplet['Time'].loc[current_inspiration_right_index],
               colors='green', ls='--')
    plt.vlines(x=triplet['Time'].loc[current_inspiration_left_index],
               ymin=current_inspiration_trigger_y - (delta_y * 0.05),
               ymax=current_inspiration_trigger_y + (delta_y * 0.05),
               colors='green', linewidth=4)
    plt.vlines(x=triplet['Time'].loc[current_inspiration_right_index],
               ymin=current_inspiration_trigger_y - (delta_y * 0.05),
               ymax=current_inspiration_trigger_y + (delta_y * 0.05),
               colors='green', linewidth=4)

    # plot current inspiration ymax
    plt.plot(triplet['Time'].loc[current_inspiration_ymax_idx],
             triplet[esophogeal_manometry_column].loc[current_inspiration_ymax_idx],
             marker='o', markersize=10, markeredgecolor='green', markerfacecolor='green', label='PesBaseline')

    plt.vlines(x=triplet['Time'].loc[current_inspiration_ymax_idx],
               ymin=current_inspiration_trigger_y,
               ymax=triplet[esophogeal_manometry_column].loc[current_inspiration_ymax_idx],
               colors='green', ls=':', linewidth=4)

    # plot area from current expiration trigger to 100ms after current expiration trigger
    current_expiration_trigger_y = triplet[esophogeal_manometry_column].loc[expiration_triggers[1]]

    plt.hlines(y=current_expiration_trigger_y,
               xmin=triplet['Time'].loc[current_expiration_left_index],
               xmax=triplet['Time'].loc[current_expiration_right_index],
               colors='red', ls='--')
    plt.vlines(x=triplet['Time'].loc[current_expiration_left_index],
               ymin=current_expiration_trigger_y - (delta_y * 0.05),
               ymax=current_expiration_trigger_y + (delta_y * 0.05),
               colors='red', linewidth=4)
    plt.vlines(x=triplet['Time'].loc[current_expiration_right_index],
               ymin=current_expiration_trigger_y - (delta_y * 0.05),
               ymax=current_expiration_trigger_y + (delta_y * 0.05),
               colors='red', linewidth=4)

    # plot current inspiration ymin
    plt.plot(triplet['Time'].loc[current_expiration_ymin_idx],
             triplet[esophogeal_manometry_column].loc[current_expiration_ymin_idx],
             marker='o', markersize=10, markeredgecolor='red', markerfacecolor='red', label='PesMin')
    plt.vlines(x=triplet['Time'].loc[current_expiration_ymin_idx],
               ymin=triplet[esophogeal_manometry_column].loc[current_expiration_ymin_idx],
               ymax=current_expiration_trigger_y,
               colors='red', ls=':', linewidth=4)

    # plot lines showing deltaPES
    plt.vlines(x=triplet['Time'].loc[round((current_inspiration_ymax_idx + current_expiration_ymin_idx) / 2)],
               ymin=current_expiration_ymin, ymax=current_inspiration_ymax, color='black', label='deltaPes')
    plt.hlines(y=current_inspiration_ymax,
               xmin=triplet['Time'].loc[current_inspiration_ymax_idx],
               xmax=triplet['Time'].loc[round((current_inspiration_ymax_idx + current_expiration_ymin_idx) / 2)],
               colors='black', ls='--')
    plt.hlines(y=current_expiration_ymin,
               xmin=triplet['Time'].loc[round((current_inspiration_ymax_idx + current_expiration_ymin_idx) / 2)],
               xmax=triplet['Time'].loc[current_expiration_ymin_idx],
               colors='black', ls='--')

    # plot solid lines at inspiration
    plt.vlines(x=triplet['Time'].loc[inspiration_triggers], ymin=ymin, ymax=ymax,
               colors='green', ls='--', label='inspiration triggers')
    # plot dashed lines at expiration
    plt.vlines(x=triplet['Time'].loc[expiration_triggers], ymin=ymin, ymax=ymax,
               colors='red', ls='--', label='expiration triggers')

    # shade inspiration and expiration regions
    for i in range(3):
        if i == 2:
            plt.axvspan(inspiration_triggers[i], expiration_triggers[i],
                        color='green', alpha=0.1, label='inspiration region')
            plt.axvspan(expiration_triggers[i], triplet['Time'].index[-1],
                        color='red', alpha=0.1, label='expiration region')
        else:
            plt.axvspan(inspiration_triggers[i], expiration_triggers[i],
                        color='green', alpha=0.1)
            plt.axvspan(expiration_triggers[i], inspiration_triggers[i + 1],
                        color='red', alpha=0.1)

    # show figure
    # only show xticks at inspiration and expiration triggers
    plt.yticks(fontsize=font_size - 10)
    plt.xticks(triplet['Time'].loc[all_triggers], fontsize=font_size - 10)

    plt.title(f'Esophogeal Manometry of Pt {patient} Day {day} Breath {breath} (deltaPes: {deltaPes:.2f})',
              fontsize=font_size)
    plt.xlabel('Time', fontsize=font_size)
    plt.ylabel('Esophogeal Pressure', fontsize=font_size)
    plt.legend(loc="lower right", fontsize=font_size - 10)

    return fig


