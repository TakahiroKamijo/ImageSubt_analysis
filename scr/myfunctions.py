#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:14:30 2020
@author: miyazakishinichi
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import numpy as np
import itertools
import seaborn

# parameters
# imaging_interval (sec)
imaging_interval = 2  # (sec)
image_num_per_hour = int(3600 / imaging_interval)
# if DTS starts within 30 min, it is not used for analysis
margin_image_num = int(3600 / imaging_interval)
# The definition of out of DTS is 10 min after DTS end
out_DTS_duration = 600  # (sec)
out_DTS_image_num = int(out_DTS_duration / imaging_interval)
# Time window of rolling average
rolling_window_duration = 600  # (sec)
rolling_window_image_num = int(rolling_window_duration / imaging_interval)
# threshold of FoQ
FoQ_threshold = 0.05

# SIS analysis parameter
seaborn.set_style(style="ticks")

# lethargus analyzer
def maxisland_start_len_mask(boolean_array, fillna_index=-1, fillna_len=0):
    """detect sequential TRUEs which is named as island.
    This function returns the start index and length of the island.
    """
    if boolean_array.ndim != 2:
        pad_size = 1
    else:
        pad_size = boolean_array.shape[1]
    pad = np.zeros(pad_size, dtype=bool)
    mask = np.vstack((pad, boolean_array, pad))

    mask_step = mask[1:] != mask[:-1]
    idx = np.flatnonzero(mask_step.T)

    if len(idx)==0:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0])
    else:
        pass
    island_starts = idx[::2]
    island_lens = idx[1::2] - idx[::2]
    n_islands_percol = mask_step.sum(0) // 2

    bins = np.repeat(np.arange(boolean_array.shape[1]), n_islands_percol)
    scale = island_lens.max() + 1

    scaled_idx = np.argsort(scale * bins + island_lens)
    grp_shift_idx = np.r_[0, n_islands_percol.cumsum()]
    max_island_starts = island_starts[scaled_idx[grp_shift_idx[1:] - 1]]

    max_island_percol_start = max_island_starts % (boolean_array.shape[0] + 1)

    valid = n_islands_percol != 0
    cut_idx = grp_shift_idx[:-1][valid]
    max_island_percol_len = np.maximum.reduceat(island_lens, cut_idx)

    out_len = np.full(boolean_array.shape[1], fillna_len, dtype=int)
    out_len[valid] = max_island_percol_len
    out_index = np.where(valid, max_island_percol_start, fillna_index)
    return out_index, out_len, island_starts, island_lens


def DTS_column_analysis(analysis_res_df,
                        FoQ_raw,
                        DTS_boolean,
                        Wake_sleep_boolean,
                        body_size):
    """[summary]
    Args:
        analysis_res_df (TYPE): [description]
        FoQ_raw (TYPE): [description]
        DTS_boolean (TYPE): [description]
        Wake_sleep_boolean (TYPE): [description]
        body_size (TYPE): [description]
    """
    # lethargus bout analysis
    max_start, max_length, all_start, all_length = maxisland_start_len_mask(DTS_boolean)
    if len(max_start) == 1 & max_start[0] == 0:
        print("There is no sleep period (FoQ > 0.05)")
        return
    # quiescent bout analysis
    _, _, Sleep_bout_starts, Sleep_bout_durations = maxisland_start_len_mask(Wake_sleep_boolean)

    # wake bout analysis
    _, _, Wake_bout_starts, Wake_bout_durations = maxisland_start_len_mask(~Wake_sleep_boolean)

    column_name = ['bodysize_pixel',
                   'FoQ_during_DTS_arb',
                   'FoQ_out_arb',
                   'duration_hour',
                   'interpretation',
                   'MeanSleepBout_sec',
                   'MeanAwakeBout_sec',
                   'Transitions_/hour',
                   "TotalQ_sec",
                   "TotalA_sec"]
    result = []
    column_num, row_num = analysis_res_df.shape
    DTS_df = pd.DataFrame()
    for i in range(row_num):
        num = i + 1
        # extract quiescent island indices
        temp_area_indices = list(itertools.chain.from_iterable \
                                     (np.where((all_start >= column_num * i) & \
                                               (all_start <= column_num * num))))
        # quiescent island length
        quiescent_lengths = all_length[temp_area_indices]
        # count only the islands which is longer than 1hour
        quiescent_island_num = np.count_nonzero(quiescent_lengths > image_num_per_hour)

        # max island (= lethargus) end
        max_q_end = max_start[i] + max_length[i]
        # out of lethargus (5 min after lethargus end)
        max_q_out = max_q_end + out_DTS_image_num
        # tempFoQ is FoQ series of current chamber
        temp_foq = FoQ_raw.iloc[:, i]
        # calculate average FOQ during lethargus
        foq_mean = temp_foq.iloc[max_start[i]:max_q_end].mean()
        # calculate average FoQ out of lethargus
        foq_out = temp_foq.iloc[max_q_end:max_q_out].mean()
        # calculate lethargus length
        lethargus_length = max_length[i] / image_num_per_hour

        # init value
        judge, mean_q_duration, mean_a_duration, transitions, \
            total_q, total_a = 0, 0, 0, 0, 0, 0

        # check start point & end point
        if quiescent_island_num > 1:
            judge = "Multiple_DTS"
        elif quiescent_island_num == 0:
            judge = "No_DTS"
        elif max_start[i] < margin_image_num:
            judge = "DTS_Start_Within_60min"
        elif max_q_end > column_num - margin_image_num:
            judge = "DTS_not_end"
        else:
            judge = "Applicable"
            # extract from  30 min before lethargus to the end
            LeFoQdf = temp_foq.iloc[max_start[i] - margin_image_num:]
            DTS_df = pd.concat([DTS_df, LeFoQdf.reset_index().iloc[:, 1]], axis=1)

            q_starts_index = np.where((Sleep_bout_starts - column_num * i > max_start[i]) \
                                      & (Sleep_bout_starts - column_num * i < max_q_end))
            a_starts_index = np.where((Wake_bout_starts - column_num * i > max_start[i]) \
                                      & (Wake_bout_starts - column_num * i < max_q_end))
            q_starts_lethargus = Sleep_bout_starts[q_starts_index] - column_num * i
            a_starts_lethargus = Wake_bout_starts[a_starts_index] - column_num * i

            # calculate total Q
            q_durations_lethargus = Sleep_bout_durations[q_starts_index]
            total_q = np.sum(q_durations_lethargus) * 2
            # calculate total A
            a_durations_lethargus = Wake_bout_durations[a_starts_index][:-1]
            total_a = np.sum(a_durations_lethargus) * 2
            # calculate mean Q
            mean_q_duration = np.mean(q_durations_lethargus) * 2
            # calculate mean A
            mean_a_duration = np.mean(a_durations_lethargus) * 2
            # averaged transitions
            transitions = len(q_durations_lethargus) / lethargus_length

            # calucurate parameters each 15 min bins
            # 3 hour -> 12 bins
            os.makedirs("./subdivided/worm{}".format(num), exist_ok=True)
            bins = 6
            if max_length[i] < 6000:
                bins = max_length[i] // 900 - 1
            else:
                pass
            ex_start = max_start[i]
            if a_starts_lethargus[0] > q_starts_lethargus[0]:
                q_durations_lethargus = q_durations_lethargus[1:]
            # if A is more than Q, A is deleted
            if len(a_durations_lethargus) > len(q_durations_lethargus):
                a_durations_lethargus = a_durations_lethargus[:-1]
            if len(a_durations_lethargus) < len(q_durations_lethargus):
                q_durations_lethargus = q_durations_lethargus[:-1]
            Lethargus_QandA = np.stack((a_durations_lethargus,
                                        q_durations_lethargus), 1)

            for j in range(bins):
                start = j * 900 + ex_start
                end = (j + 1) * 900 + ex_start
                ex_Leth_QandA = Lethargus_QandA[np.where((a_starts_lethargus > start) & \
                                                         (a_starts_lethargus < end))]
                ex_Leth_QandA = pd.DataFrame(ex_Leth_QandA, columns=["A", "Q"])
                ex_Leth_QandA.to_csv("./subdivided/worm{0}/leth_{1}.csv".format(num, j))

            # make DTS boolean dataframe
            # DTS booleanのmax_startからmax_q_endまでの間をTrueにする 他はFALSEにする
            DTS_boolean["DTS_mask"] = 0
            DTS_boolean["DTS_mask"].iloc[max_start[i]:max_q_end] = 1
            DTS_boolean["Before_DTS_mask"] = 0
            DTS_boolean["Before_DTS_mask"].iloc[max_start[i] - 300:max_start[i]] = 1
            DTS_boolean["After_DTS_mask"] = 0
            DTS_boolean["After_DTS_mask"].iloc[max_q_end:max_q_end + 300] = 1
            DTS_boolean.to_csv("./subdivided/worm{0}/DTS_mask.csv".format(num))

        temp_result = np.array([body_size, foq_mean, foq_out,
                                lethargus_length, judge, mean_q_duration,
                                mean_a_duration, transitions, total_q,
                                total_a])
        result.append(temp_result)
    result_df = pd.DataFrame(result, index=["worm" + str(i + 1) for i in range(row_num)],
                             columns=column_name)
    result_df.to_csv('./result_summary.csv')
    DTS_df.to_csv('./Lethargus_dataframe.csv')


def lethargus_analyzer(analysis_res_df, body_size, fig_rnum, fig_cnum):
    # make result folder
    os.makedirs("./results", exist_ok=True)
    os.chdir("./results")
    os.makedirs("./figures", exist_ok=True)

    # Make column names
    analysis_res_df.columns = ["worm" + str(i + 1) for i in range(analysis_res_df.shape[1])]

    # make time axis
    analysis_res_df["time_axis(min)"] = [sec / (60 / imaging_interval) for sec in range(len(analysis_res_df))]
    analysis_res_df = analysis_res_df.set_index(['time_axis(min)'])

    # make boolean array
    # if the activity > 1% of the body, Wake
    # wake = 0, sleep = 1
    Wake_sleep_boolean = analysis_res_df < body_size / 100
    Wake_sleep_boolean.to_csv('./Wake_sleep_boolean.csv')

    # calculate FoQ
    FoQ_raw = Wake_sleep_boolean.rolling(int(rolling_window_image_num), min_periods=1, center=True).mean()
    FoQ_raw.to_csv('./FoQ_data.csv')

    # Make FoQ figures each plot without timeaxis plot
    for k in FoQ_raw.columns:
        plt.figure()
        FoQ_raw[k].plot(ylim=[0, 1], colormap="gray")
        plt.savefig('./figures/{}.png'.format(k))
        plt.clf()
        plt.close()
    # make summary plot
    fig_rnum = (len(FoQ_raw.columns) + fig_cnum - 1) // fig_cnum
    fig, axes = plt.subplots(fig_rnum,
                             fig_cnum,
                             figsize=(fig_cnum * 2,
                                      fig_rnum * 2),
                             tight_layout=True,
                             facecolor="whitesmoke",
                             squeeze=False)

    # if the number of rows is  more than 1, make 2 dimentinal subplot
    if len(FoQ_raw.columns) > 1:
        for fig_num in range(len(FoQ_raw.columns)):
            temp_row = fig_num // fig_cnum
            temp_col = fig_num % fig_cnum
            axes[temp_row, temp_col].plot(FoQ_raw.index,
                                          FoQ_raw.iloc[:, fig_num],
                                          color="black")
            # put worm number on the plot
            axes[temp_row, temp_col].text(1, 1.2, f'worm{fig_num}')
            axes[temp_row, temp_col].set_ylim(0, 1)
        plt.savefig('./figures/summary.png')
        plt.clf()
        plt.close()

    # detect lethargus
    # for searching letahrgus enter and end, make boolean array
    # if the FoQ > 0.05 True, if not False
    DTS_boolean = FoQ_raw > FoQ_threshold

    DTS_column_analysis(analysis_res_df,
                        FoQ_raw,
                        DTS_boolean,
                        Wake_sleep_boolean,
                        body_size)


def SIS_analyzer(analysis_res_df, body_size, fig_rnum, fig_cnum, SIS_analysis_duration_list):
    # make result folder
    os.makedirs("./results", exist_ok=True)
    os.chdir("./results")
    os.makedirs("./figures", exist_ok=True)

    # Make column names
    analysis_res_df.columns = ["worm" + str(i + 1) for i in range(analysis_res_df.shape[1])]

    # make time axis
    analysis_res_df["time_axis(min)"] = [sec / (60 / imaging_interval) for sec in range(len(analysis_res_df))]
    analysis_res_df = analysis_res_df.set_index(['time_axis(min)'])

    # make boolean array
    # if the activity > 1% of the body, Wake
    # wake = 0, sleep = 1
    Wake_sleep_boolean = analysis_res_df < body_size / 100
    Wake_sleep_boolean.to_csv('./Wake_sleep_boolean.csv')

    # calculate FoQ
    FoQ_raw = Wake_sleep_boolean.rolling(int(rolling_window_image_num), min_periods=1, center=True).mean()
    FoQ_raw.to_csv('./FoQ_data.csv')

    # Make FoQ figures each plot without timeaxis plot
    for k in FoQ_raw.columns:
        plt.figure()
        FoQ_raw[k].plot(ylim=[0, 1], colormap="gray")
        plt.savefig('./figures/{}.png'.format(k))
        plt.clf()
        plt.close()

    # make summary plot
    fig_rnum = (len(FoQ_raw.columns) + fig_cnum - 1) // fig_cnum
    fig, axes = plt.subplots(fig_rnum,
                             fig_cnum,
                             figsize=(fig_cnum * 2,
                                      fig_rnum * 2),
                             tight_layout=True,
                             facecolor="whitesmoke",
                             squeeze=False)

    # if the number of rows is  more than 1, make 2 dimentinal subplot
    if len(FoQ_raw.columns) > 1:
        for fig_num in range(len(FoQ_raw.columns)):
            temp_row = fig_num // fig_cnum
            temp_col = fig_num % fig_cnum
            axes[temp_row, temp_col].plot(FoQ_raw.index,
                                          FoQ_raw.iloc[:, fig_num],
                                          color="black")
            # put worm number on the plot
            axes[temp_row, temp_col].text(1, 1.2, f'worm{fig_num}')
            axes[temp_row, temp_col].set_ylim(0, 1)
        plt.savefig('./figures/summary.png')
        plt.clf()
        plt.close()

    # detect lethargus
    # for searching letahrgus enter and end, make boolean array
    # if the FoQ > 0.05 True, if not False
    SIS_boolean = FoQ_raw > FoQ_threshold

    SIS_column_analysis(analysis_res_df, FoQ_raw, SIS_boolean, Wake_sleep_boolean, body_size, SIS_analysis_duration_list)


def SIS_column_analysis(analysis_res_df,
                        FoQ_raw,
                        SIS_boolean,
                        Wake_sleep_boolean,
                        body_size,
                        SIS_analysis_duration_list):
    """[summary]
    SIS state analysis for each column of dataframe
    """
    column_name = ['bodysize_pixel',
                   'Average_FoQ',
                   'MeanSleepBout_sec',
                   'MeanAwakeBout_sec',
                   'Transitions_per_hour',
                   "TotalQ_sec",
                   "TotalA_sec"]

    row_num, column_num = analysis_res_df.shape
    # analysis for each column
    for duration_hour in SIS_analysis_duration_list:
        result = []
        current_row_length = duration_hour * image_num_per_hour
        if len(analysis_res_df) > duration_hour * image_num_per_hour:
            # limit the df length to duration_hour
            limited_analysis_res_df = analysis_res_df.iloc[:int(duration_hour * image_num_per_hour)]
            limited_SIS_boolean = SIS_boolean.iloc[:int(duration_hour * image_num_per_hour)]
            limited_Wake_sleep_boolean = Wake_sleep_boolean.iloc[:int(duration_hour * image_num_per_hour)]
            limited_FoQ = FoQ_raw.iloc[:int(duration_hour * image_num_per_hour)]

            # Bout analysis
            max_start, max_length, all_start, all_length = maxisland_start_len_mask(limited_SIS_boolean)
            # quiescent bout analysis
            _, _, Sleep_bout_starts, Sleep_bout_durations = maxisland_start_len_mask(limited_Wake_sleep_boolean)
            # wake bout analysis
            _, _, Wake_bout_starts, Wake_bout_durations = maxisland_start_len_mask(~limited_Wake_sleep_boolean)

            for current_col in range(column_num):
                next_col = current_col + 1
                average_foq = limited_FoQ.iloc[:, current_col].mean()
                # init value
                mean_q_duration, mean_a_duration, transitions, total_q, total_a = 0, 0, 0, 0, 0

                q_starts_index = np.where((Sleep_bout_starts > current_row_length * current_col) \
                                          & (Sleep_bout_starts < current_row_length * next_col))
                a_starts_index = np.where((Wake_bout_starts > current_row_length * current_col) \
                                          & (Wake_bout_starts < current_row_length * next_col))

                # calculate total Q
                q_durations = Sleep_bout_durations[q_starts_index]
                total_q = np.sum(q_durations) * 2
                # calculate total A
                a_durations = Wake_bout_durations[a_starts_index][:-1]
                total_a = np.sum(a_durations) * 2
                # calculate mean Q
                mean_q_duration = np.mean(q_durations) * 2
                # calculate mean A
                mean_a_duration = np.mean(a_durations) * 2
                # averaged transitions
                transitions = len(q_durations) / duration_hour

                temp_result = np.array([body_size,
                                        average_foq,
                                        mean_q_duration,
                                        mean_a_duration,
                                        transitions,
                                        total_q,
                                        total_a])
                result.append(temp_result)
            result_df = pd.DataFrame(result,
                                     index=["worm" + str(i + 1) for i in range(column_num)],
                                     columns=column_name)
            result_df.to_csv(f'./result_{duration_hour}hour_summary.csv')