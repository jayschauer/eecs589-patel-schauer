import argparse
import glob
import json
import math
import os
import random
import seaborn as sns
import sys

import dtw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels import robust
from statsmodels.distributions.empirical_distribution import ECDF

# Add collection_scripts folder to path to import from dataset_utils
collection_path = os.path.join(os.path.dirname(__file__), '../collection_scripts')
sys.path.append(collection_path)
import dataset_utils

DPI = 200

def get_color(i):
    colors = list(sns.color_palette('deep'))
    return colors[i % len(colors)]

def load_data(data_dir):
    data = {}

    path = os.path.join(data_dir, '**/*.json')
    for filename in glob.glob(path, recursive=True):
        with open(filename, 'r') as infile:
            file_data = json.load(infile)

        label = file_data['label']
        times = file_data['time']
        sizes = file_data['size']

        if label not in data:
            data[label] = []
        
        data[label].append(list(zip(times, sizes)))

    return data

def align_with_dtw(data, filename, labels=['google.com', 'amazon.com', 'youtube.com']):
    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)

    # create figure and subplots
    num_cols = 3
    num_rows = math.ceil(len(labels) / 3)
    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False, figsize=(num_cols * 6, num_rows * 4.5))

    for label_idx, label in enumerate(labels):
        # get axes for index
        row = label_idx // num_cols
        col = label_idx % num_cols
        ax = axes[row, col]

        sample_list = data[label]

        # create reference frame
        columns = ['time', 'size']
        reference_df = pd.DataFrame(
            [[time, size] for time, size in sample_list[0]],
            columns=columns
        )

        # plot reference frame
        color = get_color(label_idx)
        ax.plot(reference_df['time'], reference_df['size'], alpha=0.1, color=color)

        # keep track of values for each index in reference frame to calculate average
        values_by_index = { i: [size] for i, size in enumerate(reference_df['size']) }
        
        for sample in sample_list[1:20]:
            # create query to align with reference
            query_df = pd.DataFrame(
                [[time, size] for time, size in sample],
                columns=columns
            )   

            # perform alignment and get aligned values
            alignment = dtw.dtw(query_df, reference_df)
            ref_times = reference_df['time'][alignment.index2] 
            query_values = query_df['size'][alignment.index1]

            # plot aligned query against reference indices
            ax.plot(ref_times, query_values, alpha=0.1, color=color)

            # for each index in reference alignment, save value from query to average later
            for i, index in enumerate(alignment.index2):
                values_by_index[index].append(query_values.iloc[i])

        medians = np.array([ np.median(values_by_index[i]) for i in values_by_index ])
        mad = 2 * np.array([ robust.mad(values_by_index[i]) for i in values_by_index ])
        
        ax.plot(reference_df['time'], medians, alpha=1, linewidth=1, color=color)

        under_line, over_line = medians - mad, medians + mad
        ax.fill_between(reference_df['time'], under_line, over_line, alpha=0.125, color=color)

        ax.set_title(label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Packet size (bytes)')

    plt.savefig(filename, bbox_inches='tight', dpi=DPI)
    plt.show()

def lengths_cdf(data, filename):
    '''
    Computes the cumulative density function (CDF) for the series lengths.

    data: data dictionary from load_data
    '''
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    lengths = [ len(sample) for sample in data ]

    ax[0].hist(lengths, bins=500)
    ax[0].set_xlabel('length (# packets)')
    ax[0].set_ylabel('count')

    cdf = ECDF(lengths)
    x = np.arange(max(lengths))
    y = cdf(x)

    ax[1].plot(x, y)
    ax[1].set_xlabel('length (# packets)')
    ax[1].set_ylabel('cdf')

    percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    for ptile in percentiles:
        val = np.where(y > ptile)[0][0]
        print(f'{ptile}: {val}')

    ax[1].grid()

    plt.savefig(filename)
    plt.show()

def sizes_cdf(data, filename):
    '''
    Computes the cumulative density function (CDF) for the packet sizes.

    data: data dictionary from load_data
    '''
    sns.set_theme(style='darkgrid', palette='deep')
    sns.set(font_scale=0.8)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Extract packet size from data
    sizes=[]
    for sample in data:
        for _, datum in sample.items():
            sizes.append(datum[0])
    sizes = np.array(sizes)

    # Plot histogram of packet sizes on first axes
    sns.histplot(sizes, log_scale=True, ax=ax[0])
    ticks = [72, 78, 125, 166, 311, np.max(sizes)]

    # Plot empirical CDF on second axes
    sns.histplot(
        data=sizes, log_scale=True, element="step", fill=False,
        cumulative=True, stat="density", common_norm=False,
        ax=ax[1]
    )

    # Labels and stuff
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('CDF')
    for axes in ax:
        axes.set_xlabel('size (bytes)')
        axes.set_xticks(ticks)
        axes.set_xticklabels(ticks)
        axes.set_yticklabels(list(axes.get_yticklabels()))

    # Compute the empirical CDF and get values for certain percentiles
    # cdf = ECDF(sizes)(np.arange(max(sizes)))
    # percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    # for ptile in percentiles:
    #     val = np.where(cdf > ptile)[0][0]
    #     print(f'{ptile}: {val}')

    plt.savefig(filename, bbox_inches='tight', dpi=DPI)
    plt.show()

def times_cdf(data, filename, max_delay=0.1):
    '''
    Computes the cumulative density function (CDF) for the total times.

    data: data dictionary from load_data
    filename: file to save plot to
    max_delay: maximum delay in seconds to add to each packet
    '''
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    base_times, delayed_times = [], []  # need list to plot, so can't just keep running sum
    for sample in data:
        times = list(sample.keys())
        
        base_times.append(times[-1]) # final timepoint = total time elapsed

        # [0, max_delay) second delay for all but first packet (first always at 0)
        total_delay = sum([random.random() * max_delay for _ in range(len(times) - 1)])
        delayed_time = times[-1] + total_delay
        delayed_times.append(delayed_time)

    ax[0].hist(base_times, bins=500, color='b', label='base')
    ax[0].hist(delayed_times, bins=500, color='r', alpha=0.6, label=f'{max_delay}s delay')
    ax[0].set_xlabel('time (seconds)')
    ax[0].set_ylabel('count')
    ax[0].legend()

    base_cdf = ECDF(base_times)
    base_x = np.arange(max(delayed_times))
    base_y = base_cdf(base_x)

    delayed_cdf = ECDF(delayed_times)
    delayed_x = np.arange(max(delayed_times))
    delayed_y = delayed_cdf(delayed_x)

    ax[1].plot(base_x, base_y, color='b', label='base')
    ax[1].plot(delayed_x, delayed_y, color='r', label=f'{max_delay}s delay')
    ax[1].set_xlabel('time (seconds)')
    ax[1].set_ylabel('cdf')
    ax[1].legend()

    plt.savefig(filename)
    plt.show()

    base_sum, delay_sum = sum(base_times), sum(delayed_times)
    latency = (delay_sum - base_sum) / base_sum
    print(f'Latency increase from base: {latency * 100:.2f}%')
    
def direction_counts(data, labels):
    cols = ['incoming', 'outgoing', 'label']
    df = pd.DataFrame([], columns = cols)

    for series, label in zip(data, labels):
        if label not in [0, 1, 2, 3, 4, 5]:
            continue

        directions = np.array([ datum[1] for _, datum in series.items() ])
        outgoing = len(np.where(directions > 0)[0])
        incoming = len(directions) - outgoing
        df.loc[len(df.index)] = [incoming, outgoing, label]

    sns.set_theme(style='darkgrid', palette='deep')
    sns.set(font_scale=0.8)

    df = pd.melt(df, id_vars='label', var_name='direction', value_name='count')
    sns.barplot(df, x='label', y='count', hue='direction')

    plt.show()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    choices = ['compare_traces', 'compare_google', 'length_cdf', 'sizes_cdf', 'times_cdf', 'counts']
    parser.add_argument('mode', type=str, choices=choices)
    args = parser.parse_args()

    # Plot average traces for different classes
    if args.mode == 'compare_traces' or args.mode == 'compare_google':
        if args.mode == 'compare_traces':
            labels = ['google.com', 'youtube.com', 'baidu.com', 'bilibili.com', 'facebook.com', 'instagram.com']
            filename = 'figs/average_traces.png'
        else:
            labels = ['google.com', 'google.co.uk', 'google.co.in']
            filename = 'figs/average_traces_google.png'

        data = {}
        for label in labels:
            data[label] = load_data(f'collection_scripts/data/processed/processed_full/{label}')[label]
        align_with_dtw(data, filename, labels)

    elif args.mode == 'length_cdf':
        data, labels = dataset_utils.load_data('datasets/data_directional_200-class_100-samples.pkl')
        lengths_cdf(data, 'figs/lengths_cdf.png')

    elif args.mode == 'sizes_cdf':
        data, labels = dataset_utils.load_data('datasets/data_directional_200-class_100-samples.pkl')
        sizes_cdf(data, 'figs/sizes_cdf.png')

    elif args.mode == 'times_cdf':
        data, labels = dataset_utils.load_data('datasets/data_directional_200-class_100-samples.pkl')
        times_cdf(data, 'figs/times_cdf.png')

    elif args.mode == 'counts':
        data, labels = dataset_utils.load_data('datasets/data_directional_200-class_100-samples.pkl')
        direction_counts(data, labels)