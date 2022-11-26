import argparse
import glob
import json
import os
import sys

import dtw
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels import robust
from statsmodels.distributions.empirical_distribution import ECDF

# Add collection_scripts folder to path to import from dataset_utils
collection_path = os.path.join(os.path.dirname(__file__), '../collection_scripts')
sys.path.append(collection_path)
import dataset_utils


def get_color(i):
    colors = list(mcolors.TABLEAU_COLORS.values())
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

def plot_data(data, output='./test.png'):
    domains = data.keys()
    fig, ax = plt.subplots(1, len(domains), figsize=(22, 5))

    for i, domain in enumerate(domains):
        time_list = data[domain]['times']
        size_list = data[domain]['sizes']

        color = color=get_color(i)
        for j, (time, size) in enumerate(zip(time_list, size_list)):
            axes = ax[i] if i > 1 else ax
            axes.scatter(time, size, color=get_color(j), label=j)
            axes.set_title(domain)
            axes.set_xlabel('time (s)')
            axes.set_ylabel('packet size (bytes)')
            axes.legend()

    plt.savefig(output)
    plt.show()

def align_with_dtw(data, labels=['google.com', 'amazon.com', 'youtube.com']):
    columns = ['time', 'size']

    fig, ax = plt.subplots(1, 3)

    for label_idx, label in enumerate(labels):

        sample_list = data[label]

        # create reference frame 
        reference_df = pd.DataFrame(
            [[time, size] for time, size in sample_list[0]],
            columns=columns
        )

        # plot reference frame
        ax[label_idx].plot(reference_df['time'], reference_df['size'], alpha=0.1, color=get_color(label_idx))

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
            ax[label_idx].plot(ref_times, query_values, alpha=0.1, color=get_color(label_idx))

            # for each index in reference alignment, save value from query to average later
            for i, index in enumerate(alignment.index2):
                values_by_index[index].append(query_values.iloc[i])

        medians = np.array([ np.median(values_by_index[i]) for i in values_by_index ])
        mad = 2 * np.array([ robust.mad(values_by_index[i]) for i in values_by_index ])

        ax[label_idx].plot(reference_df['time'], medians, alpha=1, linewidth=1, color=get_color(label_idx))

        under_line, over_line = medians - mad, medians + mad

        ax[label_idx].fill_between(reference_df['time'], under_line, over_line, alpha=0.125, color=get_color(label_idx))

        ax[label_idx].set_title(label)
        ax[label_idx].set_xlabel('time (s)')
        ax[label_idx].set_ylabel('packet size (bytes)')

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
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sizes=[]
    for sample in data:
        for _, datum in sample.items():
            sizes.append(datum[0])

    ax[0].hist(sizes, bins=500)
    ax[0].set_xlabel('size (bytes)')
    ax[0].set_ylabel('count')

    cdf = ECDF(sizes)
    x = np.arange(max(sizes))
    y = cdf(x)

    ax[1].plot(x, y)
    ax[1].set_xlabel('size (bytes)')
    ax[1].set_ylabel('cdf')

    percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    for ptile in percentiles:
        val = np.where(y > ptile)[0][0]
        print(f'{ptile}: {val}')

    ax[1].grid()

    plt.savefig(filename)
    plt.show()


if __name__=='__main__':

    # labels = ['google.com', 'amazon.com', 'youtube.com']
    # data = {}
    # for label in labels:
    #     data[label] = load_data(f'collection_scripts/data/processed/processed_full/{label}')[label]
    # align_with_dtw(data, labels)

    #lengths_cdf(data, 'plots/sizes_cdf.png')

    data, labels = dataset_utils.load_data('datasets/data_directional_200-class_100-samples.pkl')
    sizes_cdf(data, 'plot/sizes_cdf.png')