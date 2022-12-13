import argparse
import os
import sys

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sktime.base import load
from sktime import datatypes
from statsmodels.distributions.empirical_distribution import ECDF
from tabulate import tabulate

from classify import make_dataframe, make_iat_dataframe
from analyze_results import modified_accuracy_score

# Add collection_scripts folder to path to import from dataset_utils
collection_path = os.path.join(os.path.dirname(__file__), '../collection_scripts')
sys.path.append(collection_path)
from dataset_utils import load_data


def load_data_and_model(args):
    '''
    Loads data and model based on command line arguments.

    args: command line arguments

    Returns: X - list of samples to use in testing. Calling functions will modify this
             y - list of labels corresponding to each sample.
             model - trained model to use for testing.
    '''
    print('Loading data...')
    data, labels = load_data(args['data'])

    _, X_test, _, y_test = train_test_split(
        data, labels, random_state=589, test_size=0.25, shuffle=True
    )
    y = np.array(y_test)

    print('Loading model...')
    model = args['model']
    if os.path.isdir(model):  # if directory provided, use model.zip in that directory
        model = os.path.join(args['model'], 'model')
    model = load(model)

    return X_test, y, model


def pad_sizes(data, strategy, length):
    '''
    Creates a dataframe with sizes padded based on specified strategy.

    data: list of samples where each sample is a dictionary of time: size pairs
    strategy: padding method (either 'max' or 'round')
    length: length to pad to if max padding, or value to pad to multiple of if rounding 
            (e.g. if 32, pads to nearest multiple of 32). If given as a tuple, pads 
            outgoing (i.e. query) packets with first value and incoming (i.e. response)
            packets with second value.

    Returns: (df, overhead) - df is pd-multiindex dataframe, overhead is percent
    overhead compared to base
    '''

    # default: no padding
    if strategy not in ['max', 'round']:
        return make_dataframe(data), 0

    cols = ['timepoints', 'packet_size']
    padded, base = 0, 0
    Xlist = []

    for series in data:
        val_list = []
        for time, datum in series.items():
            # If tuple provided, pad incoming and outgoing packets differently
            if type(length) is tuple:
                if int(datum[1]) == 1:
                    pad_length = length[0]  # query
                else:
                    pad_length = length[1]  # response
            else:
                pad_length = length  # otherwise pad both the same

            # Apply strategy
            if strategy == 'max':
                padded_val = max(pad_length, datum[0])  # higher of length and existing value
            else:
                padded_val = math.ceil(datum[0] / pad_length) * pad_length  # round up to nearest multiple of length

            val_list.append([time, padded_val])
            base += datum[0]
            padded += padded_val

        Xlist.append(pd.DataFrame(val_list, columns=cols))

    overhead = (padded - base) / base
    return datatypes.convert_to(Xlist, to_type='pd-multiindex'), overhead


def add_time_delay(data, max_delay, direction=None):
    '''
    Creates a dataframe with delay added on to each packet.

    data: list of samples where each sample is a dictionary of time: size pairs
    length: delay (in seconds) to add to each packet. The delay for each packet
            will be added on to each subsequent packet as well.
    direction: direction of packets to modify. If None, modifies both incoming and
            outgoing packets

    Returns: (df, overhead) - df is pd-multiindex dataframe, overhead is percent
    overhead compared to base
    '''

    cols = ['timepoints', 'packet_size']
    base_time, delayed_time = 0, 0
    Xlist = []

    base_times, delayed_times = [], []

    for series in data:
        val_list = []

        delays = [0]
        # construct dataframe with new times
        for i, (time, datum) in enumerate(series.items()):
            delay = 0
            if i > 0:  # no delay since first time point always arrives at time 0
                if direction is None or direction == int(datum[1]):
                    delay = delays[-1] + random.random() * max_delay
                    delays.append(delay)
            val_list.append([time + delay, datum[0]])
            base_times.append(time)
            delayed_times.append(time + delay)

        if direction is None:
            final_time = list(series.keys())[-1]
            base_time += final_time
            delayed_time += final_time + delays[-1]

        Xlist.append(pd.DataFrame(val_list, columns=cols))

    overhead = None if direction else (delayed_time - base_time) / base_time
    return datatypes.convert_to(Xlist, to_type='pd-multiindex'), overhead, base_times, delayed_times


def add_time_delay_iat(data, max_delay, direction=None):
    """
    Modifies the dataframe by adding time to each packet

    data: pd-multiindex dataframe where first index is instance of timeseries (sample) and second index is index.
            Contains columns time, packet_size, and direction (only if direction parameter is not None).
    length: delay (in seconds) to add to each packet. The delay for each packet
            will be added on to each subsequent packet as well.
    direction: direction of packets to modify. If None, modifies both incoming and
            outgoing packets. Can be -1 or 1.

    Returns: (df, overhead) - df is pd-multiindex dataframe with only time and packet size columns, overhead is percent
    overhead compared to base
    """

    base_time, delayed_time = 0, 0
    base_times, delayed_times = [], []

    num_samples = data.index.levshape[0]
    # i is sample index
    for i in range(0, num_samples):
        sample_length = data.loc[i].index.shape[0]
        delays = pd.Series(np.random.uniform(0, max_delay, sample_length))
        delays.loc[0] = 0  # no delay for first packet since it always arrives at time zero
        if direction is not None:  # set delays for direction not delayed to zero
            delayed_direction = data.loc[i, 'direction'] == direction
            delays *= delayed_direction

        # add delays to the inter-arrival times
        sample_time = data.loc[i, 'time'].sum()
        data.loc[i, 'time'] = (data.loc[i, 'time'] + delays).values
        delayed_sample_time = data.loc[i, 'time'].sum()

        # save some statistics on the sample times:
        base_times.append(sample_time)
        delayed_times.append(delayed_sample_time)
        base_time += sample_time
        delayed_time += delayed_sample_time

    overhead = None if direction else (delayed_time - base_time) / base_time
    return data[['time', 'packet_size']], overhead, base_times, delayed_times


def predict(args):
    '''
    Basic prediction with no padding.
    '''
    X, y, model = load_data_and_model(args)
    X = make_dataframe(X)  # no modification

    print('Making predictions...')
    pred = model.predict(X)

    with open(args['label_file'], 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='weighted')
    modified_acc = modified_accuracy_score(y, pred, labels)

    print(f'Accuracy: {acc}')
    print(f'F1-Score: {f1}')
    print(f'Modified Accuracy: {modified_acc}')


def max_padded_predictions(args):
    '''
    Makes predictions with sizes padded to 25th, 50th, 75th, 90th, and 99th percentile
    of sizes in data.

    args: command line arguments

    Returns: df - dataframe with percentile, size padded to, eval metrics, and % overhead
    '''
    data, y, model = load_data_and_model(args)

    # get sizes to determine percentiles
    full_df = make_dataframe(data)
    cdf_func = ECDF(full_df['packet_size'])
    cdf = cdf_func(np.arange(max(full_df['packet_size'])))

    percentiles = [0.25, 0.5, 0.75, 0.9, 0.99]
    values = [np.where(cdf > ptile)[0][0] for ptile in percentiles]

    # dataframe to store output info
    df = pd.DataFrame([], columns=['percentile', 'size', 'accuracy', 'f1-score', 'modified_acc', 'overhead'])

    for ptile, value in zip(percentiles, values):
        print(f'Making predictions for {int(ptile * 100)}th percentile...')
        X, overhead = pad_sizes(data, 'max', value)

        pred = model.predict(X)

        with open(args['label_file'], 'r') as fh:
            labels = [line.strip() for line in fh.readlines()]

        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average='weighted')
        modified_acc = modified_accuracy_score(y, pred, labels)

        # add values to output dataframe
        df.loc[len(df)] = [int(ptile * 100), value, acc, f1, modified_acc, overhead]

    print(tabulate(df, headers=df.columns))
    return df


def block_padding_predictions(args):
    '''
    Makes predictions with sizes rounded up to multiples of various powers of 2.
    of sizes in data.

    args: command line arguments

    Returns: df - dataframe with size padded to, eval metrics, and % overhead
    '''
    data, y, model = load_data_and_model(args)

    pad_multiple_of = [1, 16, 32, 48, 64, 96, 128]

    # dataframe to store output info
    df = pd.DataFrame([], columns=['multiple', 'accuracy', 'f1-score', 'modified_accuracy', 'overhead'])

    for length in pad_multiple_of:
        print(f'Making predictions for padding to multiple of {length}...')
        X, overhead = pad_sizes(data, 'round', length)

        pred = model.predict(X)

        with open(args['label_file'], 'r') as fh:
            labels = [line.strip() for line in fh.readlines()]

        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average='weighted')
        modified_acc = modified_accuracy_score(y, pred, labels)

        # add values to output dataframe
        df.loc[len(df)] = [length, acc, f1, modified_acc, overhead]

    print(tabulate(df, headers=df.columns))
    return df


def directional_padded_predictions(args):
    '''
    Makes predictions with query sizes padded to nearest multiple of 128 bytes
    and response predictions padded to nearest multiple of 468 bytes.
    Based on RFC 8467: https://www.rfc-editor.org/rfc/rfc8467#ref-NDSS-PADDING.

    args: command line arguments

    Returns: df - dataframe with pad size, eval metrics, and % overhead
    '''
    data, y, model = load_data_and_model(args)

    # dataframe to store output info
    df = pd.DataFrame([], columns=['length', 'accuracy', 'f1-score', 'modified_accuracy', 'overhead'])

    sizes = [(128, 256), (128, 468), (128, 512), (256, 468), (256, 512)]
    for size_pair in sizes:
        print(f'Making predictions for padding to {size_pair}...')
        X, overhead = pad_sizes(data, 'round', size_pair)

        pred = model.predict(X)

        with open(args['label_file'], 'r') as fh:
            labels = [line.strip() for line in fh.readlines()]

        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average='weighted')
        modified_acc = modified_accuracy_score(y, pred, labels)

        # add values to output dataframe
        df.loc[len(df)] = [f'{size_pair}', acc, f1, modified_acc, overhead]

    print(tabulate(df, headers=df.columns))
    return df


def time_delay_predictions(args):
    '''
    Makes predictions with random delay added to times.

    args: command line arguments
    '''
    data, y, model = load_data_and_model(args)

    if args['use_iat']:
        data = make_iat_dataframe(data, with_direction=True)
        X, overhead, base_times, delayed_times = add_time_delay_iat(data, args['delay'], args['direction'])
    else:
        X, overhead, base_times, delayed_times = add_time_delay(data, args['delay'], args['direction'])

    print('Making predictions...')
    pred = model.predict(X)

    with open(args['label_file'], 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='weighted')
    modified_acc = modified_accuracy_score(y, pred, labels)

    print(f'Accuracy: {acc}')
    print(f'F1-Score: {f1}')
    print(f'Modified accuracy: {modified_acc}')

    if overhead:
        print(f'Latency increase from base: {overhead * 100:.2f}%')

    fig, ax = plt.subplots()
    ax.hist(base_times, bins=500, color='b')
    ax.hist(delayed_times, bins=500, color='r', alpha=0.6)
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('count')

    plt.show()

    return modified_acc, overhead


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True, help='Pickle file containing the data')
    parser.add_argument('--model', type=str, required=True, \
                        help="Path to serialized model saved by classifier. DON'T include the '.zip'! ")

    parser.add_argument('--label_file', type=str, required=False, default='collection_scripts/top-1k-curated', \
                        help='File containing list of labels')

    parser.add_argument('--padding', type=str, choices=['max', 'round', 'directional', 'none'], default='none',
                        help='Padding strategy for padding size. Default is no padding')

    parser.add_argument('--delay', type=float, default=None,
                        help='Maximum delay (in seconds) to add to each packet. Default is no delay')

    parser.add_argument('--direction', type=int, choices=[1, -1], default=None,
                        help='Packet direction to apply transformations to. -1 is incoming, 1 is outgoing. Default is both. Only implemented for time delay so far.')

    parser.add_argument('--use_iat', type=bool, default=False,
                        help='Whether to use inter-arrival dataframe or not. Only implemented for time delay so far.')

    parser.add_argument('--filename', type=str, required=False, help='File to save output data to.')
    args = vars(parser.parse_args())

    # Make predictions with desired modifications
    df = None
    if args['padding'] == 'max':
        df = max_padded_predictions(args)
    elif args['padding'] == 'round':
        df = block_padding_predictions(args)
    elif args['padding'] == 'directional':
        df = directional_padded_predictions(args)
    elif args['delay'] is not None:
        time_delay_predictions(args)
    else:
        predict(args)

    # Save output
    if df is not None and args['filename']:
        df.to_csv(args['filename'])
