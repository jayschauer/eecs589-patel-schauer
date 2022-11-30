import argparse
import os
import sys

import math
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sktime.base import load
from sktime import datatypes
from statsmodels.distributions.empirical_distribution import ECDF
from tabulate import tabulate

from classify import make_dataframe

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
    model = load(args['model'])

    return X_test, y, model

def pad_sizes(data, strategy, length):
    '''
    Creates a dataframe with sizes padded based on specified strategy.

    data: list of samples where each sample is a dictionary of time: size pairs
    strategy: padding method (either 'max' or 'round')
    length: length to pad to if max padding, or value to pad to multiple of if rounding 
            (e.g. if 32, pads to nearest multiple of 32)

    Returns: (df, overhead) - df is pd-multiindex dataframe, overhead is percent
    overhead compared to base
    '''

    # default: no padding
    if strategy not in ['max', 'round']:
        return make_dataframe(data), 0

    cols = ['timepoints', 'packet_size']
    padded, base = 0, 0
    Xlist = []

    # max or round padding
    for series in data:
        val_list = []
        for time, datum in series.items():
            if strategy == 'max':
                padded_val = max(length, datum[0]) # higher of length and existing value
            else:
                padded_val = math.ceil(datum[0] / length) * length  # round up to nearest multiple of length

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
            outgoing packets (not implemented yet)

    Returns: (df, overhead) - df is pd-multiindex dataframe, overhead is percent
    overhead compared to base
    '''

    cols = ['timepoints', 'packet_size']
    base_time, delayed_time = 0, 0
    Xlist = []

    for series in data:
        val_list = []

        # create list of cumulative delays for efficiency
        delays = [ 0 ]  # first time point always arrives at time 0
        for i in range(1, len(series)):
            delays.append(delays[i - 1] + (random.random() * max_delay))
    
        # construct dataframe with new times
        for i, (time, datum) in enumerate(series.items()):
            val_list.append([time + delays[i], datum[0]])

        final_time = list(series.keys())[-1]
        base_time += final_time
        delayed_time += final_time + delays[-1]

        Xlist.append(pd.DataFrame(val_list, columns=cols))


    overhead = (delayed_time - base_time) / base_time
    return datatypes.convert_to(Xlist, to_type='pd-multiindex'), overhead

def predict(args):
    '''
    Basic prediction with no padding.
    '''
    X, y, model = load_data_and_model(args)
    X = make_dataframe(X) # no modification

    print('Making predictions...')
    pred = model.predict(X)

    acc = accuracy_score(pred, y)
    print(f'Accuracy: {acc}')

def max_padded_predictions(args):
    '''
    Makes predictions with sizes padded to 25th, 50th, 75th, 90th, and 99th percentile
    of sizes in data.

    args: command line arguments
    '''
    X, y, model = load_data_and_model(args)

    # get sizes to determine percentiles
    full_df = make_dataframe(data)
    cdf_func = ECDF(full_df['packet_size'])
    cdf = cdf_func(np.arange(max(full_df['packet_size'])))

    percentiles = [0.25, 0.5, 0.75, 0.9, 0.99]
    values = [ np.where(cdf > ptile)[0][0] for ptile in percentiles ]

    # dataframe to store output info
    df = pd.DataFrame([], columns=['percentile', 'size', 'accuracy', 'overhead'])

    for ptile, value in zip(percentiles, values):
        print(f'Making predictions for {int(ptile * 100)}th percentile...')
        X, overhead = pad_sizes(X, 'max', value)

        pred = model.predict(X) 
        acc = accuracy_score(pred, y)

        # add values to output dataframe
        df.loc[len(df)] = [int(ptile * 100), value, acc, overhead]

    print(tabulate(df, headers=df.columns))

def rounded_predictions(args):
    X, y, model = load_data_and_model(args)

    pad_multiple_of = [1, 16, 32, 48, 64, 96, 128]

    # dataframe to store output info
    df = pd.DataFrame([], columns=['multiple', 'accuracy', 'overhead'])

    for length in pad_multiple_of:
        print(f'Making predictions for padding to multiple of {length}...')
        X, overhead = pad_sizes(X, 'round', length)

        pred = model.predict(X) 
        acc = accuracy_score(pred, y)

        # add values to output dataframe
        df.loc[len(df)] = [length, acc, overhead]

    print(tabulate(df, headers=df.columns))

def time_delay_predictions(args):
    data, labels, model = load_data_and_model(args)

    X, overhead = add_time_delay(data, args['delay'])

    print('Making predictions...')
    pred = model.predict(X)

    acc = accuracy_score(labels, pred)
    print(f'Accuracy: {acc}')

    print(f'Latency increase from base: {overhead * 100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Pickle file containing the data')
    parser.add_argument('--model', type=str, required=True, \
        help="Path to serialized model saved by classifier. DON'T include the '.zip'! ")
    parser.add_argument('--padding', type=str, choices=['max', 'round', 'none'], default='none',
        help='Padding strategy for padding size. Default is no padding')
    parser.add_argument('--delay', type=float, default=None,
        help='Maximum delay (in seconds) to add to each packet. Default is no delay')
    args = vars(parser.parse_args())

    if args['padding'] == 'max':
        max_padded_predictions(args)
    elif args['padding'] == 'round':
        rounded_predictions(args)
    elif args['delay'] is not None:
        time_delay_predictions(args)
    else:
        predict(args)
