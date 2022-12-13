import argparse
import copy
import pickle
import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Add collection_scripts folder to path to import from dataset_utils
collection_path = os.path.join(os.path.dirname(__file__), '../../collection_scripts')
sys.path.append(collection_path)
from dataset_utils import load_data

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from analyze_results import modified_accuracy_score
from classify import make_directional_dataframe, get_classifier

combos = {
    'ts-out': lambda X, y: ((X[['time', 'packet_size']])[X['direction'] > 0], y),
    'ts-in': lambda X, y: ((X[['time', 'packet_size']])[X['direction'] < 0], y),
    't-out': lambda X, y: ((X[['time']])[X['direction'] > 0], y),
    't-in': lambda X, y: ((X[['time']])[X['direction'] < 0], y),
    's-out': lambda X, y: ((X[['packet_size']])[X['direction'] > 0], y),
    's-in': lambda X, y: ((X[['packet_size']])[X['direction'] < 0], y),
    'd-out': lambda X, y: ((X[['direction']])[X['direction'] > 0], y),
    'd-in': lambda X, y: ((X[['direction']])[X['direction'] < 0], y),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='data file to load')
    parser.add_argument('--domain_file', type=str, required=True, help='file containing list of domains')

    args = vars(parser.parse_args())

    # Get list of domains
    with open(args['domain_file'], 'r') as fh:
        domains = [line.strip() for line in fh.readlines()]

    # Load predictions and ground truth from pickle
    print('Loading data...')
    data, labels = load_data(args['data_file'])
    # remove series and labels that don't have both outgoing and incoming
    # - causes issues when we filter by direction if a sample has zero points
    data_filtered = []
    labels_filtered = []
    for d,l in zip(data, labels):
        out_packet = False
        in_packet = False
        for p in d.values():
            out_packet = out_packet or p[1] > 0
            in_packet = in_packet or p[1] < 0
            if out_packet and in_packet:
                break
        if out_packet and in_packet:
            data_filtered.append(d)
            labels_filtered.append(l)

    # maximum-length padding
    max_length = max(map(lambda sample: len(sample), data_filtered))
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        data_filtered, labels_filtered, random_state=589, test_size=0.25, shuffle=True
    )
    X_train_base, X_test_base = make_directional_dataframe(X_train_base), make_directional_dataframe(X_test_base)
    y_train_base, y_test_base = pd.Series(y_train_base), pd.Series(y_test_base)

    methods = ['catch22']
    for method in methods:
        table = []
        for combo, filter in combos.items():
            X_train, y_train = filter(X_train_base, y_train_base)
            X_test, y_test = filter(X_test_base, y_test_base)
            # print(X_train)
            # print(y_train)
            # print(X_test)
            # print(y_test)
            print('Loading model')
            clf = get_classifier(method, max_length)
            print('Starting training...')
            clf.fit(X_train, y_train)
            print('Making predictions...')
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            modified_accuracy = modified_accuracy_score(y_test, y_pred, domains)
            table.append([combo, accuracy, f1, modified_accuracy])
        table_string = tabulate(table, headers=['Combo', 'Accuracy', 'F1', 'Modified Accuracy'])
        with open(method + '_2.txt', 'w') as table_file:
            table_file.write(table_string)
        print(table_string)