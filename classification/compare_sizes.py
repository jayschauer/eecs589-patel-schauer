import argparse
import pickle
import os
import sys

from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

import classify

from analyze_results import modified_accuracy_score

# Add collection_scripts folder to path to import from dataset_utils
collection_path = os.path.join(os.path.dirname(__file__), '../collection_scripts')
sys.path.append(collection_path)
from dataset_utils import load_data

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
    # maximum-length padding
    max_length = max(map(lambda sample: len(sample), data))
    X_train_base, X_test_base, y_train, y_test = classify.prep_data(data, labels)

    methods = ['rocket', 'catch22']
    for method in methods:
        table = []
        for combo, filter in combos.items():
            X_train = filter(X_train_base)
            X_test = filter(X_test_base)
            print('Loading model')
            clf = classify.load_model(method, max_length)
            print('Starting training...')
            clf.fit(X_train, y_train)
            print('Making predictions...')
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            modified_accuracy = modified_accuracy_score(y_test, y_pred, domains)
            table.append([combo, accuracy, f1, modified_accuracy])
        table_string = tabulate(table, headers=['Combo', 'Accuracy', 'F1', 'Modified Accuracy'])
        with open(method + '.txt', 'w') as table_file:
            table_file.write(table_string)
        print(table_string)
