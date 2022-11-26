import argparse
import pickle

from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

from analyze_results import modified_accuracy_score

from sktime.classification import BaseClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file', type=str, required=True, help='predictions file to load')
    parser.add_argument('--model_file', type=str, required=True, help='model file to load')
    parser.add_argument('--label_file', type=str, required=True, help='file containing list of labels')
    args = vars(parser.parse_args())

    # Load predictions and ground truth from pickle
    with open(args['predictions_file'], 'rb') as fh:
        d = pickle.load(fh)
        X_test = d['X']
        y_true = d['gt']

    # Get list of labels
    with open(args['label_file'], 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]

    # load model
    clf = BaseClassifier.load_from_path(args['model_file'])

    # combinations has every combination of time, packet size, and direction.
    # 'tsd' = time, packet size, and direction are included
    # 'ts' = time and packet size are included, direction is set to zero
    combos = {'tsd': X_test}
    # TODO: don't make 6 copies of the dataset, uses too much memory. Just do one at a time.
    # time and packet size non-zero
    combos['ts'] = combos['tsd'].copy()
    combos['ts']['direction'].values[:] = 0
    # time and direction non-zero
    combos['td'] = combos['tsd'].copy()
    combos['td']['packet_size'].values[:] = 0
    # packet size and direction non-zero
    combos['sd'] = combos['tsd'].copy()
    combos['sd']['time'].values[:] = 0
    # only time non-zero
    combos['t'] = combos['ts'].copy()
    combos['t']['packet_size'].values[:] = 0
    # only packet size non-zero
    combos['s'] = combos['ts'].copy()
    combos['s']['time'].values[:] = 0
    # only direction non-zero
    combos['d'] = combos['sd'].copy()
    combos['d']['packet_size'].values[:] = 0

    table = []
    for combo, X in combos.items():
        # Print metrics
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        modified_accuracy = modified_accuracy_score(y_true, y_pred, labels)
        table.append([combo, accuracy, f1, modified_accuracy])

    print(tabulate(table, headers=['Combo', 'Accuracy', 'F1', 'Modified Accuracy']))
