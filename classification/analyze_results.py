import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

def is_same_domain(str1, str2):
    '''
    Checks if two domain names have the same first component, e.g.
    google.com and google.co.uk.

    str1, str2: domains to compare.

    Returns: True if domains are the same, False otherwise.
    '''
    # Hardcoded since this is the only case like this I could see 
    # in the mispredictions
    if (str1 == 'reddit.com' and str2 == 'redd.it') or \
        (str1 == 'redd.it' and str2 == 'reddit.com'):
        return True

    # Split at . and compare first piece
    l1, l2 = str1.split('.'), str2.split('.')
    return len(l1) > 1 and len(l2) > 1 and l1[0] == l2[0]

def modified_accuracy_score(y_true, y_pred, labels):
    '''
    Computes the accuracy score with similar domains (determined by is_same_domain)
    treated as the same class.

    y_true: list of ground truth labels.
    y_pred: list of model predictions.
    labels: list of class labels where each element of y_true and y_pred is an index into that list.

    Returns: fraction of correctly classified labels.
    '''
    modified_pred = []
    for i in range(len(y_true)):
        # if domains are the same, treat as if it classified correctly
        if (is_same_domain(labels[y_true[i]], labels[y_pred[i]])):
            modified_pred.append(y_true[i])
        else:
            modified_pred.append(y_pred[i])

    return accuracy_score(y_true, modified_pred)

def main(args):
    # Load predictions and ground truth from pickle
    with open(args['file'], 'rb') as fh:
        d = pickle.load(fh)
        y_pred = d['pred']
        y_true = d['gt']

    # Create and display confusion matrix
    c_mat = confusion_matrix(y_true, y_pred)

    if args['show_plot']:
        matrix_plot = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, include_values=False, cmap='cividis')
        ticks = [i*10 for i in range(c_mat.shape[0] // 10 + 1)]
        matrix_plot.ax_.set_xticks(ticks)
        matrix_plot.ax_.set_yticks(ticks)
        plt.show()

    # Get list of labels
    with open(args['label_file'], 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]

    # Print metrics
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    print(f"F1-score: {f1_score(y_true, y_pred, average='weighted')}")
    print(f'Modified accuracy: {modified_accuracy_score(y_true, y_pred, labels)}')
    print()

    # Print incorrect combinations
    nonzero = np.where(c_mat > 0)
    incorrect = [(x, y) for (x, y) in zip(nonzero[0], nonzero[1]) if x != y]  # off-diagonal indices

    table = []
    for val in incorrect:
        true, predicted = val
        table.append([f'{labels[true]} ({true})', f'{labels[predicted]} ({predicted})', c_mat[true][predicted]])

    print(tabulate(table, headers=['True', 'Predicted', 'Count']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='predictions file to load')
    parser.add_argument('--show_plot', default=False, action='store_true', help='show confusion matrix')
    parser.add_argument('--label_file', type=str, required=True, help='file containing list of labels')
    args = vars(parser.parse_args())

    main(args)