import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--show_plot', default=False, action='store_true')
args = vars(parser.parse_args())

# Load predictions and ground truth from pickle
with open(args['file'], 'rb') as fh:
    d = pickle.load(fh)
    y_pred = d['pred']
    y_true = d['gt']

# Create and display confusion matrix
c_mat = confusion_matrix(y_true, y_pred)

if args['show_plot']:
    matrix_plot = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, include_values=False, cmap='cividis')
    matrix_plot.ax_.set_xticks([i*10 for i in range(10)])
    matrix_plot.ax_.set_yticks([i*10 for i in range(10)])
    plt.show()


# Print metrics
print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
print(f"F1-score: {f1_score(y_true, y_pred, average='weighted')}")
print()

# Print incorrect combinations
nonzero = np.where(c_mat > 0)
incorrect = [(x, y) for (x, y) in zip(nonzero[0], nonzero[1]) if x != y]  # off-diagonal indices

with open('collection_scripts/top-1k-curated', 'r') as fh:
    labels = [line.strip() for line in fh.readlines()]

table = []
for val in incorrect:
    true, predicted = val
    table.append([labels[true], labels[predicted], c_mat[true][predicted]])

print(tabulate(table, headers=['True', 'Predicted', 'Count']))