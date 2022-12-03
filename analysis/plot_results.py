import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import os
import sys

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

# Add collection_scripts folder to path to import from dataset_utils
classification_path = os.path.join(os.path.dirname(__file__), '../classification')
sys.path.append(classification_path)
from analyze_results import modified_accuracy_score

def plot_estimator_comparison():
    '''
    Creates a bar chart comparing accuracy, F1-score, and modified accuracy
    for different estimator types.
    '''
    sns.set_theme(style='darkgrid')

    # Get list of labels
    with open('collection_scripts/top-1k-curated', 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]
    
    classifiers = ['ridge', 'logisticreg', 'sgd']

    df = pd.DataFrame([], columns=['estimator', 'accuracy', 'f1-score', 'modified_accuracy'])

    for classifier in classifiers:
        pred_file = f'classification/minirocket_full_{classifier}/predictions.pkl'
        with open(pred_file, 'rb') as fh:
            d = pickle.load(fh)
            y_pred = d['pred']
            y_true = d['gt']
        
        df.loc[len(df.index)] = [
            classifier,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='weighted'),
            modified_accuracy_score(y_true, y_pred, labels)
        ]

    # reshape dataframe
    df = pd.melt(df, id_vars='estimator', var_name='metric', value_name='value')

    # plot
    with sns.color_palette('deep', n_colors=3):
        ax = sns.barplot(df, x='estimator', y='value', hue='metric', alpha=0.8)
    
    for container in ax.containers:
        for rect in container.patches:
            height = rect.get_height()
            val = f'{height:.3f}'
            ax.text(rect.get_x() + rect.get_width() / 2, height, val, ha='center', va='bottom', fontsize=8)
    
    plt.title('MINIROCKET estimator comparison', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=10)
    plt.savefig('figs/compare_estimators.png', bbox_inches='tight')
    plt.show()

def plot_classifier_comparison():
    '''
    Creates a bar chart comparing accuracy, F1-score, and modified accuracy
    for different transforms.
    '''
    sns.set_theme(style='darkgrid')

    # Get list of labels
    with open('collection_scripts/top-1k-curated', 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]
    
    classifiers = {
        'rocket': 'classification/rocket_full_default/predictions.pkl',
        'catch-22': 'classification/predictions/predictions_200-class_100-samples_catch22.pkl',
        'minirocket': 'classification/minirocket_full_ridge/predictions.pkl'
    }

    df = pd.DataFrame([], columns=['classifier', 'accuracy', 'f1-score', 'modified_accuracy'])

    for classifier, pred_file in classifiers.items():
        with open(pred_file, 'rb') as fh:
            d = pickle.load(fh)
            y_pred = d['pred']
            y_true = d['gt']
        
        df.loc[len(df.index)] = [
            classifier,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='weighted'),
            modified_accuracy_score(y_true, y_pred, labels)
        ]

    # reshape dataframe
    df = pd.melt(df, id_vars='classifier', var_name='metric', value_name='value')

    # plot
    with sns.color_palette('deep', n_colors=3):
        ax = sns.barplot(df, x='classifier', y='value', hue='metric', alpha=0.8)
    
    for container in ax.containers:
        for rect in container.patches:
            height = rect.get_height()
            val = f'{height:.3f}'
            ax.text(rect.get_x() + rect.get_width() / 2, height, val, ha='center', va='bottom', fontsize=8)
    
    plt.title('Classifier comparison', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=10)
    plt.savefig('figs/compare_classifiers.png', bbox_inches='tight')
    plt.show()

def plot_kernel_comparison():
    '''
    Creates a bar chart comparing accuracy, F1-score, and modified accuracy
    for different number of convolutional kernels with ROCKET.
    '''
    sns.set_theme(style='darkgrid')

    # Get list of labels
    with open('collection_scripts/top-1k-curated', 'r') as fh:
        labels = [line.strip() for line in fh.readlines()]
    
    kernels = {
        10000: 'classification/rocket_full_default',
        5000: 'classification/rocket_full_5000-kernels',
        8000: 'classification/rocket_full_8000-kernels',
        11000: 'classification/rocket_full_11000-kernels'
    }

    df = pd.DataFrame([], columns=['kernel_size', 'accuracy', 'f1-score', 'modified_accuracy'])

    for kernel_size in kernels.keys():
        pred_file = f'{kernels[kernel_size]}/predictions.pkl'
        with open(pred_file, 'rb') as fh:
            d = pickle.load(fh)
            y_pred = d['pred']
            y_true = d['gt']
        
        df.loc[len(df.index)] = [
            kernel_size,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='weighted'),
            modified_accuracy_score(y_true, y_pred, labels)
        ]

    # reshape dataframe
    df = pd.melt(df, id_vars='kernel_size', var_name='metric', value_name='value')

    # plot
    with sns.color_palette('deep', n_colors=3):
        ax = sns.lineplot(df, x='kernel_size', y='value', hue='metric')
    
    plt.title('ROCKET performance with different kernel sizes', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=10)
    plt.savefig('figs/compare_kernels.png', bbox_inches='tight')
    plt.show()

def plot_padding():
    pass

if __name__=='__main__':
    plot_estimator_comparison()
    plot_kernel_comparison()
    plot_classifier_comparison()