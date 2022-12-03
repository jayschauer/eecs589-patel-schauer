import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

# Add classification folder to path to import functions from scripts
classification_path = os.path.join(os.path.dirname(__file__), '../classification')
sys.path.append(classification_path)
from analyze_results import modified_accuracy_score
from make_predictions import max_padded_predictions

DPI = 200

def plot_estimator_comparison():
    '''
    Creates a bar chart comparing accuracy, F1-score, and modified accuracy
    for different estimator types.
    '''
    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)

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

    # Plot bar graph for each estimator with each metric
    with sns.color_palette('deep', n_colors=3):
        ax = sns.barplot(df, x='estimator', y='value', hue='metric', alpha=0.8)
    
    # Add value to top of each bar
    for container in ax.containers:
        for rect in container.patches:
            height = rect.get_height()
            val = f'{height:.3f}'
            ax.text(rect.get_x() + rect.get_width() / 2, height, val, ha='center', va='bottom', fontsize=8)
    
    # Labels and stuff
    ax.set_xlabel('Estimator')
    ax.set_ylabel('Value')
    plt.tick_params(axis='both', which='major')
    sns.move_legend(ax, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)
    plt.savefig('figs/compare_estimators.png', bbox_inches='tight', dpi=DPI)
    plt.show()

def plot_classifier_comparison():
    '''
    Creates a bar chart comparing accuracy, F1-score, and modified accuracy
    for different transforms.
    '''
    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)

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

    # Create bar graph for each classifier with each metric
    with sns.color_palette('deep', n_colors=3):
        ax = sns.barplot(df, x='classifier', y='value', hue='metric', alpha=0.8)
    
    # Add value to top of each bar
    for container in ax.containers:
        for rect in container.patches:
            height = rect.get_height()
            val = f'{height:.3f}'
            ax.text(rect.get_x() + rect.get_width() / 2, height, val, ha='center', va='bottom', fontsize=8)
    
    # Labels and stuff
    ax.set_xlabel('Classifier')
    ax.set_ylabel('Value')
    plt.tick_params(axis='both', which='major')
    sns.move_legend(ax, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)
    plt.savefig('figs/compare_classifiers.png', bbox_inches='tight', dpi=DPI)
    plt.show()

def plot_kernel_comparison():
    '''
    Creates a bar chart comparing accuracy, F1-score, and modified accuracy
    for different number of convolutional kernels with ROCKET.
    '''
    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)

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
        ax = sns.lineplot(df, x='kernel_size', y='value', hue='metric', marker='o')

    # Labels and stuff
    ax.set_xlabel('Kernel size')
    ax.set_ylabel('Value')
    plt.tick_params(axis='both', which='major')
    sns.move_legend(ax, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)
    plt.savefig('figs/compare_kernels.png', bbox_inches='tight', dpi=DPI)
    plt.show()

def plot_max_padding():
    '''
    Creates a chart showing the accuracy for different sizes of max padding.
    '''
    df = pd.read_csv('classification/max_pad.csv')

    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    palette = sns.color_palette('deep', n_colors=2)

    # Plot modified accuracy vs pad size on one graph
    sns.lineplot(df, x='size', y='modified_acc', ax=ax1, marker='o', color=palette[0])

    # Plot overhead vs pad size on other graph
    sns.lineplot(df, x='size', y='overhead', ax=ax2, marker='o', color=palette[1])

    handles=[Line2D([], [], marker='o', color=palette[0], label='Modified Accuracy'), Line2D([], [], marker='o', color=palette[1], label='Overhead')]

    # Labels and stuff
    ax1.set_xlabel('Pad size (bytes)')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Overhead (%)')

    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax1.set_ylim(-0.02, 1.02)
    ax2.set_ylim(-0.04, 2.04)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)
    ax2.set_yticks([tick * 2 for tick in ticks])
    ax2.set_yticklabels([f'{tick * 200:.0f}' for tick in ticks])

    ax1.legend(handles=handles)
    sns.move_legend(ax1, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=2, title=None, frameon=False)

    fig.tight_layout()
    plt.savefig('figs/max_padding.png', bbox_inches='tight', dpi=DPI)
    plt.show()

def plot_adaptive_padding():
    '''
    Creates a chart showing the accuracy for different sizes of adaptive padding.
    '''
    df = pd.read_csv('classification/adaptive_pad.csv')

    sns.set_theme(style='darkgrid')
    sns.set(font_scale=0.8)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    palette = sns.color_palette('deep', n_colors=2)

    # Plot modified accuracy vs pad size on one graph
    sns.lineplot(df, x='multiple', y='modified_accuracy', ax=ax1, marker='o', color=palette[0])

    # Plot overhead vs pad size on other graph
    sns.lineplot(df, x='multiple', y='overhead', ax=ax2, marker='o', color=palette[1])

    handles=[Line2D([], [], marker='o', color=palette[0], label='Modified Accuracy'), Line2D([], [], marker='o', color=palette[1], label='Overhead')]

    # Labels and stuff
    ax1.set_xlabel('Padding multiple (bytes)')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Overhead (%)')

    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax1.set_ylim(-0.02, 1.02)
    ax2.set_ylim(-0.04, 2.04)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(ticks)
    ax2.set_yticks([tick * 2 for tick in ticks])
    ax2.set_yticklabels([f'{tick * 200:.0f}' for tick in ticks])

    ax1.legend(handles=handles)
    sns.move_legend(ax1, 'upper center', bbox_to_anchor=(.5, 1.1), ncol=2, title=None, frameon=False)

    fig.tight_layout()
    plt.savefig('figs/adaptive_padding.png', bbox_inches='tight', dpi=DPI)
    plt.show()


if __name__=='__main__':
    # plot_estimator_comparison()
    # plot_kernel_comparison()
    # plot_classifier_comparison()

    # plot_max_padding()
    plot_adaptive_padding()