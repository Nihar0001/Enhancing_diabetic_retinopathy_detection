# scripts/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import normalize
import re

def plot_normalized_confusion_matrix(cm, class_names, title):
    """
    Plots a normalized confusion matrix using seaborn's heatmap.
    """
    cm_normalized = normalize(cm, axis=1, norm='l1')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_f1_scores(report_content, class_names, title):
    """
    Plots a bar chart of F1-scores for each class from a classification report.
    """
    metrics_lines = re.findall(r'^(\s*\d+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+)', report_content, re.MULTILINE)
    
    f1_scores = {}
    
    for line in metrics_lines:
        line = line.strip()
        parts = line.split()
        class_name = parts[0]
        f1_score = float(parts[3])
        f1_scores[class_name] = f1_score
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette='viridis')
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("F1-Score")
    plt.show()

# This function is not called by analyze_results.py in this version, but it is useful for combined charts.
def create_combined_bar_chart(all_metrics, metric_name, title):
    """Creates a bar chart to compare a specific metric across all models."""
    model_names = list(all_metrics.keys())
    
    all_scores = {}
    for class_name in ['0', '1', '2', '3', '4']:
        all_scores[class_name] = [all_metrics[model][class_name][metric_name] for model in model_names]

    x = np.arange(len(model_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i, class_name in enumerate(all_scores.keys()):
        offset = width * (i - (len(all_scores) - 1) / 2)
        ax.bar(x + offset, all_scores[class_name], width, label=f'Class {class_name}')

    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Classes')
    fig.tight_layout()
    plt.show()