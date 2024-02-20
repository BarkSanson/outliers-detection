import os

import seaborn as sns
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, df, plot_path):
        self._plot_path = plot_path
        self._df = df

        os.makedirs(self._plot_path, exist_ok=True)

    def plot_predictions(self, title, target, predictions):
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=self._df, x=self._df.index, y=target)
        sns.scatterplot(data=predictions, x=predictions.index, y=target, color='red', markers='o', label='Outliers')
        plt.title(title)
        plt.show()

    def plot_roc_auc(self, fpr, tpr, auc, title):
        plt.figure(figsize=(20, 10))
        sns.lineplot(x=fpr, y=tpr, label=f'ROC curve (area = {auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self._plot_path, f"{title}.png"))
