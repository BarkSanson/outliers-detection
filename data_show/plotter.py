import os

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px


class Plotter:
    def __init__(self, df, plot_path):
        self._plot_path = plot_path
        self._df = df

        # Change sns default style
        os.makedirs(self._plot_path, exist_ok=True)

    def plot_predictions(self, title, target, predictions):
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=self._df, x=self._df.index, y=target)
        sns.scatterplot(data=predictions, x=predictions.index, y=target, color='red', markers='o', label='Outliers')
        plt.title(title)
        plt.show()

    def plot_data(self, target, title, color=None):
        fig = px.line(self._df, x=self._df.index, y=target, title=title, color=color)
        fig.write_image(os.path.join(self._plot_path, f'{title}.png'))

        fig.show()

    def plot_confusion_matrix(self, station, model, confusion_matrix):
        confusion_matrices_path = os.path.join(self._plot_path, "confusion_matrices")
        os.makedirs(confusion_matrices_path, exist_ok=True)

        sns.heatmap(confusion_matrix,
                    annot=True,
                    cmap='Blues',
                    fmt='g',
                    xticklabels=['Outlier', 'Inlier'],
                    yticklabels=['Outlier', 'Inlier'])

        plt.savefig(os.path.join(confusion_matrices_path, f'{station} best {model} confusion matrix.png'))
        plt.clf()

    def plot_coincidences(self, df, station, models, start_date, intervals):
        time_series_path = os.path.join(self._plot_path, "time_series")
        os.makedirs(time_series_path, exist_ok=True)

        start = start_date
        for i in range(len(intervals) - 1):
            interval = df[start:intervals[i]]

            agrees = interval[interval['all_models_agree'] == 1]

            sns.lineplot(data=interval, x=interval.index, y='value', color="green")

            colors = ["orange", "blue", "purple"]
            for model, color in zip(models, colors):
                outliers_model = interval[interval[f'best_{model.name}_predictions'] == 1]
                sns.scatterplot(
                    data=outliers_model,
                    x=outliers_model.index,
                    y='value',
                    hue=f'best_{model.name}_predictions',
                    palette={1: color}
                )

            sns.scatterplot(data=agrees, x=agrees.index, y='value', hue='all_models_agree', palette={1: "red"})

            plt.savefig(
                os.path.join(time_series_path, f'{station} {start.date()} to {intervals[i].date()} time series.png'))
            start = intervals[i]

            plt.clf()

    def plot_model_outliers(self, df, outliers, model, params):
        sns.lineplot(data=df, x=df.index, y='value')

        sns.scatterplot(data=outliers, x=outliers.index, y='value', color='red', markers='o', label='Outliers')

        plt.savefig(os.path.join(self._plot_path, f'{model}_{params}_outliers.png'))
        plt.clf()
