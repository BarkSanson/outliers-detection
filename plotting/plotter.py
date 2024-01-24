import os

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go


class Plotter:
    def __init__(self, df, plot_path):
        self.plot_path = plot_path
        self.df = df

        os.makedirs(self.plot_path, exist_ok=True)

    def plot_predictions(self, title, target, predictions):
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=self.df, x=self.df.index, y=target)
        sns.scatterplot(data=predictions, x=predictions.index, y=target, color='red', markers='o', label='Outliers')
        plt.title(title)
        plt.show()

    def plot_data(self, target, title, color=None):
        fig = px.line(self.df, x=self.df.index, y=target, title=title, color=color)
        fig.write_image(os.path.join(self.plot_path, f'{title}.png'))

        fig.show()
