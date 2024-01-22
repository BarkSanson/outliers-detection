import os

import plotly.express as px


class Plotter:
    def __init__(self, df, plot_path):
        self.plot_path = plot_path
        self.df = df

        os.makedirs(self.plot_path, exist_ok=True)

    def plot_predictions(self, title, target, predictions):
        fig = px.scatter(self.df, x=self.df.index, y=target, color=predictions, opacity=1, title=title)
        fig.write_image(os.path.join(self.plot_path, f'{title}.png'))
        fig.show()

    def plot_data(self, df, target, title, color=None):
        fig = px.line(df, x=self.df.index, y=target, title=title, color=color)
        fig.write_image(os.path.join(self.plot_path, f'{title}.png'))
        fig.show()
