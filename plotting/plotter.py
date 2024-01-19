import plotly.express as px


class Plotter:
    def __init__(self, df, plot_path, predictions=None, model=None):
        self.model = model
        self.plot_path = plot_path
        self.df = df
        self.predictions = predictions

    def plot_predictions(self, target):
        if self.model is None:
            print('No model to predict with')
            return

        predictions = self.predictions if self.predictions is not None else self.model.predict(self.df)
        fig = px.scatter(self.df, x=self.df.index, y=target, color=predictions, opacity=1)
        fig.show()

    def plot_data(self, df, target, title):
        fig = px.line(df, x=self.df.index, y=target, title=title)
        fig.write_image(f'{title}.svg')
