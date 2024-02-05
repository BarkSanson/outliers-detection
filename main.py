import argparse
import os
from itertools import product

import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

from data_fetch import Requester
from models import Trainer, Serializer
from config import ConfigReader
from data_show import Plotter, Printer
from window_generator import WindowGenerator
from util.dates import parse_dates


def model_scores(y_true, y_pred):
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    return precision, recall, f1


def main():
    args = parse_args()
    (stations, start_date, end_date, data_path,
     results_path, plot_data, config_path, models_path, window_sizes) = (
        args.stations, args.start_date, args.end_date, args.data_path,
        args.results_path, args.plot_data, args.config_path, args.models_path, args.window_sizes)

    config_reader = ConfigReader(config_path)

    models = config_reader.read()

    start_date, end_date = parse_dates(start_date, end_date)

    requester = Requester(stations, data_path, start_date.date(), end_date.date())

    serializer = None
    if models_path:
        serializer = Serializer(models_path)

    dfs = requester.do_request()

    for station, df in dfs.items():
        station_path = os.path.join(results_path, station.replace(' ', ''))
        os.makedirs(station_path, exist_ok=True)

        plots_path = os.path.join(station_path, 'plots')

        printer = Printer(station_path)
        plotter = Plotter(df, plots_path)

        if plot_data:
            plotter.plot_data('value', f'{station} water level', 'quality')

        df = df.drop(columns=['quality'])

        # Mark outliers
        z_scores = (df - df.mean()) / df.std()
        # If z_score is greater than 3, it is an outlier
        df['outlier'] = z_scores.map(lambda x: 1 if abs(x) > 3 else 0)

        results = {}
        for window_size in window_sizes:
            windowed_df = WindowGenerator.split_window(df, window_size)

            trainer = Trainer(windowed_df)
            for model in models:
                param_combinations = list(product(*model.params.values()))

                for params in param_combinations:
                    title = f"{station}_{model.name}_outliers_with_{params}_for_w{window_size}"

                    kwargs = dict(zip(model.params.keys(), params))

                    # Try to get cached model. If none exists, fit and save it if
                    # the user specified a models_path
                    if serializer:
                        try:
                            serialized_model = serializer.load_model(title)
                            print(f"Model {model.name} exists in {models_path} with {kwargs}. Loading...")
                            labels, _ = serialized_model.labels_, serialized_model.decision_scores_
                        except FileNotFoundError:
                            print(f"No model {model.name} with {kwargs} exists in {models_path}. Fitting and saving...")
                            _, labels, _ = trainer.fit_and_save(model.name, serializer, title, **kwargs)
                    else:
                        print(f"No models_path specified. Fitting {model.name} with {kwargs} for {station}...")
                        # If no models_path is specified, just fit the model
                        _, labels, _ = trainer.fit(model.name, **kwargs)

                    precision, recall, f1 = model_scores(windowed_df['outlier'], labels)

                    if window_size not in results:
                        results[window_size] = {}

                    if model.name not in results[window_size]:
                        results[window_size][model.name] = []

                    results[window_size][model.name].append((params, precision, recall, f1))

        for res in results.values():
            for model, data in res.items():
                data.sort(key=lambda x: x[-1], reverse=True)

        best_models = []
        for window_size, res in results.items():
            for model, data in res.items():
                best_models.append((window_size, model, *data[0]))

        best_models.sort(key=lambda x: x[-1], reverse=True)
        printer.print_best_scores(best_models)


def fit_and_save_model(model, trainer, serializer, title, station, **params):
    print(f"Fitting {model.name} with {params} for {station}...")
    trained_model, labels, decision_scores = trainer.fit(model.name, **params)
    print(f"Done fitting {model.name} with {params} for {station}!")
    serializer.save_model(trained_model, title)

    return labels, decision_scores


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("start_date", type=str, help="Start date")
    parser.add_argument("end_date", type=str, help="End date")
    parser.add_argument("-s", "--stations", nargs="+", type=str, help="Station names", required=True)
    parser.add_argument("-d",
                        "--data_path",
                        type=str,
                        help="Path to data folder. Current working directory by default",
                        default=os.getcwd())
    parser.add_argument("-r",
                        "--results_path",
                        help="Path to store results, both plots and tables. Current working directory by default",
                        default=os.getcwd())
    parser.add_argument("-p",
                        "--plot_data",
                        help="Set this to true if you want to plot the data in your browser with interactive plots",
                        action="store_true")
    parser.add_argument("-c",
                        "--config_path",
                        type=str,
                        help="Path to config file",
                        required=True)
    parser.add_argument("-m",
                        "--models_path",
                        type=str,
                        help="Path to models folder, where models will be saved and loaded from."
                             "If not specified, models won't be saved.")
    parser.add_argument("-w",
                        "--window_sizes",
                        type=int,
                        nargs='+',
                        help="Window sizes to use for sliding window. Default only a window size of 5",
                        default=[5])

    return parser.parse_args()


if __name__ == "__main__":
    main()
