import argparse
import os
import sys
import datetime
from itertools import product

import pandas as pd

from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix

from data_fetch import Requester
from models import Trainer
from config import ConfigReader
from data_show import Plotter, Printer


def scores(y_true, y_pred):
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    return precision, recall, f1


def main():
    args = parse_args()
    stations, start_date, end_date, data_path, results_path, plot_data, config_path = (
        args.stations, args.start_date, args.end_date, args.data_path, args.results_path, args.plot_data,
        args.config_path)

    config_reader = ConfigReader(config_path)

    models = config_reader.read()

    start_date, end_date = parse_dates(start_date, end_date)

    requester = Requester(stations, data_path, start_date.date(), end_date.date())

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

        # If z_score is greater than 3, it is an outlier, but we want to mark the outliers as -1 and
        # the inliers as 1
        df['outlier'] = z_scores.map(lambda x: 1 if abs(x) > 3 else 0)

        trainer = Trainer(df)
        results = {}
        for model in models:
            param_combinations = list(product(*model.params.values()))

            for params in param_combinations:
                title = f"{station} {model.name} outliers with {params}"
                if os.path.exists(os.path.join(station_path, f'{title}.png')):
                    print(f'{title} already exists')
                    continue

                kwargs = dict(zip(model.params.keys(), params))

                print(f"Fitting {model.name} with {kwargs} for {station}...")
                _, labels = trainer.fit(model.name, **kwargs)
                print(f"Done fitting {model.name} with {kwargs} for {station}!")

                precision, recall, f1 = scores(df['outlier'], labels)

                if model.name not in results:
                    results[model.name] = []

                results[model.name].append((labels, *params, precision, recall, f1))

        printer.print_scores(models, results)

        # Take the best model of each type and plot their confusion matrix
        for model, res in results.items():
            best_model = res[0]

            df[f'best_{model}_predictions'] = best_model[0]

            predicted = list(map(lambda x: 'Outlier' if x == 1 else 'Inlier', best_model[0]))
            actual = df['outlier'].map(lambda x: 'Outlier' if x == 1 else 'Inlier')

            cf_matrix = confusion_matrix(actual, predicted, labels=["Outlier", "Inlier"])

            plotter.plot_confusion_matrix(station, model, cf_matrix)

        df['all_models_agree'] = df[[f'best_{model.name}_predictions' for model in models]].apply(
            lambda x: 1 if all(x == 1) else 0, axis=1)

        intervals = pd.date_range(start_date, end_date, freq='1W')
        plotter.plot_coincidences(df, station, models, start_date, intervals)


def parse_dates(start_date, end_date):
    try:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        print('Invalid date format. Please use YYYY-MM-DD.')
        sys.exit(1)

    # Check if start date is before today
    if start_date > datetime.datetime.today():
        print('Start date must be before today.')
        sys.exit(1)

    # Check if end date is before today
    if end_date >= datetime.datetime.today():
        print('End date must be before today.')
        sys.exit(1)

    # Check if start date is before end date
    if start_date > end_date:
        print('Start date must be before end date.')
        sys.exit(1)

    return start_date, end_date


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

    return parser.parse_args()


if __name__ == "__main__":
    main()
