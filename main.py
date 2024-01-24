import argparse
import os
import sys
import datetime
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import tabulate
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix

from data_fetch import Requester
from plotting import Plotter
from models import Trainer
from config import ConfigReader


def custom_accuracy_score(y_true, y_pred):
    # Count the number of outliers that were correctly predicted
    correct_outliers = len(y_true[(y_true == -1) & (y_pred == -1)])

    score = correct_outliers / len(y_true[y_true == -1])

    return score


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

        plotter = Plotter(df, station_path)

        if plot_data:
            plotter.plot_data('value', f'{station} water level', 'quality')

        df = df.drop(columns=['quality'])

        # Mark outliers
        z_scores = (df - df.mean()) / df.std()

        # If z_score is greater than 3, it is an outlier, but we want to mark the outliers as -1 and
        # the inliers as 1
        df['outlier'] = z_scores.map(lambda x: -1 if abs(x) > 3 else 1)

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

                accuracy = custom_accuracy_score(df['outlier'], labels)

                if model.name not in results:
                    results[model.name] = []

                results[model.name].append((labels, *params, accuracy))

        # Create a table with the accuracy of each model
        for model, res in results.items():
            accuracy_path = os.path.join(station_path, "accuracies")
            os.makedirs(accuracy_path, exist_ok=True)

            res.sort(key=lambda x: x[-1], reverse=True)

            current_model = filter(lambda x: x.name == model, models)
            model_params = next(current_model).params

            params_names = list(model_params.keys())

            # Map the results to a list of lists, where each list is a row in the table
            data = list(map(lambda x: [*x[1:-1], x[-1]], res))

            with open(os.path.join(accuracy_path, f'{model}.txt'), 'w') as f:
                # Write a table that has the accuracy and the parameters used, but not the labels
                f.write(tabulate.tabulate(data, headers=[*params_names, 'Accuracy'], tablefmt='orgtbl'))

        # Take the best model of each type and plot their confusion matrix
        for model, res in results.items():
            confusion_matrices_path = os.path.join(station_path, "confusion_matrices")
            os.makedirs(confusion_matrices_path, exist_ok=True)
            best_model = res[0]

            df[f'best_{model}_predictions'] = best_model[0]

            predicted = list(map(lambda x: 'Outlier' if x == -1 else 'Inlier', best_model[0]))
            actual = df['outlier'].map(lambda x: 'Outlier' if x == -1 else 'Inlier')

            cf_matrix = confusion_matrix(actual, predicted, labels=["Outlier", "Inlier"])
            sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Outlier', 'Inlier'],
                        yticklabels=['Outlier', 'Inlier'])
            plt.savefig(os.path.join(confusion_matrices_path, f'{station} best {model} confusion matrix.png'))
            plt.clf()

        intervals = pd.date_range(start_date, end_date, freq='1W')

        df['all_models_agree'] = df[[f'best_{model.name}_predictions' for model in models]].apply(
            lambda x: 1 if all(x == -1) else 0, axis=1)

        time_series_path = os.path.join(station_path, "time_series")
        os.makedirs(time_series_path, exist_ok=True)

        start = start_date
        for i in range(len(intervals) - 1):
            interval = df[start:intervals[i]]

            agrees = interval[interval['all_models_agree'] == 1]

            sns.lineplot(data=interval, x=interval.index, y='value', color="green")

            colors = ["orange", "blue"]
            for model, color in zip(models, colors):
                outliers_model = interval[interval[f'best_{model.name}_predictions'] == -1]
                sns.scatterplot(
                    data=outliers_model,
                    x=outliers_model.index,
                    y='value',
                    hue=f'best_{model.name}_predictions',
                    palette={-1: color}
                )

            sns.scatterplot(data=agrees, x=agrees.index, y='value', hue='all_models_agree', palette={1: "red"})

            plt.savefig(
                os.path.join(time_series_path, f'{station} {start.date()} to {intervals[i].date()} time series.png'))
            start = intervals[i]

            plt.clf()


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
