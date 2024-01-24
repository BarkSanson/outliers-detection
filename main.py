import argparse
import os
import sys
import datetime
from itertools import product

import matplotlib.pyplot as plt
import tabulate
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix

from data_fetch import Requester
from plotting import Plotter
from models import Trainer
from config import ConfigReader


def main():
    args = parse_args()
    stations, start_date, end_date, data_path, plot_path, plot_data, config_path = (
        args.stations, args.start_date, args.end_date, args.data_path, args.plot_path, args.plot_data, args.config_path)

    config_reader = ConfigReader(config_path)

    models = config_reader.read()

    start_date, end_date = parse_dates(start_date, end_date)

    requester = Requester(stations, data_path, start_date.date(), end_date.date())

    dfs = requester.do_request()

    for station, df in dfs.items():
        station_path = os.path.join(plot_path, station.replace(' ', ''))
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
                df[f'predicted_outlier_{model.name}_{params}'] = labels

                accuracy = accuracy_score(df['outlier'], labels)

                if model.name not in results:
                    results[model.name] = []

                results[model.name].append((model.name, params, labels, accuracy))

        for model, results in results.items():
            results.sort(key=lambda x: x[2], reverse=True)

            with open(os.path.join(station_path, 'results.txt'), 'w') as f:
                f.write(tabulate.tabulate(results, headers=['Model', 'Params', 'Accuracy']))

        # Take the best model of each type and plot their confusion matrix
        for model, results in results.items():
            best_model = results[0]

            cf_matrix = confusion_matrix(best_model[2], df['outlier'])
            sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
            plt.savefig(os.path.join(station_path, f'{station} {model} confusion matrix.png'))


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
    parser.add_argument("-p",
                        "--plot_path",
                        help="Path to store plots. Current working directory by default",
                        default=os.getcwd())
    parser.add_argument("-P",
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
