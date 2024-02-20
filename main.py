import argparse
import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc

from active_outlier_detection.detection_pipeline import KSBLWINIForest
from data_fetch import Requester
from data_show import Plotter
from utils import dates


def main():
    args = parse_args()
    (stations, start_date, end_date, data_path,
     results_path, plot_data, config_path, models_path, window_sizes, n_trees) = (
        args.stations, args.start_date, args.end_date, args.data_path,
        args.results_path, args.plot_data, args.config_path, args.models_path, args.window_sizes, args.n_trees)

    start_date, end_date = dates.parse_dates(start_date, end_date)

    requester = Requester(stations, data_path, start_date.date(), end_date.date())

    dfs = requester.do_request()

    for station, df in dfs.items():
        station_path = os.path.join(results_path, station.replace(' ', ''))
        os.makedirs(station_path, exist_ok=True)

        plots_path = os.path.join(station_path, 'plots')

        z_score = (df['value'] - df['value'].mean()) / df['value'].std()
        df['outlier'] = z_score.abs() >= 3

        plotter = Plotter(df, plots_path)

        df = df.drop(columns=['quality'])

        res = pd.DataFrame()
        for window_size in window_sizes:
            ksblwin_iforest = KSBLWINIForest(window_size=window_size)

            score_list = np.array([])
            labels_list = np.array([])
            for dt, element in df.iterrows():
                scores, labels = ksblwin_iforest.run_pipe(element['value'])

                if scores is not None:
                    labels_list = np.concatenate([labels_list, labels])
                    score_list = np.concatenate([score_list, scores])

            scores_series = pd.Series(score_list, index=df.index[:len(score_list)], name="scores")
            labels_series = pd.Series(labels_list, index=df.index[:len(labels_list)], name="labels")
            res[f"score_w{window_size}"] = scores_series
            res[f"label_w{window_size}"] = labels_series

            fpr, tpr, thresholds = \
                roc_curve(df[:len(score_list)]['outlier'], res[:len(score_list)][f"score_w{window_size}"])
            roc_auc = auc(fpr, tpr)
            plotter.plot_roc_auc(fpr, tpr, roc_auc, f"{station} ROC AUC for window size {window_size}")




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
                        required=False)
    parser.add_argument("-m",
                        "--models_path",
                        type=str,
                        help="Path to models folder, where models will be saved and loaded from."
                             "If not specified, models won't be saved.")
    parser.add_argument("-w",
                        "--window_sizes",
                        type=int,
                        nargs="*",
                        help="Window sizes to use for block window. Default is 50",
                        default=[50])
    parser.add_argument("-t",
                        "--n_trees",
                        type=int,
                        nargs="*",
                        help="Number of trees in the isolation forest. Default is 100",
                        default=[100])

    return parser.parse_args()


if __name__ == "__main__":
    main()
