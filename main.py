import argparse
import os
import sys
import datetime

from data_fetch import Requester
from plotting import Plotter


def main():
    args = parse_args()
    stations, start_date, end_date, data_path, plot_path, plot_data = (
        args.stations, args.start_date, args.end_date, args.data_path, args.plot_path, args.plot_data)

    start_date, end_date = parse_dates(start_date, end_date)

    requester = Requester(stations, data_path, start_date.date(), end_date.date())

    dfs = requester.do_request()

    if plot_data:
        for station, df in dfs.items():
            plotter = Plotter(df, plot_path)
            plotter.plot_data(df, 'value', f'{station} water level')

    # TODO: Train models


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

    parser.add_argument("--stations", nargs="+", type=str, help="Station names", required=True)
    parser.add_argument("start_date", type=str, help="Start date")
    parser.add_argument("end_date", type=str, help="End date")
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
                        help="Plot data",
                        action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main()
