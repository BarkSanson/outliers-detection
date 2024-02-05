import os
import argparse
from pathlib import Path

from .requester import Requester
from util.dates import parse_dates


def main():
    args = parse_args()

    station_names, data_path = \
        args.station_names, args.data_path
    start_date, end_date = \
        parse_dates(args.start_date, args.end_date)

    data_path = Path(data_path)

    os.makedirs(data_path, exist_ok=True)

    req = Requester(station_names, data_path, start_date.date(), end_date.date())

    req.do_request()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Request measures from the Environment Agency API for a specific station.')

    parser.add_argument(
        '-s',
        '--station_names',
        type=str,
        nargs='+',
        help='Station names to request data from.')

    parser.add_argument(
        'start_date',
        type=str,
        help='The start date of the data to request.')

    parser.add_argument(
        'end_date',
        type=str,
        help='The end date of the data to request.')

    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        default=os.getcwd(),
        help='The path to store the data. Current working directory by default.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
