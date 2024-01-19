import sys
import datetime
import os
import argparse
from pathlib import Path

from requester import Requester


ROOT = os.getcwd()
DATA_PATH = os.path.join(ROOT, 'data')


def main():
    args = parse_args()

    station_name, data_path = \
        args.station_name, args.data_path
    start_date, end_date = \
        parse_dates(args.start_date, args.end_date)

    data_path = Path(data_path)

    os.makedirs(data_path, exist_ok=True)

    req = Requester(station_name, data_path, BASE_URL, start_date.date(), end_date.date())

    df = req.do_request()

    print(df.head())




def parse_args():
    parser = argparse.ArgumentParser(
        description='Request measures from the Environment Agency API for a specific station.')

    parser.add_argument(
        'station_name',
        type=str,
        help='Measure ID of the station to request data for.')

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
        default=DATA_PATH,
        help='The path to store the data. Same directory as this script by default.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
