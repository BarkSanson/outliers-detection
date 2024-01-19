import sys
import datetime
import os
import argparse
from pathlib import Path

from .requester import Requester


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
