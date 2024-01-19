import requests
import os

from io import StringIO

import pandas as pd


class Requester:
    BASE_URL = 'https://environment.data.gov.uk/hydrology/id'

    def __init__(
            self,
            station_names,
            data_path,
            start_date,
            end_date,
    ):
        self._station_names = station_names
        self._data_path = data_path
        self._start_date = start_date
        self._end_date = end_date

    def do_request(self):
        dfs = {}
        for station_name in self._station_names:
            processed_station_name = station_name.replace(' ', '%20')
            measure_id_req_url = f"{Requester.BASE_URL}/stations.json?search={processed_station_name}"
            name_json = requests.get(measure_id_req_url).json()
            measure_id = name_json["items"][0]["notation"]

            processed_station_name = station_name.replace(' ', '')
            request_url = \
                (f"{Requester.BASE_URL}/measures/{measure_id}-level-i-900-m-qualified/readings.csv"
                 f"?mineq-date={self._start_date}&maxeq-date={self._end_date}&_limit=2000000")
            data_path = os.path.join(
                self._data_path,
                f'{processed_station_name}_{self._start_date}_{self._end_date}.csv')
            if os.path.exists(data_path):
                print(f'Data already exists between {self._start_date} and {self._end_date} for {station_name}')
                df = pd.read_csv(data_path)
                dfs[station_name] = df
                continue

            print(f'Requesting data for {station_name}...')
            res = requests.get(request_url)

            df = pd.read_csv(StringIO(res.text))

            df = df.drop(columns=['measure'])

            df.to_csv(
                os.path.join(
                    self._data_path,
                    f'{processed_station_name}_{self._start_date}_{self._end_date}.csv'),
                    index=False)

            dfs[station_name] = df

        print('Done!')

        return dfs
