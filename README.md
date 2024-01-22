# Outlier detection

## Introduction
This code is used to detect outliers in the data of the water level stations of
the [Environment Agency of the UK](https://www.gov.uk/government/organisations/environment-agency). Each of its modules 
can be used by itself, but the main script in the root of the project is used to run the whole process
of outlier detection.

The code was written in Python 3.10.

## Usage
First of all, all required packages must be installed. This can be done by running the following command:
```shell
pip install -r requirements.txt
```

Models to train are required to be into a config file. A sample of the config file
can be found in `sample_config.json`. The config file must be passed to the main script
as a parameter. The main script can be run by executing the following command:
```shell
python main.py <start_date> <end_date> -s [station_name1, station_name2, ...] -c <config_file>
```

For example:
```shell
python main.py 2020-01-01 2020-12-31 -s "Sunbury Lock" Gloucester -c sample_config.json
```

You can execute
```shell
python main.py -h
```
to see the available parameters.

The station names must be the names of the stations as they appear on the
[map](https://environment.data.gov.uk/hydrology/landing). If the station name is separated by spaces, 
it must be enclosed in quotes.