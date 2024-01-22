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

The main script can be run by executing the following command:
```shell
python main.py <start-date> <end-date> -s [station_name1, station_name2, ...]
```
The station names must be the names of the stations as they appear on the
[map](https://environment.data.gov.uk/hydrology/landing). If the station name is separated by spaces, 
it must be enclosed in quotes.

You can run `python main.py -h` to see the available parameters.