# UK Hydrology API requester

This code is used to request data from the UK Hydrology API. You can find 
information about the API [here](https://environment.data.gov.uk/hydrology/doc/reference). 

## Usage
The usage is pretty simple. Executing
```bash
python -m data_fetch <start-date> <end-date> -s [station_name1, station_name2, ...] 
```
Will result in a CSV file with name `<station-name>_<start-date>_<end_date>.csv` being created in the current directory 
with the data from each station. The station names must be the names of the stations as they appear on the
[map](https://environment.data.gov.uk/hydrology/landing).

There is only one parameter that can be changed in the script, and that is the
`data_path` parameter (`-d`), and it is used to select the directory where the
data will be saved. By default, it is set to the current working directory.

If the station name is separated by spaces, it must be enclosed in quotes.

## Limitations
- The API has a limit of 2,000,000 rows per request. 
- Part of the code is pretty much hardcoded, so it might not work for all
  stations.