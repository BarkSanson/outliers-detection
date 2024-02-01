import pandas as pd


class WindowGenerator:
    @staticmethod
    def split_window(df, window_length):
        if window_length > len(df):
            raise (
                ValueError(f'Window length ({window_length}) cannot be greater than the length of the dataframe ({len(df)})'))

        if window_length <= 0:
            raise ValueError(f'Window length ({window_length}) must be greater than 0')

        if window_length == 1:
            return df

        # Create the windows and, if any of the window values has a 1 in 'outlier'
        # column, then the window is an outlier
        windows = []
        for i in range(len(df) - window_length):
            window = df.iloc[i:i + window_length]
            outlier = 1 if 1 in window['outlier'].values else 0

            windows.append([*window['value'].values.flatten(), outlier])

        windows_df = pd.DataFrame(windows, columns=[*[f'timestep_{i}' for i in range(window_length)], 'outlier'])

        return windows_df
