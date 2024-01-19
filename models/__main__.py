import os

import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from plotting.plotter import Plotter

DATA_PATH = os.path.join(os.path.pardir, "data")

TEST_SIZE = 0.3


def fit_iso_forest(df):
    iforest = IsolationForest(n_estimators=100, n_jobs=-1, warm_start=False)
    iforest.fit(df)

    return iforest


def fit_predict_lof(df):
    lof = LocalOutlierFactor(n_neighbors=2, n_jobs=-1)

    return lof, lof.fit_predict(df)

def main():
    station = "SunburyLock"

    df = pd.read_csv(os.path.join(DATA_PATH, f"{station}.csv"))

    df["dateTime"] = pd.to_datetime(df["dateTime"])
    df.set_index("dateTime", inplace=True)
    df = df.drop(["measure", "date", "completeness", "quality", "qcode"], axis=1)

    # Drop NaN values
    df = df.dropna()

    print(df.shape)

    print(df.describe())
    print(df.head())

    start = df.index.min()
    end = pd.to_datetime("2019-12-31")

    # X_train, X_test = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    iforest = fit_iso_forest(df)
    lof, lof_predict = fit_predict_lof(df[start:end])

    intervals = pd.date_range(start=df.index.min(), end=df.index.max(), freq="3M")

    ncols = 1
    nrows = len(intervals) // ncols

    # fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    # for i, interval in enumerate(intervals):
    #    # Plot original data and outlier data of LOF
    #    current_interval = df.loc[interval:interval + pd.DateOffset(months=3)]
    #    preds = lof_predict[i * len(current_interval):(i + 1) * len(current_interval)]

    #    axes[i].scatter(current_interval.index, current_interval["value"], c=preds, s=2.)

    #    axes[i].set_xlabel("Data")
    #    axes[i].set_ylabel("Nivell de l'aigua (m)")
    #    axes[i].set_title(
    #        f"Nivell de l'aigua de {station} entre {interval} i {interval + pd.DateOffset(months=3)}")
    #    axes[i].plot()

    # tuner = Tuner(model=IsolationForest(), df=df, n_estimators=[50, 100, 200, 300, 400, 500],
    #              max_samples=[100, 200, 300, 400, 500], contamination=[0.01, 0.05, 0.1, 0.2],n_jobs=[-1])

    # best_params, best_estimator = tuner.tune()
    # print(best_params)

    # fig2, axes2 = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    ## Plot isolation forest results
    # for i, interval in enumerate(intervals):
    #    # Plot original data and outlier data of LOF
    #    current_interval = df.loc[interval:interval + pd.DateOffset(months=3)]
    #    preds = best_estimator.predict(current_interval)

    #    axes2[i].scatter(current_interval.index, current_interval["value"],
    #                     c=preds, s=2.)
    #    axes2[i].set_xlabel("Data")
    #    axes2[i].set_ylabel("Nivell de l'aigua (m)")
    #    axes2[i].set_title(
    #        f"Nivell de l'aigua de {station} entre {interval} i {interval + pd.DateOffset(months=3)}")
    #    axes2[i].plot()

    # plt.show()


if __name__ == "__main__":
    main()
