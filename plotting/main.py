import os

import matplotlib.pyplot as plt
import pandas as pd
import calendar

DATA_PATH = os.path.join(os.getcwd(), "..", "data")
PLOTS_PATH = os.path.join(os.getcwd(), "plots")
STATIONS = ["Gloucester", "Ruislip", "SunburyLock", "OsneyLock",
            "ShirleyPond", "GurneyStreet", "Polperro", "SalisburyHarnhamBridge",
            "Halesowen", "StoneferryBridge"]

MONTHS = 3


def main():
    os.makedirs(PLOTS_PATH, exist_ok=True)
    for station in STATIONS:
        df = pd.read_csv(os.path.join(DATA_PATH, f"{station}.csv"))
        df["dateTime"] = pd.to_datetime(df["dateTime"])
        df.drop(["measure"], axis=1)
        df.set_index("dateTime", inplace=True)

        print(station)
        print(df.describe())

        intervals = pd.date_range(start=df.index.min(), end=df.index.max(), freq="3M")

        row_count = len(intervals) // MONTHS + len(intervals) % MONTHS
        cols_count = 3

        fig, axes = plt.subplots(7, cols_count, figsize=(30, 5 * row_count))

        os.makedirs(os.path.join(PLOTS_PATH, station), exist_ok=True)
        # Print rows that quality is "Missing"
        print(df[df["quality"] == "Missing"])
        for i, interval in enumerate(intervals):
            current_row = i // MONTHS
            current_col = i % cols_count
            current_interval = df.loc[interval:interval + pd.DateOffset(months=3)]

            for label, grp in current_interval.groupby("quality"):
                color = "r" if label == "Missing" else "g" if label == "Good" else "b"
                axes[current_row, current_col].plot(grp.index, grp["value"], label=label, color=color)
                axes[current_row, current_col].set_xlabel("Data")
                axes[current_row, current_col].set_ylabel("Nivell de l'aigua (m)")
                axes[current_row, current_col].set_title(
                    f"Nivell de l'aigua de {station} entre {interval} i {interval + pd.DateOffset(months=3)}")
                axes[current_row, current_col].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                PLOTS_PATH,
                station,
                f"{station}-{df.index.min()}_{df.index.max()}.png",
            )
        )


if __name__ == "__main__":
    main()
