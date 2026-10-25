"""Build data/adelaide_weather_2015.csv from the original weatherAUS.csv.

Source: Australian Bureau of Meteorology daily observations, as packaged in
the "Rain in Australia" (weatherAUS) dataset. Keeps Adelaide rows from 2015,
a subset of columns, and writes missing values ("NA") as empty cells.

Usage:
    python data/make_adelaide_weather.py path/to/weatherAUS.csv
"""

import csv
import sys
from pathlib import Path

COLUMNS = [
    "Date", "MinTemp", "MaxTemp", "Rainfall", "WindGustDir", "WindGustSpeed",
    "Humidity3pm", "Pressure3pm", "Temp3pm", "RainToday", "RainTomorrow",
]


def main(src):
    out = Path(__file__).with_name("adelaide_weather_2015.csv")
    with open(src, newline="", encoding="utf-8") as fin:
        rows = [
            r for r in csv.DictReader(fin)
            if r["Location"] == "Adelaide" and r["Date"].startswith("2015")
            and r["RainTomorrow"] not in ("", "NA")  # the label must be known
        ]
    with open(out, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, lineterminator="\n")
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow(["" if r[c] == "NA" else r[c] for c in COLUMNS])
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python data/make_adelaide_weather.py path/to/weatherAUS.csv")
    main(sys.argv[1])