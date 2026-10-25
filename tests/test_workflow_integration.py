"""Exercise the mixed CSV workflow across module boundaries."""

from pathlib import Path

import numpy as np

from numcompute.io import load_csv, load_csv_chunks, read_header
from numcompute.metrics import accuracy, confusion_matrix
from numcompute.pipeline import Pipeline, ThresholdClassifier
from numcompute.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute.stats import RunningStats


def test_weather_numeric_categorical_pipeline_metrics_and_streaming_agree():
    path = Path(__file__).resolve().parents[1] / "data/adelaide_weather_2015.csv"
    header = read_header(path)
    cols = [header.index(c) for c in ["MaxTemp", "Rainfall", "Humidity3pm"]]
    numeric = load_csv(path, skip_header=1, usecols=cols)
    before = numeric.copy()
    categorical = load_csv(path, skip_header=1, usecols=header.index("WindGustDir"), dtype=str)
    y = (load_csv(path, skip_header=1, usecols=header.index("RainTomorrow"), dtype=str)
         .ravel() == "Yes").astype(int)
    scaled = Pipeline([("fill", SimpleImputer()), ("scale", StandardScaler())])
    encoded = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent", missing_values="")),
        ("encode", OneHotEncoder()),
    ])
    combined = np.hstack([scaled.fit_transform(numeric), encoded.fit_transform(categorical)])
    assert combined.shape[0] == len(y)
    assert combined.shape[1] == 3 + len(encoded["encode"].categories_[0])
    assert np.isfinite(combined).all()
    np.testing.assert_array_equal(combined[:, 3:].sum(axis=1), 1)
    model = Pipeline([("fill", SimpleImputer()), ("model", ThresholdClassifier(feature=2))])
    predictions = model.fit(numeric, y).predict(numeric)
    matrix = confusion_matrix(y, predictions)
    assert matrix.sum() == len(y)
    assert matrix.trace() / matrix.sum() == accuracy(y, predictions)
    running = RunningStats()
    for chunk in load_csv_chunks(path, 37, skip_header=1, usecols=cols[0]):
        running.update(chunk.ravel())
    np.testing.assert_allclose([running.mean, running.variance],
                               [np.nanmean(numeric[:, 0]), np.nanvar(numeric[:, 0])])
    np.testing.assert_array_equal(numeric, before)
