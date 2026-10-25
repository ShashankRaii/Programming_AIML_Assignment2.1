"""Unit tests for numcompute.pipeline."""

import numpy as np
import pytest

from numcompute.pipeline import (
    BaseEstimator,
    BaseTransformer,
    FeatureUnion,
    MeanRegressor,
    NotFittedError,
    Pipeline,
    ThresholdClassifier,
)
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler

X = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 40.0]])


class AddOne(BaseTransformer):
    """Minimal custom transformer used to test the protocol."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted("fitted_")
        return np.asarray(X) + 1


# Pipeline
def test_spec_example_scale_then_encode():
    """The exact example from the assignment specification."""
    pipe = Pipeline([("scale", StandardScaler()), ("encode", OneHotEncoder())])
    data = np.array([[1.0], [2.0], [1.0]])
    X_tr = pipe.fit_transform(data)
    np.testing.assert_array_equal(X_tr, [[1, 0], [0, 1], [1, 0]])


def test_pipeline_equals_manual_chaining():
    pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    manual = StandardScaler().fit_transform(SimpleImputer().fit_transform(X))
    np.testing.assert_allclose(pipe.fit_transform(X), manual)
    np.testing.assert_allclose(pipe.transform(X), manual)


def test_pipeline_fit_returns_self_and_named_steps():
    pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    assert pipe.fit(X) is pipe
    assert isinstance(pipe.named_steps["scale"], StandardScaler)
    assert pipe["impute"] is pipe[0]
    assert len(pipe) == 2
    assert isinstance(pipe[:1], Pipeline) and len(pipe[:1]) == 1


def test_pipeline_learned_state_comes_from_previous_step():
    pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]).fit(X)
    # Scaler saw imputed data, so its column-1 count includes the imputed value.
    assert pipe["scale"].n_samples_seen_.tolist() == [4, 4]


def test_pipeline_with_estimator_predicts():
    y = np.array([0, 0, 1, 1])
    pipe = Pipeline([("impute", SimpleImputer()), ("clf", ThresholdClassifier(feature=0))])
    np.testing.assert_array_equal(pipe.fit(X, y).predict(X), y)
    with pytest.raises(TypeError, match="no transform"):
        pipe.transform(X)
    with pytest.raises(TypeError, match="no transform"):
        pipe.fit_transform(X, y)


def test_pipeline_predict_requires_estimator():
    pipe = Pipeline([("scale", StandardScaler())]).fit(X)
    with pytest.raises(TypeError, match="no predict"):
        pipe.predict(X)


def test_pipeline_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]).transform(X)


def test_pipeline_custom_transformer():
    pipe = Pipeline([("add", AddOne()), ("add_again", AddOne())])
    np.testing.assert_array_equal(pipe.fit_transform(np.array([[1]])), [[3]])


@pytest.mark.parametrize("compose", [Pipeline, FeatureUnion])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_minimal_fit_x_protocol(compose, method):
    class Minimal:
        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X) + 1

    obj = compose([("minimal", Minimal())])
    getattr(obj, method)([[1.0]], [0])
    np.testing.assert_array_equal(obj.transform([[2.0]]), [[3.0]])


def test_keyword_only_y_and_fit_transform_x_protocol():
    class KeywordTarget:
        def fit(self, X, *, y):
            self.target = y
            return self

        def transform(self, X):
            return np.asarray(X)

    step = KeywordTarget()
    Pipeline([("step", step)]).fit([[1]], [9])
    assert step.target == [9]

    class Direct(AddOne):
        def fit_transform(self, X):
            return self.fit(X).transform(X)

    np.testing.assert_array_equal(Pipeline([("step", Direct())]).fit_transform([[1]]), [[2]])


def test_internal_type_error_is_not_retried():
    class Broken(AddOne):
        calls = 0

        def fit(self, X, y=None):
            self.calls += 1
            raise TypeError("error inside fit")

    step = Broken()
    with pytest.raises(TypeError, match="error inside fit"):
        Pipeline([("broken", step)]).fit([[1]])
    assert step.calls == 1


def test_pipeline_requires_its_own_fit_and_runtime_error_contract():
    assert issubclass(NotFittedError, RuntimeError)
    step = AddOne().fit([[1]])
    with pytest.raises(NotFittedError):
        Pipeline([("step", step)]).transform([[1]])


def test_union_rejects_scalar_and_three_dimensional_outputs():
    class BadShape(AddOne):
        def transform(self, X):
            return np.zeros((1, 1, 1))

    with pytest.raises(ValueError, match="1-D or 2-D"):
        FeatureUnion([("bad", BadShape())]).fit_transform([[1]])


@pytest.mark.parametrize(
    "steps, err, msg",
    [
        ([], ValueError, "non-empty"),
        ("scale", ValueError, "non-empty"),
        ([StandardScaler()], TypeError, "tuples"),
        ([("a", StandardScaler()), ("a", MinMaxScaler())], ValueError, "unique"),
        ([("est", MeanRegressor()), ("scale", StandardScaler())], TypeError, "fit and transform"),
        ([("bad", object())], TypeError, "must implement fit"),
    ],
)
def test_pipeline_invalid_steps(steps, err, msg):
    with pytest.raises(err, match=msg):
        Pipeline(steps)


# FeatureUnion
def test_feature_union_stacks_columns():
    data = np.array([[0.0], [5.0], [10.0]])
    union = FeatureUnion([("std", StandardScaler()), ("minmax", MinMaxScaler())])
    out = union.fit_transform(data)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 1], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(union.transform(data), out)


def test_feature_union_inside_pipeline():
    pipe = Pipeline(
        [
            ("impute", SimpleImputer()),
            ("union", FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())])),
        ]
    )
    assert pipe.fit_transform(X).shape == (4, 4)


def test_feature_union_errors():
    with pytest.raises(NotFittedError):
        FeatureUnion([("std", StandardScaler())]).transform(X)
    with pytest.raises(TypeError, match="fit and transform"):
        FeatureUnion([("est", MeanRegressor())])


# Base classes and reference estimators
def test_base_classes_require_overrides():
    with pytest.raises(NotImplementedError):
        BaseTransformer().fit(X)
    with pytest.raises(NotImplementedError):
        BaseTransformer().transform(X)
    with pytest.raises(NotImplementedError):
        BaseEstimator().fit(X, None)
    with pytest.raises(NotImplementedError):
        BaseEstimator().predict(X)


def test_not_fitted_error_is_value_error():
    assert issubclass(NotFittedError, ValueError)


def test_mean_regressor():
    model = MeanRegressor().fit(X, [1.0, np.nan, 3.0, 5.0])
    assert model.mean_ == 3.0
    np.testing.assert_array_equal(model.predict(X[:2]), [3.0, 3.0])
    with pytest.raises(ValueError, match="same length"):
        MeanRegressor().fit(X, [1.0, 2.0])
    with pytest.raises(ValueError, match="non-NaN"):
        MeanRegressor().fit(X[:1], [np.nan])
    with pytest.raises(NotFittedError):
        MeanRegressor().predict(X)


def test_threshold_classifier_string_labels_and_reversed_direction():
    data = np.array([[9.0], [8.0], [1.0], [2.0]])
    clf = ThresholdClassifier().fit(data, ["low", "low", "high", "high"])
    assert clf.threshold_ == 5.0
    assert clf.predict(np.array([[0.0], [10.0]])).tolist() == ["high", "low"]


def test_threshold_classifier_errors():
    with pytest.raises(ValueError, match="exactly 2 classes"):
        ThresholdClassifier().fit(X, [0, 1, 2, 0])
    with pytest.raises(ValueError, match="out of range"):
        ThresholdClassifier(feature=5).fit(X, [0, 0, 1, 1])
    clf = ThresholdClassifier().fit(X, [0, 0, 1, 1])
    with pytest.raises(ValueError, match="features"):
        clf.predict(np.array([[1.0]]))
