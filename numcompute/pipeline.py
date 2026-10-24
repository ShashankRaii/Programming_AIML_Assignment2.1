"""Transformer/Estimator protocol, Pipeline and FeatureUnion.

Transformers have fit(X, y=None) -> self and transform(X); estimators have
fit(X, y) -> self and predict(X). Anything learned in fit is stored in an
attribute ending in "_" (mean_, categories_, ...), and using a component
before fit raises NotFittedError.

MeanRegressor and ThresholdClassifier are tiny example estimators for the
demo. The assignment does not ask for real models.
"""

import inspect
import numpy as np
from .utils import check_array, check_labels, check_same_length

__all__ = [
    "NotFittedError",
    "BaseTransformer",
    "BaseEstimator",
    "Pipeline",
    "FeatureUnion",
    "MeanRegressor",
    "ThresholdClassifier",
]


class NotFittedError(ValueError, AttributeError):
    """Raised when transform/predict is called before fit."""

class _ParamsMixin:
    """get_params, a readable repr and the fitted/shape checks."""

    def get_params(self):
        """Return the constructor arguments as a dict."""
        signature = inspect.signature(type(self).__init__)
        names = [
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind is not param.VAR_POSITIONAL
        ]
        return {name: getattr(self, name) for name in names if hasattr(self, name)}

    def __repr__(self):
        params = ", ".join(f"{key}={value!r}" for key, value in self.get_params().items())
        return f"{type(self).__name__}({params})"

    def _check_is_fitted(self, attribute):
        if not hasattr(self, attribute):
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet; call fit before using it"
            )

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} was fitted "
                f"with {self.n_features_in_} features"
            )


class BaseTransformer(_ParamsMixin):
    """Base class for preprocessors. Subclasses implement fit and transform."""

    def fit(self, X, y=None):
        raise NotImplementedError(f"{type(self).__name__} must implement fit")

    def transform(self, X):
        raise NotImplementedError(f"{type(self).__name__} must implement transform")

    def fit_transform(self, X, y=None):
        """Same as fit(X, y).transform(X).

        Parameters
        X : array_like, shape (n_samples, n_features)
        y : ignored

        Returns
        X_out : ndarray, shape (n_samples, n_features_out)
        """
        return self.fit(X, y).transform(X)


class BaseEstimator(_ParamsMixin):
    """Base class for models. Subclasses implement fit and predict."""

    def fit(self, X, y):
        raise NotImplementedError(f"{type(self).__name__} must implement fit")

    def predict(self, X):
        raise NotImplementedError(f"{type(self).__name__} must implement predict")


def _has(obj, method):
    return callable(getattr(obj, method, None))


def _fit_transform_one(step, X, y):
    if _has(step, "fit_transform"):
        return step.fit_transform(X, y)
    return step.fit(X, y).transform(X)


def _validate_named_steps(steps, owner):
    if not isinstance(steps, (list, tuple)) or len(steps) == 0:
        raise ValueError(f"{owner} requires a non-empty list of (name, component) tuples")
    names = []
    for item in steps:
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
            raise TypeError(f"{owner} steps must be (name, component) tuples; got {item!r}")
        names.append(item[0])
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{owner} step names must be unique; duplicated: {duplicates}")
    return list(steps)


class Pipeline(_ParamsMixin):
    """Run a list of steps one after another.

    Parameters
    steps : list of (str, component) tuples
        All steps except the last need fit and transform. The last one only
        needs fit, so it can be a transformer or an estimator.

    Attributes
    named_steps : dict
        Step name -> component.

    Raises
    TypeError
        If a step is not a (name, component) tuple or is missing a method.
    ValueError
        If steps is empty or two steps share a name.

    Complexity
    The sum of the steps' own costs.

    Examples
    >>> import numpy as np
    >>> from numcompute.preprocessing import MinMaxScaler, StandardScaler
    >>> pipe = Pipeline([("std", StandardScaler()), ("minmax", MinMaxScaler())])
    >>> pipe.fit_transform(np.array([[1.0], [2.0], [3.0]])).ravel()
    array([0. , 0.5, 1. ])
    """

    def __init__(self, steps):
        self.steps = _validate_named_steps(steps, "Pipeline")
        for name, step in self.steps[:-1]:
            if not (_has(step, "fit") and _has(step, "transform")):
                raise TypeError(
                    f"intermediate step {name!r} must implement fit and transform; "
                    f"got {type(step).__name__}"
                )
        last_name, last = self.steps[-1]
        if not _has(last, "fit"):
            raise TypeError(f"final step {last_name!r} must implement fit")

    @property
    def named_steps(self):
        return dict(self.steps)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, key):
        """pipe[0], pipe["scale"] or pipe[:2] (a smaller Pipeline)."""
        if isinstance(key, slice):
            return Pipeline(self.steps[key])
        if isinstance(key, str):
            return self.named_steps[key]
        return self.steps[key][1]

    def _transform_intermediate(self, X):
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X

    def fit(self, X, y=None):
        """Fit each step on the output of the previous one.

        Parameters
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,), optional
            Passed to every step; transformers ignore it.

        Returns
        self : Pipeline
        """
        for _, step in self.steps[:-1]:
            X = _fit_transform_one(step, X, y)
        self.steps[-1][1].fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit all steps and return the transformed X.

        Raises
        TypeError
            If the last step has no transform (e.g. it is an estimator).
        """
        last_name, last = self.steps[-1]
        if not _has(last, "transform"):
            raise TypeError(f"final step {last_name!r} has no transform; use fit/predict")
        for _, step in self.steps[:-1]:
            X = _fit_transform_one(step, X, y)
        return _fit_transform_one(last, X, y)

    def transform(self, X):
        """Pass X through every fitted step.

        Raises
        TypeError
            If the last step has no transform.
        NotFittedError
            If the pipeline has not been fitted.
        """
        last_name, last = self.steps[-1]
        if not _has(last, "transform"):
            raise TypeError(f"final step {last_name!r} has no transform; use predict")
        return last.transform(self._transform_intermediate(X))

    def predict(self, X):
        """Transform X with the earlier steps, then predict with the last one.

        Raises
        TypeError
            If the last step has no predict.
        """
        last_name, last = self.steps[-1]
        if not _has(last, "predict"):
            raise TypeError(f"final step {last_name!r} has no predict; use transform")
        return last.predict(self._transform_intermediate(X))


class FeatureUnion(BaseTransformer):
    """Apply several transformers to the same X and put the results side by side.
    Parameters
    transformer_list : list of (str, transformer) tuples

    Raises
    TypeError, ValueError
        Same rules as Pipeline. ValueError if the outputs have different
        numbers of rows.

    Complexity
    The transformers' costs plus O(n_samples * n_features_out) for hstack.

    Examples
    >>> import numpy as np
    >>> from numcompute.preprocessing import MinMaxScaler, StandardScaler
    >>> union = FeatureUnion([("std", StandardScaler()), ("minmax", MinMaxScaler())])
    >>> union.fit_transform(np.array([[1.0], [3.0]]))
    array([[-1.,  0.],
           [ 1.,  1.]])
    """

    def __init__(self, transformer_list):
        self.transformer_list = _validate_named_steps(transformer_list, "FeatureUnion")
        for name, step in self.transformer_list:
            if not (_has(step, "fit") and _has(step, "transform")):
                raise TypeError(f"transformer {name!r} must implement fit and transform")

    @staticmethod
    def _stack(blocks):
        blocks = [np.asarray(block) for block in blocks]
        blocks = [block.reshape(-1, 1) if block.ndim == 1 else block for block in blocks]
        rows = {block.shape[0] for block in blocks}
        if len(rows) > 1:
            raise ValueError(f"transformer outputs have different row counts: {sorted(rows)}")
        return np.hstack(blocks)

    def fit(self, X, y=None):
        for _, step in self.transformer_list:
            step.fit(X, y)
        self.n_transformers_ = len(self.transformer_list)
        return self

    def transform(self, X):
        """Return all transformer outputs stacked, shape (n_samples, total width)."""
        self._check_is_fitted("n_transformers_")
        return self._stack([step.transform(X) for _, step in self.transformer_list])

    def fit_transform(self, X, y=None):
        blocks = [_fit_transform_one(step, X, y) for _, step in self.transformer_list]
        self.n_transformers_ = len(self.transformer_list)
        return self._stack(blocks)


class MeanRegressor(BaseEstimator):
    """Example estimator: always predicts the mean of the training targets.

    Attributes
    mean_ : float
        Mean of y, ignoring NaNs.

    Complexity
    fit O(n) time, O(1) extra space; predict O(n) time and space.

    Examples
    >>> MeanRegressor().fit([[0], [1]], [2.0, 4.0]).predict([[5], [6], [7]])
    array([3., 3., 3.])
    """

    def fit(self, X, y):
        """Store the mean of y.

        Raises
        ValueError
            If X and y have different lengths or y has no non-NaN value.
        """
        X = check_array(X, ensure_2d=True)
        y = check_array(y, dtype=np.float64)
        check_same_length(X, y)
        valid = y[~np.isnan(y)]
        if valid.size == 0:
            raise ValueError("y must contain at least one non-NaN value")
        self.mean_ = float(valid.mean())
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Return mean_ for every row, shape (n_samples,)."""
        self._check_is_fitted("mean_")
        X = check_array(X, ensure_2d=True)
        self._check_n_features(X)
        return np.full(X.shape[0], self.mean_)


class ThresholdClassifier(BaseEstimator):
    """Example binary classifier that splits on one feature.

    The threshold is the midpoint between the two class means of
    X[:, feature]. Only meant to give the demo some realistic metrics.

    Parameters
    feature : int, default 0
        Column used for the decision.

    Attributes
    classes_ : ndarray, shape (2,)
    threshold_ : float

    Raises
    ValueError
        If y does not have exactly two classes.

    Complexity
    fit and predict: O(n) time and space.

    Examples
    >>> X = [[1.0], [2.0], [8.0], [9.0]]
    >>> clf = ThresholdClassifier().fit(X, [0, 0, 1, 1])
    >>> clf.threshold_
    5.0
    >>> clf.predict([[0.0], [10.0]])
    array([0, 1])
    """

    def __init__(self, feature=0):
        self.feature = feature

    def fit(self, X, y):
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        y = check_labels(y)
        check_same_length(X, y)
        if not -X.shape[1] <= self.feature < X.shape[1]:
            raise ValueError(f"feature {self.feature} is out of range for {X.shape[1]} columns")
        classes, encoded = np.unique(y, return_inverse=True)
        if classes.size != 2:
            raise ValueError(f"ThresholdClassifier needs exactly 2 classes; got {classes.size}")
        column = X[:, self.feature]
        valid = ~np.isnan(column)
        sums = np.bincount(encoded[valid], weights=column[valid], minlength=2)
        counts = np.bincount(encoded[valid], minlength=2)
        if np.any(counts == 0):
            raise ValueError("each class needs at least one non-NaN value in the feature column")
        means = sums / counts
        self.classes_ = classes
        self.threshold_ = float(means.mean())
        # The second class can sit below the first, e.g. labels "high"/"low".
        self.positive_above_ = bool(means[1] >= means[0])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Return labels, shape (n_samples,). Rows with NaN get classes_[0]."""
        self._check_is_fitted("threshold_")
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        self._check_n_features(X)
        column = X[:, self.feature]
        above = column > self.threshold_
        positive = above if self.positive_above_ else ~above & ~np.isnan(column)
        return self.classes_[positive.astype(np.intp)]