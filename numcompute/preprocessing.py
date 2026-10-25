"""Feature transformers: StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder.

All of them take a 2-D array (n_samples, n_features) and follow the
fit / transform / fit_transform API from numcompute.pipeline. The input is
never modified.

The scalers skip NaNs when learning their statistics and leave NaNs in place
when transforming. OneHotEncoder does not accept NaNs, so impute first.
"""

import numpy as np

from .pipeline import BaseTransformer
from .stats import mean as _nanmean
from .stats import median as _nanmedian
from .utils import check_array, check_categories

__all__ = ["StandardScaler", "MinMaxScaler", "SimpleImputer", "OneHotEncoder"]


def _check_numeric(X):
    return check_array(X, ensure_2d=True, dtype=np.float64, allow_inf=False)


def _require_samples(X, name):
    if X.shape[0] == 0:
        raise ValueError(f"{name}.fit requires at least one sample; got shape {X.shape}")


def _check_flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")


def _column_moments(X):
    """Mean, population std and non-NaN count of every column, ignoring NaNs."""
    valid = ~np.isnan(X)
    counts = valid.sum(axis=0)
    safe_counts = np.maximum(counts, 1)
    lo = np.where(valid, X, np.inf).min(axis=0)
    hi = np.where(valid, X, -np.inf).max(axis=0)
    lo, hi = np.where(counts, lo, 0.0), np.where(counts, hi, 0.0)
    center = lo / 2 + hi / 2
    dev = np.where(valid, X - center, 0.0)
    peak = np.abs(dev).max(axis=0, initial=0.0)
    unit = dev / np.where(peak > 0, peak, 1.0)
    offset = np.clip(unit.sum(axis=0) / safe_counts, -1.0, 1.0)
    with np.errstate(over="ignore"):
        mean = np.clip(center + offset * peak, lo, hi)
    mean = np.where(lo == hi, lo, mean)
    deviations = np.where(valid, unit - offset, 0.0)
    # Midrange centering bounds the population standard deviation by peak.
    rms = np.minimum(np.sqrt((deviations**2).sum(axis=0) / safe_counts), 1.0)
    return mean, peak * rms, counts


def _safe_scale(scale, reference):
    """Use 1 instead of 0 for constant columns so we never divide by zero."""
    # mean([0.1, 0.1, 0.1]) is not exactly 0.1, so a constant column can end up
    # with a std of a few ulps. The cutoff is relative, so real small data is kept.
    tiny = 10 * np.finfo(np.float64).eps * np.abs(reference)
    return np.where((scale == 0) | (scale <= tiny), 1.0, scale)


def _standardise(X, center, scale):
    with np.errstate(over="ignore", invalid="ignore"):
        out = (X - center) / scale
    bad = ~np.isfinite(out) & np.isfinite(X)
    if bad.any():
        out = np.where(bad, (X / 2 - center / 2) / scale * 2, out)
    return out


class StandardScaler(BaseTransformer):
    """Standardise each column to mean 0 and standard deviation 1.

    z = (x - mean_) / scale_. NaNs are ignored in fit and stay NaN in transform.

    Parameters
    with_mean : bool, default True
        Subtract the column mean.
    with_std : bool, default True
        Divide by the column standard deviation (population, ddof=0).

    Attributes
    mean_ : ndarray, shape (n_features,)
    scale_ : ndarray, shape (n_features,)
        Standard deviations, with 1.0 for constant columns.
    var_ : ndarray, shape (n_features,)
    n_samples_seen_ : ndarray of int, shape (n_features,)
        Non-NaN values per column.
    n_features_in_ : int

    Raises
    TypeError
        If X is not numeric.
    ValueError
        If X is not 2-D, contains inf, has no rows in fit, or has a different
        number of columns in transform.
    NotFittedError
        If transform is called before fit.

    Complexity
    fit and transform: O(n * d) time and space.

    Examples
    >>> import numpy as np
    >>> X = np.array([[1.0, 10.0], [3.0, 10.0]])
    >>> StandardScaler().fit_transform(X)
    array([[-1.,  0.],
           [ 1.,  0.]])
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        """Learn the column-wise transformation parameters and return self.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Real numeric matrix, converted to float64; infinities are rejected.
        y : optional
            Ignored; accepted for composition with supervised components.

        Returns
        -------
        self : StandardScaler

        Raises
        ------
        TypeError
            Unsupported input type or non-boolean configuration flag.
        ValueError
            Invalid matrix shape, infinity or feature-count mismatch.
        ValueError
            No training rows.

        Complexity
        ----------
        O(n*d) time and auxiliary space for n samples and d features.

        Notes
        -----
        Inputs are never modified. NaNs remain in place; all-NaN training features pass through.
        Constant features use scale 1. var_ can be infinite when its square cannot be
        represented, even when scale_ is finite. Computations are column-wise; float64
        conversion may round large integers.
        """
        _check_flag(self.with_mean, "with_mean")
        _check_flag(self.with_std, "with_std")
        X = _check_numeric(X)
        _require_samples(X, "StandardScaler")
        mean, std, counts = _column_moments(X)
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = counts
        with np.errstate(over="ignore"):
            self.var_ = std**2
        self.mean_ = mean if self.with_mean else np.zeros(X.shape[1])
        self.scale_ = _safe_scale(std, mean) if self.with_std else np.ones(X.shape[1])
        return self

    def transform(self, X):
        """Return an independent standardized matrix.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Real numeric matrix, converted to float64; infinities are rejected.

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, n_features)

        Raises
        ------
        TypeError
            Unsupported input type or non-boolean configuration flag.
        ValueError
            Invalid matrix shape, infinity or feature-count mismatch.
        NotFittedError
            Called before fitting (a RuntimeError, also a ValueError/AttributeError).

        Complexity
        ----------
        O(n*d) time and auxiliary space for n samples and d features.

        Notes
        -----
        Inputs are never modified. NaNs remain in place; all-NaN training features pass through.
        Constant features use scale 1. var_ can be infinite when its square cannot be
        represented, even when scale_ is finite. Computations are column-wise; float64
        conversion may round large integers. Empty sample axes are accepted.
        """
        self._check_is_fitted("scale_")
        X = _check_numeric(X)
        self._check_n_features(X)
        return _standardise(X, self.mean_, self.scale_)

    def inverse_transform(self, X):
        """Undo the fitted centering and scaling.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Real numeric matrix, converted to float64; infinities are rejected.

        Returns
        -------
        X_out : ndarray of float64, shape matching X

        Raises
        ------
        TypeError
            Unsupported input type or non-boolean configuration flag.
        ValueError
            Invalid matrix shape, infinity or feature-count mismatch.
        NotFittedError
            Called before fitting (a RuntimeError, also a ValueError/AttributeError).

        Complexity
        ----------
        O(n*d) time and output space.

        Notes
        -----
        Inputs are never modified. Computations act per column (axis 0). NaNs follow the class
        policy. Float64 conversion may round large integers. Reconstruction has ordinary float64
        rounding and overflow limits.
        """
        self._check_is_fitted("scale_")
        X = _check_numeric(X)
        self._check_n_features(X)
        return X * self.scale_ + self.mean_


class MinMaxScaler(BaseTransformer):
    """Scale each column linearly into feature_range, (0, 1) by default.

    A constant training column maps to the lower bound; subsequent changes
    use a unit data range. NaNs are ignored in fit and stay NaN in transform.
    Features fitted with no valid values transform to NaN.

    Parameters
    feature_range : (lo, hi), default (0, 1)
        Finite numbers with lo < hi.
    clip : bool, default False
        Clip values of new data that fall outside the training range.

    Attributes
    data_min_, data_max_ : ndarray, shape (n_features,)
        NaN for a column that was all NaN.
    data_range_ : ndarray, shape (n_features,)
    n_features_in_ : int

    Raises
    TypeError, ValueError, NotFittedError
        Same as StandardScaler, plus ValueError for a bad feature_range.

    Complexity
    fit and transform: O(n * d) time and space.

    Examples
    >>> import numpy as np
    >>> MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.array([[0.0], [5.0], [10.0]]))
    array([[-1.],
           [ 0.],
           [ 1.]])
    """

    def __init__(self, feature_range=(0, 1), clip=False):
        self.feature_range = feature_range
        self.clip = clip

    def _range(self):
        _check_flag(self.clip, "clip")
        try:
            lo, hi = (float(v) for v in self.feature_range)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feature_range must be a (min, max) pair of numbers; got {self.feature_range!r}"
            ) from exc
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"feature_range must satisfy finite min < max; got {(lo, hi)}")
        return lo, hi

    def fit(self, X, y=None):
        """Learn the column-wise transformation parameters and return self.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Real numeric matrix, converted to float64; infinities are rejected.
        y : optional
            Ignored; accepted for composition with supervised components.

        Returns
        -------
        self : MinMaxScaler

        Raises
        ------
        TypeError
            Unsupported input type or non-boolean configuration flag.
        ValueError
            Invalid matrix shape, infinity or feature-count mismatch.
        ValueError
            No training rows.

        Complexity
        ----------
        O(n*d) time and auxiliary space for n samples and d features.

        Notes
        -----
        Inputs are never modified. NaNs remain in place; all-NaN training features return NaN.
        Constant training features use a unit data range for extrapolation. Unrepresentable
        extrapolated outputs become infinity silently. Computations are column-wise; float64
        conversion may round large integers.
        """
        self._range()
        X = _check_numeric(X)
        _require_samples(X, "MinMaxScaler")
        valid = ~np.isnan(X)
        has_data = valid.any(axis=0)
        data_min = np.where(valid, X, np.inf).min(axis=0)
        data_max = np.where(valid, X, -np.inf).max(axis=0)
        self.data_min_ = np.where(has_data, data_min, np.nan)
        self.data_max_ = np.where(has_data, data_max, np.nan)
        # Halves are a fallback only: halving subnormal ranges loses data.
        self._half_range = np.where(has_data, data_max / 2 - data_min / 2, 0.0)
        with np.errstate(over="ignore"):
            self.data_range_ = self.data_max_ - self.data_min_
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return an independent scaled matrix.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Real numeric matrix, converted to float64; infinities are rejected.

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, n_features)

        Raises
        ------
        TypeError
            Unsupported input type or non-boolean configuration flag.
        ValueError
            Invalid matrix shape, infinity or feature-count mismatch.
        NotFittedError
            Called before fitting (a RuntimeError, also a ValueError/AttributeError).

        Complexity
        ----------
        O(n*d) time and auxiliary space for n samples and d features.

        Notes
        -----
        Inputs are never modified. NaNs remain in place; all-NaN training features return NaN.
        Constant training features use a unit data range for extrapolation. Unrepresentable
        extrapolated outputs become infinity silently. Computations are column-wise; float64
        conversion may round large integers. Empty sample axes are accepted.
        """
        self._check_is_fitted("data_min_")
        lo, hi = self._range()
        X = _check_numeric(X)
        self._check_n_features(X)
        denominator = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            unit = (X - self.data_min_) / denominator
            fallback = (X / 2 - self.data_min_ / 2) / np.where(
                np.isinf(denominator), self._half_range, denominator / 2
            )
            unit = np.where(np.isinf(denominator) | np.isinf(unit), fallback, unit)
            # Convex interpolation avoids an overflowing target hi - lo.
            inside = lo * (1 - unit) + hi * unit
            outside = lo + unit * (hi - lo)
            out = np.where((unit >= 0) & (unit <= 1), inside, outside)
            alternative = (lo / 2 + unit * (hi / 2 - lo / 2)) * 2
            out = np.where(np.isinf(out) | np.isnan(out), alternative, out)
        if self.clip:
            out = np.clip(out, lo, hi)
        return out


def _missing_mask(X, missing_values):
    if isinstance(missing_values, (float, np.floating)) and np.isnan(missing_values):
        if X.dtype.kind == "f":
            return np.isnan(X)
        if X.dtype.kind == "O":
            return X != X  # NaN is the only value that isn't equal to itself
        return np.zeros(X.shape, dtype=bool)
    return X == missing_values


class SimpleImputer(BaseTransformer):
    """Fill missing values in each column.

    Parameters
    strategy : {"mean", "median", "most_frequent", "constant"}, default "mean"
        mean and median need numeric data; most_frequent and constant also
        work on string columns.
    fill_value : scalar, optional
        Used with strategy="constant". Defaults to 0, or "missing" for strings.
    missing_values : scalar, default np.nan
        What counts as missing. Use "" for string columns from load_csv(dtype=str).

    Attributes
    statistics_ : ndarray, shape (n_features,)
        The value used to fill each column.
    n_features_in_ : int

    Raises
    ValueError
        Unknown strategy, a column with nothing but missing values (mean,
        median, most_frequent), no rows in fit, or a column-count mismatch.
    TypeError
        mean/median on non-numeric data, or an incompatible fill_value.

    Complexity
    ----------
    fit: O(n * d) for mean/constant, O(d * n log n) for median and
    most_frequent. transform: O(n * d) time and space.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]])
    >>> SimpleImputer(strategy="mean").fit_transform(X)
    array([[1., 6.],
           [3., 4.],
           [2., 8.]])
    >>> SimpleImputer(strategy="most_frequent", missing_values="").fit_transform(
    ...     np.array([["a"], [""], ["a"], ["b"]]))
    array([['a'],
           ['a'],
           ['a'],
           ['b']], dtype='<U1')
    """

    _STRATEGIES = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy="mean", fill_value=None, missing_values=np.nan):
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values

    def _validate(self, X):
        if self.strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}; got {self.strategy!r}")
        if self.strategy in ("mean", "median"):
            if np.asarray(X).dtype.kind not in "biuf":
                raise TypeError(f"strategy={self.strategy!r} requires numeric data")
            return check_array(X, ensure_2d=True, dtype=np.float64, allow_inf=False)
        return check_categories(X, ensure_2d=True)

    def fit(self, X, y=None):
        """Learn the column-wise transformation parameters and return self.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Numeric matrix, or comparable categorical strings/numbers where supported.
        y : optional
            Ignored; accepted for composition with supervised components.

        Returns
        -------
        self : SimpleImputer

        Raises
        ------
        TypeError
            Unsupported input or incomparable category types.
        ValueError
            Invalid strategy, shape, missing data or feature-count mismatch.
        ValueError
            No training rows.

        Complexity
        ----------
        O(n*d) for mean/constant; O(d*n*log(n)) for median/mode; O(n*d) space.

        Notes
        -----
        Inputs are never modified. The configured missing marker is replaced. For mean/median,
        NaNs are ignored when fitting; an entirely missing feature is rejected unless the
        strategy is constant. Other values remain unchanged.
        """
        X = self._validate(X)
        _require_samples(X, "SimpleImputer")
        mask = _missing_mask(X, self.missing_values)
        if self.strategy in ("mean", "median"):
            empty = np.flatnonzero(mask.all(axis=0))
            if empty.size:
                raise ValueError(
                    f"column(s) {empty.tolist()} contain only missing values; "
                    "use strategy='constant'"
                )
            data = np.where(mask, np.nan, X)
            reducer = _nanmean if self.strategy == "mean" else _nanmedian
            self.statistics_ = np.asarray(reducer(data, axis=0, ignore_nan=True))
        elif self.strategy == "constant":
            self.statistics_ = self._constant_fill(X)
        else:
            self.statistics_ = self._modes(X, mask)
        self.n_features_in_ = X.shape[1]
        return self

    def _constant_fill(self, X):
        default = "missing" if X.dtype.kind in "USO" else 0
        fill = default if self.fill_value is None else self.fill_value
        try:
            dtype = object if X.dtype.kind == "O" else np.result_type(
                X.dtype, np.asarray(fill).dtype
            )
        except TypeError as exc:
            raise TypeError(
                f"fill_value {fill!r} is incompatible with data of dtype {X.dtype}"
            ) from exc
        return np.full(X.shape[1], fill, dtype=dtype)

    def _modes(self, X, mask):
        # One np.unique per column; the loop is over features, not rows.
        # Ties go to the smallest value because np.unique returns sorted values.
        modes = []
        for j in range(X.shape[1]):
            column = X[~mask[:, j], j]
            if column.size == 0:
                raise ValueError(f"column {j} contains only missing values; use 'constant'")
            values, counts = np.unique(column, return_counts=True)
            modes.append(values[np.argmax(counts)])
        return np.array(modes, dtype=X.dtype if X.dtype.kind != "O" else object)

    def transform(self, X):
        """Return an independent imputed matrix.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Numeric matrix, or comparable categorical strings/numbers where supported.

        Returns
        -------
        X_out : ndarray, shape (n_samples, n_features)
            Numeric mean/median output is float64; other strategies promote the input and fill
            dtypes as needed.

        Raises
        ------
        TypeError
            Unsupported input or incomparable category types.
        ValueError
            Invalid strategy, shape, missing data or feature-count mismatch.
        NotFittedError
            Called before fitting (a RuntimeError, also a ValueError/AttributeError).

        Complexity
        ----------
        O(n*d) time and auxiliary space for n samples and d features.

        Notes
        -----
        Inputs are never modified. The configured missing marker is replaced. For mean/median,
        NaNs are ignored when fitting; an entirely missing feature is rejected unless the
        strategy is constant. Other values remain unchanged. Empty sample axes are accepted.
        """
        self._check_is_fitted("statistics_")
        X = self._validate(X)
        self._check_n_features(X)
        mask = _missing_mask(X, self.missing_values)
        out = np.where(mask, self.statistics_[np.newaxis, :], X)
        return out.astype(X.dtype) if X.dtype.kind == "f" else out


class OneHotEncoder(BaseTransformer):
    """Turn each categorical column into 0/1 indicator columns.

    A column with k categories becomes k output columns. Works on strings
    and numbers.

    Parameters
    handle_unknown : {"error", "ignore"}, default "error"
        What to do with a category not seen in fit: raise, or encode it as
        all zeros.

    Attributes
    categories_ : list of ndarray
        Sorted categories of each column.
    n_features_in_ : int

    Raises
    ValueError
        NaN in X, no rows in fit, an unseen category with
        handle_unknown="error", a bad handle_unknown or a column-count mismatch.
    TypeError
        If a column mixes types that can't be compared (e.g. str and float).

    Complexity
    fit: O(d * n log n). transform: O(n * K) time and space, K = total
    number of categories.

    Examples
    >>> import numpy as np
    >>> enc = OneHotEncoder().fit(np.array([["red"], ["blue"], ["red"]]))
    >>> enc.categories_[0]
    array(['blue', 'red'], dtype='<U4')
    >>> enc.transform(np.array([["red"], ["blue"]]))
    array([[0., 1.],
           [1., 0.]])
    >>> enc.get_feature_names_out(["colour"]).tolist()
    ['colour_blue', 'colour_red']
    """

    def __init__(self, handle_unknown="error"):
        self.handle_unknown = handle_unknown

    def _validate(self, X):
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(
                f"handle_unknown must be 'error' or 'ignore'; got {self.handle_unknown!r}"
            )
        X = check_categories(X, ensure_2d=True)
        if X.size and _missing_mask(X, np.nan).any():
            raise ValueError("OneHotEncoder does not accept NaN; impute missing values first")
        return X

    def fit(self, X, y=None):
        """Learn the column-wise transformation parameters and return self.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Numeric matrix, or comparable categorical strings/numbers where supported.
        y : optional
            Ignored; accepted for composition with supervised components.

        Returns
        -------
        self : OneHotEncoder

        Raises
        ------
        TypeError
            Unsupported input or incomparable category types.
        ValueError
            Invalid strategy, shape, missing data or feature-count mismatch.
        ValueError
            No training rows.

        Complexity
        ----------
        O(d*n*log(n)) time and O(n*d + K) space.

        Notes
        -----
        Inputs are never modified. Missing NaNs must be imputed first. Feature category order is
        sorted, with columns concatenated in input feature order. Unknown categories raise or
        become zeros according to handle_unknown.
        """
        X = self._validate(X)
        _require_samples(X, "OneHotEncoder")
        categories = []
        for j in range(X.shape[1]):
            try:
                categories.append(np.unique(X[:, j]))
            except TypeError as exc:
                raise TypeError(f"column {j} mixes values that cannot be compared") from exc
        self.categories_ = categories
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return an independent one-hot encoded matrix.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Numeric matrix, or comparable categorical strings/numbers where supported.

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, K), K = total fitted categories

        Raises
        ------
        TypeError
            Unsupported input or incomparable category types.
        ValueError
            Invalid strategy, shape, missing data or feature-count mismatch.
        NotFittedError
            Called before fitting (a RuntimeError, also a ValueError/AttributeError).

        Complexity
        ----------
        O(n*K) time and auxiliary space, including the output.

        Notes
        -----
        Inputs are never modified. Missing NaNs must be imputed first. Feature category order is
        sorted, with columns concatenated in input feature order. Unknown categories raise or
        become zeros according to handle_unknown. Empty sample axes are accepted.
        """
        self._check_is_fitted("categories_")
        X = self._validate(X)
        self._check_n_features(X)
        blocks = []
        for j, cats in enumerate(self.categories_):
            # (n, 1) == (1, k) broadcasts to an (n, k) table of matches
            block = X[:, j, np.newaxis] == cats[np.newaxis, :]
            unknown = ~block.any(axis=1)
            if self.handle_unknown == "error" and unknown.any():
                values = np.unique(X[unknown, j]).tolist()
                raise ValueError(f"column {j} has categories unseen during fit: {values}")
            blocks.append(block.astype(np.float64))
        return np.hstack(blocks) if blocks else np.empty((X.shape[0], 0))

    def get_feature_names_out(self, input_features=None):
        """Return output names in the exact one-hot column order.

        Parameters
        ----------
        input_features : sequence of str or None
            Input names, one per fitted feature; defaults to x0, x1, ... .

        Returns
        -------
        names : ndarray of str, shape (K,)
            Names such as city_Adelaide.

        Raises
        ------
        ValueError
            Name count differs from the fitted feature count.
        NotFittedError
            Called before fitting (a RuntimeError, also a ValueError/AttributeError).

        Complexity
        ----------
        O(K) time and output space, excluding string lengths.

        Notes
        -----
        Does not modify the fitted categories or supplied names. No numeric reduction or NaN
        processing.
        """
        self._check_is_fitted("categories_")
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in_)]
        if len(input_features) != self.n_features_in_:
            raise ValueError(
                f"input_features has {len(input_features)} names; expected {self.n_features_in_}"
            )
        return np.array(
            [f"{name}_{cat}" for name, cats in zip(input_features, self.categories_)
             for cat in cats]
        )
