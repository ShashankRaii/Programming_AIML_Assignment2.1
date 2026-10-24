"""Unit tests for numcompute.io."""

import numpy as np
import pytest

from numcompute.io import load_csv, load_csv_chunks, read_header

# Fixtures
@pytest.fixture
def write_file(tmp_path):
    """Return a helper that writes text to a temp file and returns its path."""

    def _write(text, name="data.csv"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write


BASIC = "a,b,c\n1,2,3\n4,5,6\n"
WITH_MISSING = "a,b,c\n1,,3\n4,5,6\n7,8,\n"



# load_csv: normal behaviour
def test_load_basic_comma(write_file):
    X = load_csv(write_file(BASIC), skip_header=1)
    assert X.shape == (2, 3)
    assert X.dtype == float
    np.testing.assert_array_equal(X, [[1, 2, 3], [4, 5, 6]])


def test_load_tab_delimiter(write_file):
    path = write_file("a\tb\n1\t2\n3\t4\n", name="data.tsv")
    X = load_csv(path, delimiter="\t", skip_header=1)
    np.testing.assert_array_equal(X, [[1, 2], [3, 4]])


def test_missing_filled_with_nan_by_default(write_file):
    X = load_csv(write_file(WITH_MISSING), skip_header=1)
    assert X.shape == (3, 3)
    assert np.isnan(X[0, 1]) and np.isnan(X[2, 2])
    assert np.isnan(X).sum() == 2


def test_missing_filled_with_custom_value(write_file):
    X = load_csv(write_file(WITH_MISSING), skip_header=1, fill_value=-1)
    assert X[0, 1] == -1 and X[2, 2] == -1
    assert not np.isnan(X).any()


def test_missing_skip_drops_rows(write_file):
    X = load_csv(write_file(WITH_MISSING), skip_header=1, missing="skip")
    np.testing.assert_array_equal(X, [[4, 5, 6]])


def test_skip_ignores_fill_value(write_file):
    X = load_csv(
        write_file(WITH_MISSING), skip_header=1, missing="skip", fill_value=0
    )
    assert X.shape == (1, 3)


def test_usecols_selects_columns(write_file):
    X = load_csv(write_file(BASIC), skip_header=1, usecols=(0, 2))
    np.testing.assert_array_equal(X, [[1, 3], [4, 6]])


def test_string_dtype_for_categorical(write_file):
    path = write_file("city,size\nAdelaide,S\nPerth,M\n")
    X = load_csv(path, skip_header=1, dtype=str)
    assert X.shape == (2, 2)
    assert X[0, 0] == "Adelaide" and X[1, 1] == "M"


def test_whitespace_around_values_is_stripped(write_file):
    X = load_csv(write_file("a,b\n 1 , 2 \n"), skip_header=1)
    np.testing.assert_array_equal(X, [[1, 2]])

# load_csv: edge cases

def test_single_row_is_2d(write_file):
    X = load_csv(write_file("1,2,3\n"))
    assert X.shape == (1, 3)


def test_single_column_is_2d(write_file):
    X = load_csv(write_file("1\n2\n3\n"))
    assert X.shape == (3, 1)


def test_empty_file_returns_empty_array(write_file):
    X = load_csv(write_file(""))
    assert X.shape == (0, 0)


def test_header_only_file_returns_empty_array(write_file):
    X = load_csv(write_file("a,b,c\n"), skip_header=1)
    assert X.size == 0


def test_all_rows_missing_with_skip(write_file):
    X = load_csv(write_file("1,\n,2\n"), missing="skip")
    assert X.shape[0] == 0

# load_csv: errors

def test_file_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_csv("does_not_exist.csv")


def test_invalid_missing_option(write_file):
    with pytest.raises(ValueError, match="missing must be one of"):
        load_csv(write_file(BASIC), missing="drop")


def test_negative_skip_header(write_file):
    with pytest.raises(ValueError, match="skip_header"):
        load_csv(write_file(BASIC), skip_header=-1)


def test_non_string_delimiter(write_file):
    with pytest.raises(TypeError, match="delimiter"):
        load_csv(write_file(BASIC), delimiter=1)


def test_empty_delimiter(write_file):
    with pytest.raises(ValueError, match="delimiter"):
        load_csv(write_file(BASIC), delimiter="")


def test_inconsistent_row_lengths_raise(write_file):
    with pytest.raises(ValueError):
        load_csv(write_file("1,2,3\n4,5\n"))

# read_header

def test_read_header(write_file):
    assert read_header(write_file("age , income,city\n1,2,3\n")) == [
        "age",
        "income",
        "city",
    ]


def test_read_header_empty_file(write_file):
    assert read_header(write_file("")) == []


# load_csv_chunks
def test_chunks_match_full_load(write_file):
    rows = "\n".join(f"{i},{i * 2}" for i in range(10))
    path = write_file("x,y\n" + rows + "\n")
    chunks = list(load_csv_chunks(path, chunk_size=3, skip_header=1))
    assert [c.shape[0] for c in chunks] == [3, 3, 3, 1]
    np.testing.assert_array_equal(
        np.vstack(chunks), load_csv(path, skip_header=1)
    )


def test_chunk_larger_than_file(write_file):
    chunks = list(load_csv_chunks(write_file(BASIC), 100, skip_header=1))
    assert len(chunks) == 1 and chunks[0].shape == (2, 3)


def test_chunks_skip_missing_rows(write_file):
    chunks = list(
        load_csv_chunks(write_file(WITH_MISSING), 2, skip_header=1, missing="skip")
    )
    np.testing.assert_array_equal(np.vstack(chunks), [[4, 5, 6]])


def test_chunks_ignore_blank_lines(write_file):
    chunks = list(load_csv_chunks(write_file("1,2\n\n\n\n3,4\n"), 2))
    np.testing.assert_array_equal(np.vstack(chunks), [[1, 2], [3, 4]])


def test_chunks_empty_file(write_file):
    assert list(load_csv_chunks(write_file(""), 5)) == []


@pytest.mark.parametrize("bad, err", [(0, ValueError), (-2, ValueError), (2.5, TypeError)])
def test_invalid_chunk_size(write_file, bad, err):
    with pytest.raises(err, match="chunk_size"):
        list(load_csv_chunks(write_file(BASIC), bad))