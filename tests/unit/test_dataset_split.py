import numpy as np
import pytest

from mlp.dataset import SupervisedDataset, split_dataset
from mlp.exceptions import DatasetError


def make_dataset(sample_count: int) -> SupervisedDataset:
    identifiers = np.arange(sample_count, dtype=np.float64).reshape(-1, 1)
    return SupervisedDataset(
        identifiers,
        identifiers * 10.0,
        ("id",),
        ("target",),
    )


def all_identifiers(split) -> list[int]:
    return [
        *split.training.inputs[:, 0].astype(int).tolist(),
        *split.validation.inputs[:, 0].astype(int).tolist(),
        *split.test.inputs[:, 0].astype(int).tolist(),
    ]


def test_split_uses_remainder_for_test_without_losing_examples() -> None:
    split = split_dataset(make_dataset(11), 70, 15, 15, seed=42)

    assert split.training.sample_count == 7
    assert split.validation.sample_count == 1
    assert split.test.sample_count == 3
    assert sorted(all_identifiers(split)) == list(range(11))
    assert len(set(all_identifiers(split))) == 11


def test_real_dataset_counts_follow_documented_rounding() -> None:
    split = split_dataset(make_dataset(179), 70, 15, 15, seed=42)
    assert (
        split.training.sample_count,
        split.validation.sample_count,
        split.test.sample_count,
    ) == (125, 26, 28)


def test_inputs_and_outputs_remain_paired() -> None:
    split = split_dataset(make_dataset(20), 60, 20, 20, seed=5)
    for subset in (split.training, split.validation, split.test):
        np.testing.assert_array_equal(
            subset.expected_outputs[:, 0], subset.inputs[:, 0] * 10.0
        )


def test_same_seed_reproduces_and_different_seed_changes_split() -> None:
    dataset = make_dataset(30)
    first = split_dataset(dataset, 70, 15, 15, seed=8)
    second = split_dataset(dataset, 70, 15, 15, seed=8)
    different = split_dataset(dataset, 70, 15, 15, seed=9)

    np.testing.assert_array_equal(first.training.inputs, second.training.inputs)
    assert not np.array_equal(first.training.inputs, different.training.inputs)


@pytest.mark.parametrize(
    "percentages",
    [(70, 20, 20), (-1, 50, 51), (101, 0, -1), (float("nan"), 50, 50)],
)
def test_rejects_invalid_percentages(percentages) -> None:
    with pytest.raises(DatasetError):
        split_dataset(make_dataset(10), *percentages, seed=1)


def test_rejects_small_dataset_that_would_empty_requested_subset() -> None:
    with pytest.raises(DatasetError, match="validação vazia"):
        split_dataset(make_dataset(2), 70, 15, 15, seed=1)


def test_rejects_zero_percentage_for_required_subset() -> None:
    with pytest.raises(DatasetError, match="maiores que zero"):
        split_dataset(make_dataset(10), 80, 20, 0, seed=1)


def test_subsets_are_independent_read_only_copies() -> None:
    dataset = make_dataset(10)
    split = split_dataset(dataset, 60, 20, 20, seed=1)

    assert not np.shares_memory(dataset.inputs, split.training.inputs)
    assert not split.training.inputs.flags.writeable

