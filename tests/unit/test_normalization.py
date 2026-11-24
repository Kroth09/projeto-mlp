import numpy as np
import pytest

from mlp.dataset import DatasetSplit, DatasetSubset
from mlp.exceptions import NormalizationError
from mlp.normalization import ZScoreNormalizer, normalize_split


def test_fit_transform_centers_and_scales_training_columns() -> None:
    inputs = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]])
    original = inputs.copy()
    normalizer = ZScoreNormalizer()

    normalized = normalizer.fit_transform(inputs)

    np.testing.assert_allclose(normalized[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(normalized[:, 0].std(ddof=0), 1.0, atol=1e-12)
    np.testing.assert_array_equal(normalized[:, 1], np.zeros(3))
    np.testing.assert_array_equal(normalizer.means, [2.0, 10.0])
    np.testing.assert_allclose(normalizer.standard_deviations, [np.sqrt(8 / 3), 1.0])
    np.testing.assert_array_equal(inputs, original)


def test_transform_requires_fit_and_matching_dimensions() -> None:
    normalizer = ZScoreNormalizer()
    with pytest.raises(NormalizationError, match="ajustado"):
        normalizer.transform(np.array([[1.0]]))

    normalizer.fit(np.array([[0.0, 1.0], [2.0, 3.0]]))
    with pytest.raises(NormalizationError, match="atributos incompatível"):
        normalizer.transform(np.array([[1.0]]))


def test_serialization_round_trip_preserves_transformation() -> None:
    training = np.array([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]])
    values = np.array([[7.0, 16.0]])
    fitted = ZScoreNormalizer().fit(training)

    restored = ZScoreNormalizer.from_dict(fitted.to_dict())

    np.testing.assert_allclose(restored.transform(values), fitted.transform(values))


def test_custom_zero_std_divisor_is_respected_and_serialized() -> None:
    fitted = ZScoreNormalizer(zero_std_divisor=2.5).fit(
        np.array([[7.0], [7.0]])
    )

    assert fitted.standard_deviations[0] == 2.5
    assert fitted.to_dict()["zero_std_divisor"] == 2.5
    restored = ZScoreNormalizer.from_dict(fitted.to_dict())
    assert restored.zero_std_divisor == 2.5
    np.testing.assert_array_equal(restored.transform(np.array([[7.0]])), [[0.0]])


@pytest.mark.parametrize("invalid_divisor", [0.0, -1.0, float("inf"), True, "texto"])
def test_rejects_invalid_zero_std_divisor(invalid_divisor) -> None:
    with pytest.raises(NormalizationError, match="zero_std_divisor"):
        ZScoreNormalizer(zero_std_divisor=invalid_divisor)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"method": "minmax", "means": [0], "standard_deviations": [1]},
        {"method": "zscore", "means": [0, 1], "standard_deviations": [1]},
        {"method": "zscore", "means": [0], "standard_deviations": [0]},
        {"method": "zscore", "means": [0], "standard_deviations": [float("inf")]},
        {
            "method": "zscore",
            "means": [0],
            "standard_deviations": [1],
            "zero_std_divisor": True,
        },
    ],
)
def test_rejects_invalid_serialized_parameters(data) -> None:
    with pytest.raises(NormalizationError):
        ZScoreNormalizer.from_dict(data)


def test_normalize_split_uses_only_training_statistics_and_preserves_outputs() -> None:
    split = DatasetSplit(
        training=DatasetSubset(np.array([[0.0], [2.0]]), np.array([[10.0], [20.0]])),
        validation=DatasetSubset(np.array([[100.0]]), np.array([[30.0]])),
        test=DatasetSubset(np.array([[200.0]]), np.array([[40.0]])),
        seed=42,
    )

    normalized, normalizer = normalize_split(split)

    np.testing.assert_array_equal(normalizer.means, [1.0])
    np.testing.assert_array_equal(normalizer.standard_deviations, [1.0])
    np.testing.assert_array_equal(normalized.training.inputs, [[-1.0], [1.0]])
    np.testing.assert_array_equal(normalized.validation.inputs, [[99.0]])
    np.testing.assert_array_equal(normalized.test.inputs, [[199.0]])
    np.testing.assert_array_equal(
        normalized.training.expected_outputs, split.training.expected_outputs
    )
    np.testing.assert_array_equal(
        normalized.validation.expected_outputs, split.validation.expected_outputs
    )
    np.testing.assert_array_equal(
        normalized.test.expected_outputs, split.test.expected_outputs
    )

