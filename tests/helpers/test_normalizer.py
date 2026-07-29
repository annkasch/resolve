import pytest
import torch

from resolve.helpers.normalizer import Normalizer


def test_zscore_fit_transform_standardizes_each_feature():
    values = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    transformed = Normalizer("zscore").fit_transform(values, "features")

    torch.testing.assert_close(
        transformed.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        transformed.std(dim=0, unbiased=False),
        torch.ones(2),
        atol=1e-6,
        rtol=0,
    )


def test_minmax_inverse_transform_restores_input():
    values = torch.tensor(
        [
            [1.0, -4.0],
            [3.0, 0.0],
            [5.0, 8.0],
        ]
    )
    normalizer = Normalizer("minmax")

    transformed = normalizer.fit_transform(values, "features")
    restored = normalizer.inverse_transform(transformed, "features")

    torch.testing.assert_close(transformed.min(dim=0).values, torch.zeros(2))
    torch.testing.assert_close(transformed.max(dim=0).values, torch.ones(2))
    torch.testing.assert_close(restored, values)


def test_default_normalizer_leaves_values_unchanged():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    transformed = Normalizer().fit_transform(values, "features")

    torch.testing.assert_close(transformed, values)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_transform_and_inverse_transform_preserve_tensor_metadata(dtype):
    values = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=dtype,
    )
    normalizer = Normalizer("zscore")
    normalizer.fit(values, "features")

    transformed = normalizer.transform(values, "features")
    restored = normalizer.inverse_transform(transformed, "features")

    assert transformed.dtype == values.dtype
    assert transformed.device == values.device
    assert restored.dtype == values.dtype
    assert restored.device == values.device
    torch.testing.assert_close(restored, values)


@pytest.mark.parametrize("method", ("zscore", "minmax", None))
def test_chunked_fit_matches_single_tensor_fit(method):
    values = torch.tensor(
        [
            [1.0, 100.0],
            [2.0, 50.0],
            [7.0, -20.0],
            [9.0, 10.0],
            [12.0, 5.0],
        ],
        dtype=torch.float32,
    )
    labels = ("first", "second")
    direct = Normalizer(method)
    direct.fit(values, "features", labels)
    chunked = Normalizer(method)
    chunked.fit_chunks(
        torch.split(values, (2, 1, 2)),
        "features",
        labels,
    )

    torch.testing.assert_close(
        chunked.transform(values, "features"),
        direct.transform(values, "features"),
        rtol=1e-6,
        atol=1e-6,
    )
    assert chunked.feature_labels["features"] == labels
    assert (
        chunked.scalers["features"].n_samples_seen_
        == values.shape[0]
    )
