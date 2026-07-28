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
