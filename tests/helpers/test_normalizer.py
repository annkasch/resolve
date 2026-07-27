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
