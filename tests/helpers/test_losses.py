import math

import pytest
import torch

from resolve.helpers.losses import (
    AsymmetricFocalWithFPPenalty,
    bce_with_logits,
    brier,
    gaussian_nll,
    logit_normal_bernoulli_nll,
    recon_loss_mse,
    skip_loss,
    zero_loss,
)


def test_binary_losses_return_per_sample_values_and_probabilities():
    logits = torch.zeros(2)
    targets = torch.tensor([0.0, 1.0])

    bce, bce_probabilities = bce_with_logits(logits, targets)
    brier_score, brier_probabilities = brier(logits, targets)

    torch.testing.assert_close(bce, torch.full((2,), math.log(2)))
    torch.testing.assert_close(bce_probabilities, torch.full((2,), 0.5))
    torch.testing.assert_close(brier_score, torch.full((2,), 0.25))
    torch.testing.assert_close(brier_probabilities, torch.full((2,), 0.5))


def test_gaussian_nll_requires_location_and_scale():
    targets = torch.zeros(2)

    with pytest.raises(ValueError, match=r"z=\[mu, sigma\]"):
        gaussian_nll(torch.zeros(2), targets)


def test_gaussian_nll_is_minimized_at_the_mean_for_unit_scale():
    mean = torch.tensor([-1.0, 1.0])
    scale = torch.ones(2)

    loss, prediction = gaussian_nll([mean, scale], mean)

    torch.testing.assert_close(
        loss,
        torch.full((2,), 0.5 * math.log(2 * math.pi)),
    )
    torch.testing.assert_close(prediction, mean)


def test_skip_and_zero_losses_preserve_per_sample_shape():
    values = torch.ones(3)

    skipped, skipped_prediction = skip_loss([values], values)
    zero, zero_prediction = zero_loss([values], values)

    assert skipped.shape == (3,)
    assert torch.isnan(skipped).all()
    assert torch.isnan(skipped_prediction).all()
    torch.testing.assert_close(zero, torch.zeros(3))
    torch.testing.assert_close(zero_prediction, torch.zeros(3))


def test_reconstruction_loss_averages_over_features():
    prediction = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    target = torch.tensor([[1.0, 1.0], [4.0, 4.0]])

    loss, reported_loss = recon_loss_mse([prediction], None, target)

    torch.testing.assert_close(loss, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(reported_loss, loss)


def test_focal_loss_honors_reduction():
    logits = torch.tensor([-1.0, 1.0])
    targets = torch.tensor([0.0, 1.0])

    per_sample = AsymmetricFocalWithFPPenalty(reduction="none")(logits, targets)
    mean = AsymmetricFocalWithFPPenalty(reduction="mean")(logits, targets)

    assert per_sample.shape == (2,)
    assert torch.isfinite(per_sample).all()
    torch.testing.assert_close(mean, per_sample.mean())


def test_logit_normal_loss_rejects_unsupported_quadrature_size():
    mean = torch.zeros(2)
    scale = torch.ones(2)
    targets = torch.zeros(2)

    with pytest.raises(ValueError, match="not supported"):
        logit_normal_bernoulli_nll([mean, scale], targets, num_points=7)
