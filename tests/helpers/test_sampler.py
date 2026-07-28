import numpy as np
import torch

from resolve.helpers.sampler import Sampler


def test_batch_order_depends_only_on_requested_epoch():
    indices = torch.arange(30)
    sampler = Sampler(["== 1"], seed=17, shuffle="global")

    epoch_five, _, _ = sampler.build_batches(
        indices,
        batch_size=7,
        epoch=5,
    )
    sampler.build_batches(indices, batch_size=7, epoch=1)
    repeated, _, _ = sampler.build_batches(
        indices,
        batch_size=7,
        epoch=5,
    )

    assert all(
        torch.equal(left, right)
        for left, right in zip(epoch_five, repeated)
    )


def test_positive_sampling_is_stateless_and_preserves_sticky_overlap():
    sampler = Sampler(["== 1"], seed=23, shuffle="global")
    indices = torch.arange(40)
    groups = torch.zeros(40, dtype=torch.long)
    targets = torch.zeros((40, 1))
    targets[::10] = 1

    sampler.groupaware_pos_sampling(
        groups,
        targets,
        indices,
        target_pos_frac=0.25,
        max_pos_reuse_per_epoch=2,
        epoch=0,
    )
    epoch_four, _, _, _ = sampler.groupaware_pos_sampling(
        groups,
        targets,
        indices,
        target_pos_frac=0.25,
        max_pos_reuse_per_epoch=2,
        epoch=4,
    )
    repeated, _, _, _ = sampler.groupaware_pos_sampling(
        groups,
        targets,
        indices,
        target_pos_frac=0.25,
        max_pos_reuse_per_epoch=2,
        epoch=4,
    )
    next_epoch, _, _, _ = sampler.groupaware_pos_sampling(
        groups,
        targets,
        indices,
        target_pos_frac=0.25,
        max_pos_reuse_per_epoch=2,
        epoch=5,
    )

    torch.testing.assert_close(epoch_four, repeated)
    assert not torch.equal(epoch_four, next_epoch)

    negative_mask = targets[:, 0] == 0
    epoch_four_negatives = {
        index
        for index in epoch_four.tolist()
        if negative_mask[index]
    }
    next_epoch_negatives = {
        index
        for index in next_epoch.tolist()
        if negative_mask[index]
    }
    assert epoch_four_negatives != next_epoch_negatives
    assert epoch_four_negatives & next_epoch_negatives


def test_beta_mixup_does_not_consume_global_rng_state():
    sampler = Sampler(["== 1"], seed=29)
    theta = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    phi = (theta + 100).clone()
    targets = torch.tensor(
        [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]
    )

    torch.manual_seed(101)
    np.random.seed(101)
    expected_torch = torch.rand(3)
    expected_numpy = np.random.random(3)

    torch.manual_seed(101)
    np.random.seed(101)
    sampler._mix_negatives_positives(
        theta,
        phi,
        targets,
        use_beta=[0.2, 0.4],
        seed=31,
        mix_ratio=0.5,
    )

    torch.testing.assert_close(torch.rand(3), expected_torch)
    np.testing.assert_allclose(np.random.random(3), expected_numpy)
