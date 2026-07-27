import torch

from resolve.helpers.splitter import Splitter


def test_global_split_is_reproducible_and_complete():
    values = torch.arange(12)
    groups = torch.arange(12)
    splitter = Splitter(mode="global", seed=17)

    first_train, first_test = splitter.train_test_split(
        values,
        groups=groups,
        test_size=0.25,
    )
    second_train, second_test = splitter.train_test_split(
        values,
        groups=groups,
        test_size=0.25,
    )

    torch.testing.assert_close(first_train, second_train)
    torch.testing.assert_close(first_test, second_test)
    assert sorted(torch.cat((first_train, first_test)).tolist()) == list(range(12))


def test_groupwise_split_holds_out_each_group_without_overlap():
    values = torch.arange(12)
    groups = torch.repeat_interleave(torch.arange(3), repeats=4)

    train, test, train_indices, test_indices = Splitter(
        mode="batch_wise",
        seed=23,
    ).train_test_groupwise_split(
        values,
        groups=groups,
        test_size=0.25,
        return_indices=True,
    )

    assert train_indices.numel() == 9
    assert test_indices.numel() == 3
    assert set(train_indices.tolist()).isdisjoint(test_indices.tolist())
    assert sorted(torch.cat((train_indices, test_indices)).tolist()) == list(range(12))
    torch.testing.assert_close(train, values[train_indices])
    torch.testing.assert_close(test, values[test_indices])
    torch.testing.assert_close(
        torch.bincount(groups[test_indices], minlength=3),
        torch.ones(3, dtype=torch.long),
    )
