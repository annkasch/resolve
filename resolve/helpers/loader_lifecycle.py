from __future__ import annotations

import weakref

from torch.utils.data import DataLoader


def _shutdown_iterator(iterator):
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


class _TrackedDataLoaderIterator:
    def __init__(self, iterator, owner):
        self._iterator = iterator
        self._owner = weakref.ref(owner)
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except StopIteration:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        iterator = self._iterator
        self._iterator = None
        owner = self._owner()
        if owner is not None:
            owner._iteration_finished(self)
        if owner is None or not owner.persistent_workers:
            _shutdown_iterator(iterator)

    def __del__(self):
        self.close()


class ReusableDataLoader(DataLoader):
    """A DataLoader that exposes whether one of its iterators is active."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iteration_active = False
        self._active_iterator = None

    @property
    def iteration_active(self):
        return self._iteration_active

    def __iter__(self):
        if self._iteration_active:
            raise RuntimeError(
                "This DataLoader already has an active iterator. Finish or "
                "close it before starting another iteration."
            )
        tracked = _TrackedDataLoaderIterator(super().__iter__(), self)
        self._iteration_active = True
        self._active_iterator = weakref.ref(tracked)
        return tracked

    def _iteration_finished(self, iterator):
        active = (
            self._active_iterator()
            if self._active_iterator is not None
            else None
        )
        if active is iterator:
            self._iteration_active = False
            self._active_iterator = None

    def close_active_iteration(self):
        active = (
            self._active_iterator()
            if self._active_iterator is not None
            else None
        )
        if active is not None:
            active.close()
        else:
            self._iteration_active = False
            self._active_iterator = None


def shutdown_loader(loader):
    """Stop a cached DataLoader, isolating PyTorch's private shutdown API.

    PyTorch does not currently expose a public method for eagerly terminating
    persistent DataLoader workers. Keep the compatibility-sensitive access in
    this helper so future PyTorch changes have one repair point.
    """

    if loader is None:
        return
    close_active = getattr(loader, "close_active_iteration", None)
    if callable(close_active):
        close_active()

    iterator = getattr(loader, "_iterator", None)
    _shutdown_iterator(iterator)
    if hasattr(loader, "_iterator"):
        loader._iterator = None
