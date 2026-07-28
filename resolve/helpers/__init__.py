from importlib import import_module


_EXPORTS = {
    "Trainer": ".training_wrapper",
    "BatchCollection": ".batch_types",
    "ContextSet": ".batch_types",
    "DataLoaderManager": ".dataloader_manager",
    "DataValidationError": ".data_source",
    "AsymmetricFocalWithFPPenalty": ".losses",
    "bce_with_logits": ".losses",
    "gaussian_nll": ".losses",
    "recon_loss_mse": ".losses",
    "skip_loss": ".losses",
    "brier": ".losses",
    "logit_normal_bernoulli_nll": ".losses",
    "zero_loss": ".losses",
    "InMemoryIterableData": ".iterable_dataset",
    "Normalizer": ".normalizer",
    "QuerySet": ".batch_types",
    "ModelsManager": ".model_manager",
    "ModelVisualizer": ".model_visualizer",
    "UMAPAnalyzer": ".feature_analysis",
    "Sampler": ".sampler",
    "Splitter": ".splitter",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *__all__))
