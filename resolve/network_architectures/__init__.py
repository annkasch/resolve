from importlib import import_module


_EXPORTS = {
    "Autoencoder": ".autoencoder",
    "IsolationForestWrapper": ".isolation_forest",
    "VariationalAutoencoder": ".variational_auotencoder",
    "NeuralDensityRatioEstimator": ".neural_density_ratio_estimator",
    "SupervisedContrastive": ".supervised_contrastive",
    "InfoNCE": ".info_nce",
    "FTTransformer": ".ft_transformer",
    "TransformerEncoder": ".transformer_encoder",
    "XGBoostWrapper": ".xgboost",
    "XGBWithLeafCache": ".xgboost",
    "LightGBMWrapper": ".lightgbm",
    "LGBMWithLeafCache": ".lightgbm",
    "LeafCache": ".leaf_cache",
    "GNNBinaryClassifier": ".graph_neural_network",
    "NormalizingFlowClassifier": ".nf_classifier",
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
