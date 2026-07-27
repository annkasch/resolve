from importlib import import_module


_EXPORTS = {
    "AttnLNP": ".attn_latent_neural_process_model",
    "AttnCNP": ".attn_neural_process_model",
    "DeterministicEncoder": ".conditional_neural_process_model",
    "DeterministicDecoder": ".conditional_neural_process_model",
    "ConditionalNeuralProcess": ".conditional_neural_process_model",
    "CrossAttention": ".class_attention",
    "CrossAttentionDual": ".class_attention",
    "SimplePoolAttention": ".class_attention",
    "FeatureEncoder": ".feature_encoder",
    "MLPEncoder": ".feature_encoder",
    "MLP": ".feature_encoder",
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
