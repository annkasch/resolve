from importlib import import_module


_EXPORTS = {
    "AttnCNP": ".attn_cnp_memory_bank",
    "MemoryBank": ".memory_bank",
    "TargetTransformerDecoder": ".transformer_decoder",
    "TreeConditionedCNP": ".tree_conditioned_neural_process_model",
    "BDTFTTransformer": ".bdt_transformer",
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
