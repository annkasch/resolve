from importlib import import_module


_EXPORTS = {
    "PCEMultiFidelityModel": ".bayes_pce_multi_fidelity_model",
    "PCEMultiFidelityModelVisualizer": ".bayes_pce_multi_fidelity_visualizer",
    "MFGPModel": ".multi_fidelity_surrogate_model",
    "MFGPInequalityConstraints": ".multi_fidelity_surrogate_model",
    "GPMultiFidelityVisualizer": ".multi_fidelity_visualizer",
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
