import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("module_name", "expected_export"),
    [
        ("resolve.helpers", "Normalizer"),
        ("resolve.network_architectures", "Autoencoder"),
        (
            "resolve.conditional_neural_process_family",
            "ConditionalNeuralProcess",
        ),
        ("resolve.regression_models", "MFGPModel"),
        ("resolve.dev", "MemoryBank"),
    ],
)
def test_package_namespaces_expose_public_api(module_name, expected_export):
    module = importlib.import_module(module_name)

    assert expected_export in module.__all__
    assert expected_export in dir(module)


def test_network_namespace_does_not_eagerly_import_optional_backends():
    optional_modules = {
        "resolve.network_architectures.graph_neural_network",
        "resolve.network_architectures.lightgbm",
    }
    for module_name in optional_modules:
        sys.modules.pop(module_name, None)

    importlib.reload(importlib.import_module("resolve.network_architectures"))

    assert optional_modules.isdisjoint(sys.modules)
