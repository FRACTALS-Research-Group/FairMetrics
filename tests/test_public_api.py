import importlib
import pytest

MODULES = [
    "fairmetrics",
    "fairmetrics.metrics",
    "fairmetrics.accuracy",
    "fairmetrics.independence",
    "fairmetrics.separation",
    "fairmetrics.sufficiency",
    "fairmetrics.tests",
    "fairmetrics_experiments",
    "fairmetrics_experiments.binarization",
    "fairmetrics_experiments.fairness_tests",
    "fairmetrics_experiments.run_hpo",
]

CORE_SYMBOLS = [
    "demographic_parity",
    "equal_opportunity",
    "multiclass_equal_opportunity",
    "binary_predictive_parity",
    "multiclass_predictive_parity",
    "equal_accuracy",
]

EXPERIMENT_SYMBOLS = [
    ("fairmetrics_experiments.fairness_tests", "fairness_test_suite"),
    ("fairmetrics_experiments.run_hpo", "run_hpo"),
    ("fairmetrics_experiments.run_hpo", "main"),
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", ["fairmetrics", "fairmetrics.tests"])
def test_all_exports_exist(module):
    mod = importlib.import_module(module)
    missing = [n for n in getattr(mod, "__all__", []) if not hasattr(mod, n)]
    assert not missing, f"{module}.__all__ contiene nomi non definiti: {missing}"


@pytest.mark.parametrize("name", CORE_SYMBOLS)
def test_core_symbols(name):
    import fairmetrics
    assert callable(getattr(fairmetrics, name))


@pytest.mark.parametrize("module,name", EXPERIMENT_SYMBOLS)
def test_experiment_symbols(module, name):
    mod = importlib.import_module(module)
    assert callable(getattr(mod, name))
