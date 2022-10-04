from typing import Dict

import pandas as pd
import pytest
import yaml

from ambrozia.designer import Designer, design, design_binary


@pytest.mark.smoke
def test_instance():
    """
    Check that simple instance without args work
    """
    designer = Designer()


@pytest.mark.smoke
def test_constructors(ltv_and_retention_dataset):
    """
    Test different constructors
    """
    # Use only dataframe
    designer = Designer(dataframe=ltv_and_retention_dataset)

    # Use only metrics names
    designer = Designer(metrics="metric name")
    designer = Designer(metrics=["some metric"])

    # Use many params
    designer = Designer(
        dataframe=ltv_and_retention_dataset, metrics="metric name", sizes=[100, 200], effects=1.05, method="empiric"
    )


@pytest.mark.smoke
def test_corret_type(designer_simple_table, designer_ltv):
    """
    Check, that method run is callable and return correct type.
    """
    # Check that for one metic it is DataFrame
    assert isinstance(designer_simple_table.run("size"), pd.DataFrame)
    # Check that for many metrics it is a Dictionary
    metrics = ["retention", "LTV"]
    assert isinstance(designer_ltv.run("effect", metrics=metrics), Dict)


@pytest.mark.unit
@pytest.mark.parametrize(
    "param_to_design, designer, expected_value",
    [
        ("size", pytest.lazy_fixture("designer_simple_table"), 736),
        ("effect", pytest.lazy_fixture("designer_simple_table"), "54.3%"),
        ("power", pytest.lazy_fixture("designer_simple_table"), "17.7%"),
        ("size", pytest.lazy_fixture("designer_ltv"), 1552),
        ("effect", pytest.lazy_fixture("designer_ltv"), "17.6%"),
        ("power", pytest.lazy_fixture("designer_ltv"), "35.6%"),
    ],
)
def test_run_theory(param_to_design, expected_value, designer):
    """
    Some tests for method run and theory approach.
    """
    if param_to_design != "power":
        assert designer.run(param_to_design).iloc[0][0] == expected_value
    else:
        assert designer.run(param_to_design).iloc[0].iloc[0] == expected_value


@pytest.mark.unit
@pytest.mark.parametrize(
    "param_to_design, method, size",
    [
        ("effect", "empiric", 200),
        ("effect", "theory", 200),
        ("power", "theory", 200),
    ],
)
def test_as_numeric(param_to_design, method, size, ltv_and_retention_dataset):
    """
    Check flag as_numeric works correctly
    """
    run_kwargs = {}
    if method == "empiric":
        run_kwargs["random_seed"] = 1
    designer = Designer(ltv_and_retention_dataset, sizes=size, effects=1.2, metrics="LTV")
    float_value: float = designer.run(param_to_design, method, as_numeric=True, **run_kwargs).iloc[0, 0]
    designer = Designer(ltv_and_retention_dataset, sizes=size, effects=1.2, metrics="LTV")
    string_value: str = designer.run(param_to_design, method, as_numeric=False, **run_kwargs).iloc[0, 0]
    if param_to_design == "power":
        assert "{:.1f}".format((float_value) * 100) + "%" == string_value
    else:
        assert "{:.1f}".format((float_value - 1) * 100) + "%" == string_value


@pytest.mark.unit
@pytest.mark.parametrize("to_design", ["size", "effect", "power"])
@pytest.mark.parametrize("method", ["theory", "empiric"])
@pytest.mark.parametrize("effects", [1.05, 1.1, 1.2])
@pytest.mark.parametrize("sizes", [50, 100, 500, 1000])
def test_every_type_run(to_design, method, effects, sizes, designer_ltv):
    """
    Use cortesian product of all params to check, that all posible combinations are working
    """
    if method != "empiric":
        designer_ltv.run(to_design, method=method, effects=effects, sizes=sizes)
    else:
        designer_ltv.run(to_design, method=method, effects=effects, sizes=sizes, parallel=False, bs_samples=10)


@pytest.mark.unit
@pytest.mark.parametrize("to_design", ["size", "effect", "power"])
@pytest.mark.parametrize("effects", [1.01, 1.05, 1.1])
@pytest.mark.parametrize("sizes", [500, 5000, 10000])
def test_binary(to_design, effects, sizes, designer_ltv, designer_simple_table, designer_ltv_spark):
    """
    Tests for binary apporach for designing an experiment
    """
    designer_ltv.run(to_design, method="binary", effects=effects, sizes=sizes, metrics="retention")
    designer_simple_table.run(to_design, method="binary", effects=effects, sizes=sizes, metrics="retention")
    designer_ltv_spark.run(to_design, method="binary", effects=effects, sizes=sizes, metrics="retention")


@pytest.mark.unit
def test_config_constructor(ltv_and_retention_dataset, designer_ltv):
    """
    Config constructor testing
    """
    with open("./tests/config.yaml", "r") as stream:
        designer_from_config = yaml.load(stream, Loader=yaml.Loader)
    designer_from_config.set_dataframe(ltv_and_retention_dataset)
    value_from_config = designer_from_config.run("size")
    assert value_from_config.equals(designer_ltv.run("size"))


@pytest.mark.unit
def test_design_function(ltv_and_retention_dataset, designer_ltv):
    """
    Design function must reteutn the same, that designer class
    """
    result_design_func = design(
        "size", dataframe=ltv_and_retention_dataset, metrics="LTV", method="theory", effects=1.01
    )

    result_designer_class = designer_ltv.run("size", effects=1.01, metrics="LTV")
    assert result_designer_class.equals(result_design_func)


@pytest.mark.smoke
@pytest.mark.parametrize("to_design", ["size", "effect", "power"])
@pytest.mark.parametrize("effects", [1.05, [1.01, 1.05]])
@pytest.mark.parametrize("sizes", [200, [100, 200]])
@pytest.mark.parametrize("beta", [0.2, [0.1, 0.2]])
def test_design_binary_function(to_design, effects, sizes, beta):
    """
    Design binary function smoke test
    """
    pa: float = 0.3
    design_binary(to_design=to_design, prob_a=pa, sizes=sizes, effects=effects, second_type_errors=beta)


@pytest.mark.unit
@pytest.mark.parametrize(
    "param_to_design, designer, expected_value",
    [
        ("size", pytest.lazy_fixture("designer_ltv_spark"), 1552),
        ("effect", pytest.lazy_fixture("designer_ltv_spark"), "17.6%"),
        ("power", pytest.lazy_fixture("designer_ltv_spark"), "35.6%"),
    ],
)
def test_run_theory_spark(param_to_design, expected_value, designer):
    """
    Some tests for method run and theory approach
    """
    if param_to_design != "power":
        assert designer.run(param_to_design).iloc[0][0] == expected_value
    else:
        assert designer.run(param_to_design).iloc[0].iloc[0] == expected_value


@pytest.mark.unit
@pytest.mark.parametrize(
    "param_to_design, designer",
    [
        ("size", pytest.lazy_fixture("designer_ltv_spark")),
        ("effect", pytest.lazy_fixture("designer_ltv_spark")),
        ("power", pytest.lazy_fixture("designer_ltv_spark")),
    ],
)
def test_empiric_spark(param_to_design, designer):
    """
    Check empiric design for spark works
    """
    result = designer.run(
        param_to_design, "empiric", second_type_errors=0.5, effects=1.5, sizes=150, as_numeric=True
    ).iloc[0, 0]
    assert result > 0


@pytest.mark.smoke
def test_not_available_dataframe():
    with pytest.raises(TypeError) as error:
        Designer(dataframe=2, metrics="abc", effects=1.2).run("size", "theory")
    assert str(error.value).startswith("Type of table must be one of")
