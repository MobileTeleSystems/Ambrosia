import os
from typing import List

import numpy as np
import pandas as pd
import pytest
import yaml

from ambrosia.splitter import Splitter, load_from_config, split

store_path: str = "tests/configs/dumped_splitter.yaml"


@pytest.mark.smoke()
def test_instance():
    """
    Check that simple instance without args work
    """
    splitter = Splitter()


@pytest.mark.smoke()
def test_constructors(results_ltv_retention_conversions):
    """
    Constructors tests
    """
    splitter = Splitter(id_column="id")
    # Fake df
    splitter = Splitter(dataframe=results_ltv_retention_conversions)
    splitter = Splitter(fit_columns="some_fit")
    splitter = Splitter(fit_columns=["some metric"])


@pytest.mark.smoke()
def test_setter_method():
    """
    Setter methods testing
    """
    splitter = Splitter()
    splitter.set_id_column("id")
    splitter.set_group_size(1000)
    try:
        splitter.set_group_size("some size")
    except Exception as err:
        assert isinstance(err, TypeError)


@pytest.mark.unit()
@pytest.mark.parametrize("strat_columns", ["l", "e", ["l", "e"]])
@pytest.mark.parametrize("fit_columns", ["a", "b", ["m", "a", "b"]])
@pytest.mark.parametrize("groups_size", [100, 300, 500])
def test_all_inputs_metric(strat_columns, fit_columns, groups_size, data_split):
    """
    Test metric method
    table size
    """
    strat_amount: int = 1 if isinstance(strat_columns, str) else len(strat_columns)
    splitter = Splitter(dataframe=data_split, strat_columns=strat_columns, fit_columns=fit_columns)
    table: pd.DataFrame = splitter.run(method="metric", groups_size=groups_size)
    for label in ["A", "B"]:
        assert table[table.group == label].shape[0] == groups_size


@pytest.mark.unit()
@pytest.mark.parametrize("strat_columns", ["l", ["l", "e"]])
@pytest.mark.parametrize("groups_size", [10, 200, 2000])
@pytest.mark.parametrize("salt", ["salt", "salt_other", "abcde"])
@pytest.mark.parametrize("id_column", [None, "index"])
@pytest.mark.parametrize("groups_number", [2, 3, 4])
@pytest.mark.parametrize("hash_function", ["sha256", "sha512", "blake2"])
def test_split_hash_stable(strat_columns, groups_size, salt, id_column, groups_number, hash_function, data_split):
    """
    Test that hash split work deterministic with fixed hash
    """
    splitter = Splitter(dataframe=data_split, strat_columns=strat_columns, groups_size=groups_size, id_column=id_column)
    first_result: pd.DataFrame = splitter.run(
        method="hash", salt=salt, groups_number=groups_number, hash_function=hash_function
    )
    second_result: pd.DataFrame = splitter.run(
        method="hash", salt=salt, groups_number=groups_number, hash_function=hash_function
    )
    third_result: pd.DataFrame = splitter.run(
        method="hash", salt=salt, groups_number=groups_number, hash_function=hash_function
    )
    assert first_result.equals(second_result)
    assert second_result.equals(third_result)


@pytest.mark.unit()
@pytest.mark.parametrize("groups_size", [30, 500, 1000])
@pytest.mark.parametrize("groups_number", [2, 3, 4, 5])
@pytest.mark.parametrize("method", ["simple", "hash"])
@pytest.mark.parametrize("strat_columns", [None, "l", "e", ["l", "e"]])
def test_many_groups_split(groups_size, groups_number, method, strat_columns, data_split):
    """
    Test, that for many groups indices are not intersect
    """
    splitter = Splitter(
        dataframe=data_split,
        strat_columns=strat_columns,
        groups_size=groups_size,
    )
    result: pd.DataFrame = splitter.run(method=method, groups_number=groups_number)
    labels: list[str] = ["A", "B", "C", "D", "E"]
    for i in range(groups_number):
        label: str = labels[i]
        assert len(result[result.group == label]) == groups_size


@pytest.mark.unit()
@pytest.mark.parametrize("groups_size", [50, 300])
@pytest.mark.parametrize("groups_number", [2, 3])
def test_index_metric(groups_size, groups_number, data_index_split):
    """
    Test metric split with strange index
    """
    splitter = Splitter(dataframe=data_index_split, strat_columns=None, groups_size=groups_size, fit_columns=["x", "y"])
    result: pd.DataFrame = splitter.run(method="metric", groups_number=groups_number)
    labels: list[str] = ["A", "B", "C", "D", "E"]
    for i in range(groups_number):
        label: str = labels[i]
        assert len(result[result.group == label]) == groups_size


@pytest.mark.unit()
@pytest.mark.parametrize("groups_size", [50, 100, 1000])
@pytest.mark.parametrize("method", ["simple", "hash", "metric"])
@pytest.mark.parametrize("id_column", [None, "index"])
@pytest.mark.parametrize("strat_columns", [None, "l", "e", ["l", "e"]])
def test_fixed_b_group(groups_size, method, id_column, strat_columns, data_split):
    """
    Test fixed group
    """
    fit_columns: List[str] = ["a", "b"]
    splitter = Splitter(dataframe=data_split, fit_columns=fit_columns, id_column=id_column)
    b_ind: np.ndarray = np.random.choice(data_split.index.values, size=groups_size, replace=False)
    result: pd.DataFrame = splitter.run(method=method, test_group_ids=b_ind, strat_columns=strat_columns)
    assert len(result[result.group == "A"]) == groups_size
    assert len(result[result.group == "B"]) == groups_size

    if id_column is not None:
        assert np.sum([0 if x not in result[result.group == "B"][id_column] else 1 for x in b_ind]) == groups_size
    else:
        assert np.sum([0 if x not in result[result.group == "B"].index.values else 1 for x in b_ind]) == groups_size


@pytest.mark.unit()
@pytest.mark.parametrize("strat_columns", ["l", ["l", "e"]])
@pytest.mark.parametrize("groups_size", [10, 2000])
@pytest.mark.parametrize("fit_columns", ["a", ["a", "b"]])
@pytest.mark.parametrize("salt", ["salt"])
@pytest.mark.parametrize("id_column", [None, "index"])
@pytest.mark.parametrize("groups_number", [2, 4])
@pytest.mark.parametrize("method", ["simple", "hash", "metric"])
def test_split_function(strat_columns, groups_size, salt, id_column, fit_columns, groups_number, method, data_split):
    """
    Test standalone split function
    """
    result: pd.DataFrame = split(
        dataframe=data_split,
        strat_columns=strat_columns,
        groups_size=groups_size,
        salt=salt,
        fit_columns=fit_columns,
        groups_number=groups_number,
        method=method,
    )
    labels: list[str] = ["A", "B", "C", "D", "E"]
    for i in range(groups_number):
        label: str = labels[i]
        assert len(result[result.group == label]) == groups_size


@pytest.mark.unit()
@pytest.mark.parametrize("factor", [0.1, 0.5, 0.2111, 0.7983])
@pytest.mark.parametrize("method", ["simple", "hash", "metric"])
@pytest.mark.parametrize("strat_columns", [None, "retention"])
def test_full_split(ltv_and_retention_dataset, factor, method, strat_columns):
    """
    Test full split table with split factors
    """
    total_size: int = ltv_and_retention_dataset.shape[0]
    splitter = Splitter(ltv_and_retention_dataset, fit_columns="LTV")
    result = splitter.run(method, part_of_table=factor, strat_columns=strat_columns)
    size_a: int = result[result.group == "A"].shape[0]
    size_b: int = result[result.group == "B"].shape[0]
    assert size_a == round(total_size * factor)
    assert size_b == round(total_size * (1 - factor))


@pytest.mark.unit()
@pytest.mark.parametrize("method", ["hash"])
@pytest.mark.parametrize("groups_number", [2, 3])
@pytest.mark.parametrize("strat_columns", [None, "retention"])
def test_spark_split(method, groups_number, strat_columns, splitter_ltv_spark):
    """
    Test spark split
    """
    group_size: int = 50
    result = splitter_ltv_spark.run(method, groups_number=groups_number, strat_columns=strat_columns)
    total_size = result.count()
    assert total_size == group_size * groups_number


@pytest.mark.unit()
@pytest.mark.parametrize("factor", [0.1, 0.9])
@pytest.mark.parametrize("method", ["hash"])
@pytest.mark.parametrize("strat_columns", [None, "retention"])
def test_full_split_spark(ltv_and_retention_dataset, splitter_ltv_spark, factor, method, strat_columns):
    """
    Test full split table with split factors for spark tables
    """
    total_size: int = ltv_and_retention_dataset.shape[0]
    result = splitter_ltv_spark.run(method, part_of_table=factor, strat_columns=strat_columns)
    size_a: int = result.where("group == 'A'").count()
    size_b: int = result.where("group == 'B'").count()
    assert size_a == round(total_size * factor)
    assert size_b == round(total_size * (1 - factor))

@pytest.mark.unit
def test_splitter_load_from_config(ltv_and_retention_dataset):
    """
    Test Splitter class dump and load from yaml abilities.
    """
    method: str = "hash"
    salt: str = "test"
    splitter = Splitter(
        dataframe=ltv_and_retention_dataset, groups_size=1000, strat_columns="retention", fit_columns="LTV"
    )
    split_res = splitter.run(method=method, salt=salt, groups_number=3)
    with open(store_path, "w") as outfile:
        yaml.dump(splitter, outfile, default_flow_style=False)

    loaded_splitter = load_from_config(store_path)
    loaded_splitter.set_dataframe(ltv_and_retention_dataset)
    split_res_from_config = loaded_splitter.run(method=method, salt=salt, groups_number=3)
    os.remove(store_path)
    assert split_res.equals(split_res_from_config)

@pytest.mark.unit()
@pytest.mark.parametrize("id_column", [None, "id"])
def test_duplication_exception(id_column):
    """
    Test for an exception for duplicate values in an id column of pandas data frame.
    """
    duplicated_data: pd.DataFrame = pd.concat(
        [pd.DataFrame({id_column: range(500)}), pd.DataFrame({id_column: range(500)})]
    )
    splitter = Splitter(duplicated_data, id_column=id_column)
    with pytest.raises(Exception) as exc_info:
        splitter.run(method="hash", groups_size=100)
    if id_column is None:
        msg_part: str = "Index"
    else:
        msg_part: str = f"Id column {id_column}"
    error_msg: str = f"{msg_part} contains duplicates, ids must be unique for split"
    assert str(exc_info.value) == error_msg

@pytest.mark.unit()
def test_duplication_exception_spark(local_spark_session, ltv_and_retention_dataset):
    """
    Test for an exception for duplicate values in an id column of spark data frame.
    """
    table_one = local_spark_session.createDataFrame(ltv_and_retention_dataset.reset_index())
    table_two = local_spark_session.createDataFrame(ltv_and_retention_dataset.reset_index())
    duplicated_data = table_one.union(table_two)
    id_column: str = "index"
    splitter = Splitter(duplicated_data, id_column=id_column)
    with pytest.raises(Exception) as exc_info:
        splitter.run(method="hash", groups_size=1000)
    assert str(exc_info.value) == f"Id column {id_column} contains duplicates, ids must be unique for split"
