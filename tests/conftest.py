from typing import Dict, List

import numpy as np
import pandas as pd
import pyspark
import pytest
import scipy.stats as sps

from ambrosia.designer import Designer
from ambrosia.splitter import Splitter
from ambrosia.tester import Tester
from ambrosia.tools.stratification import Stratification


@pytest.fixture()
def local_spark_session() -> None:
    """
    Local Spark session
    """
    spark = pyspark.sql.SparkSession.builder.master("local[1]").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture()
@pytest.mark.designer()
def simple_binary_retention_table() -> pd.DataFrame:
    """
    Simple data frame, contains
    """
    df = pd.DataFrame([[1, 2], [0, 3], [0, 1], [1, 22], [0, 9]], columns=["retention", "some metric"])
    return df


@pytest.fixture()
@pytest.mark.designer()
def ltv_and_retention_dataset() -> pd.DataFrame:
    """
    Data Frame
    Retention is bernoulli  with p = 0.4
    LTV (Life Time Value) is Exponential(scale = 100)
    """
    df = pd.read_csv("./tests/test_data/ltv_retention.csv")
    return df


@pytest.fixture()
@pytest.mark.designer()
def designer_simple_table(simple_binary_retention_table):
    """
    Designer instance of simple_binary_retention_table
    """
    designer = Designer(
        dataframe=simple_binary_retention_table, metrics="retention", sizes=[100, 20], effects=1.2, method="theory"
    )
    return designer


@pytest.fixture()
@pytest.mark.designer()
def designer_ltv(ltv_and_retention_dataset):
    """
    Designer instance of ltv_and_retention_dataset
    """
    designer = Designer(
        dataframe=ltv_and_retention_dataset, metrics="LTV", sizes=[500, 1000], effects=1.1, method="theory"
    )
    return designer


@pytest.fixture()
@pytest.mark.designer()
def designer_ltv_spark(local_spark_session, ltv_and_retention_dataset):
    """
    Designer based on spark dataframe
    """
    table = local_spark_session.createDataFrame(ltv_and_retention_dataset)
    designer = Designer(dataframe=table, metrics="LTV", sizes=[500, 1000], effects=1.1, method="theory")
    return designer


@pytest.fixture()
@pytest.mark.tester()
def results_ltv_retention_conversions() -> pd.DataFrame:
    """
    Table with results of experiment
    """
    df_result = pd.read_csv("./tests/test_data/result_ltv_ret_conv.csv")
    return df_result


@pytest.fixture()
@pytest.mark.tester()
def tester_spark_ltv_ret(local_spark_session, results_ltv_retention_conversions):
    """
    Spark tester
    """
    table = local_spark_session.createDataFrame(results_ltv_retention_conversions)
    tester = Tester(dataframe=table, metrics=["retention", "conversions", "ltv"], column_groups="group")
    return tester


@pytest.fixture()
@pytest.mark.tester()
def tester_on_ltv_retention(results_ltv_retention_conversions):
    """
    Tester based on results_ltv_retention_conversions
    """
    tester = Tester(
        dataframe=results_ltv_retention_conversions,
        metrics=["retention", "conversions", "ltv"],
        column_groups="group",
    )
    return tester


@pytest.fixture()
@pytest.mark.stratification()
def stratification_table() -> pd.DataFrame:
    """
    Table for stratification
    """
    df = pd.read_csv("./tests/test_data/stratification_data.csv")
    return df


@pytest.fixture()
@pytest.mark.stratification()
def stratificator(stratification_table):
    """
    Stratification instance on stratification_table
    """
    strat = Stratification()
    strat.fit(stratification_table, columns=["gender", "retention"])
    return strat


@pytest.fixture()
@pytest.mark.stratification()
def answer_ids_strat(stratificator) -> Dict:
    """
    Answer for test checking B group ids
    """
    answer = {
        ("Female", 0): ([129, 645, 860], 109),
        ("Female", 1): ([43, 258, 473, 688, 946, 989], 167),
        ("Male", 0): ([344, 602, 817, 903], 300),
        ("Male", 1): ([0, 86, 172, 215, 301, 387, 430, 516, 559, 731, 774], 400),
    }
    return answer


@pytest.fixture()
@pytest.mark.stratification()
def id_for_b_strat(stratification_table) -> np.ndarray:
    """
    Group B ids for stratification test
    """
    return stratification_table.loc[np.arange(0, 1000, 43)]["id"]


@pytest.fixture()
@pytest.mark.stratification()
def answer_ids_strat_column(stratificator) -> Dict:
    """
    Answer for test checking B group ids using id column
    """
    answer = {
        ("Female", 0): ([904, 4516, 6021], 109),
        ("Female", 1): ([302, 1807, 3312, 4817, 6623, 6924], 167),
        ("Male", 0): ([2409, 4215, 5720, 6322], 300),
        ("Male", 1): ([1, 603, 1205, 1506, 2108, 2710, 3011, 3613, 3914, 5118, 5419], 400),
    }
    return answer


@pytest.fixture()
@pytest.mark.splitter()
def data_split() -> pd.DataFrame:
    """
    Table for splitter tests
    """
    table: pd.DataFrame = pd.read_csv("tests/test_data/splitter_dataframe.csv")
    return table


@pytest.fixture()
@pytest.mark.splitter()
def data_index_split() -> pd.DataFrame:
    """
    Table with strange index
    """
    size: int = 1000
    ind: List[str] = [str(x) + "a" for x in np.arange(size)]
    metric_x: np.ndarray = sps.norm.rvs(loc=0, scale=2, size=size)
    metric_y: np.ndarray = sps.norm.rvs(loc=3, scale=5, size=size)
    return pd.DataFrame(np.array([metric_x, metric_y]).T, columns=["x", "y"], index=ind)


@pytest.fixture()
@pytest.mark.splitter()
def splitter_ltv_spark(local_spark_session, ltv_and_retention_dataset):
    """
    Splitter based on spark dataframe
    """
    table = local_spark_session.createDataFrame(ltv_and_retention_dataset.reset_index())
    splitter = Splitter(dataframe=table, groups_size=50, id_column="index")
    return splitter


@pytest.fixture()
@pytest.mark.variance()
def data_variance_lin() -> pd.DataFrame:
    """
    Table with Y = 2 * X_1 + 3 * X_2 + 4 * X_3 + N(0, 0.1)
    """
    table: pd.DataFrame = pd.read_csv("tests/test_data/var_table.csv")
    return table


@pytest.fixture()
@pytest.mark.variance()
def data_nonlin_var() -> pd.DataFrame:
    """
    Table with Y = X_1 ** 2 + 3 * sqrt(X_2) + 4 * log(X_3) ** 5 + N(0, 0.1)
    """
    table: pd.DataFrame = pd.read_csv("tests/test_data/nonlin_var_table.csv")
    return table


@pytest.fixture()
@pytest.mark.aggregate()
def data_for_agg() -> pd.DataFrame:
    """
    Table for aggregation
    columns: id | gender | watched | sessions | day | platform
    """
    table: pd.DataFrame = pd.read_csv("tests/test_data/week_metrics.csv")
    return table


@pytest.fixture()
@pytest.mark.designer()
def robust_moments() -> pd.DataFrame:
    """
    Data frame for testing work of robust preprocessing tools.
    These table is based on nonlin_var_table.csv.
    Columns and rows are transformed using various preprocessing techniques
    manually.
    """
    df = pd.read_csv("tests/test_data/robust_moments.csv", index_col=0)
    return df
