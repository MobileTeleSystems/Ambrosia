#  Copyright 2022 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Groups splitting methods.

Module contains `Splitter` core class and `split` method which are
intended to solve group splitting problems, primarily for A/B/.. tests.
Group splitting tasks usually include following parameters: number of groups,
group sizes, and splitting algorithm.

Currently, group splitting problems could be solved using data provided
in form of both pandas and Spark(with some restrictions) dataframes.

"""
from __future__ import annotations

from typing import Optional

import yaml

from ambrosia import types
from ambrosia.tools import log, type_checks
from ambrosia.tools.ab_abstract_component import ABMetaClass, ABToolAbstract

from .handlers import data_shape, handle_full_split, split_data_handler

SPLITTING_BOUND_CONST: float = 0.5


class Splitter(yaml.YAMLObject, ABToolAbstract, metaclass=ABMetaClass):
    """
    Unit for creating experimental groups from batch data.

    Split your data into groups of selected size with respect to:
        - Stratification columns
        - Metric distance of objects in feature space
        - Set of passed ids

    Parameters
    ----------
    dataframe : PassedDataType, optional
        Dataframe or string name of .csv table which
        contains data used for groups split.
    id_column : IdColumnNameType, optional
        Name of id column which is used in hash split.
    groups_size : int, optional
        Size of the splitted groups.
    test_group_ids : PeriodColumnNamesType, optional
        Ids of objects which are in B(test) group.
        Used in tasks of post experiment A(control) group pick up.
    fit_columns : PeriodColumnNamesType, optional
        List of columns names which values will be interpreted as
        coordinates of points in multidimensional space during metric split.
    strat_columns : PeriodColumnNamesType, optional
        Columns for stratification.
        https://en.wikipedia.org/wiki/Stratified_sampling

    Attributes
    ----------
    dataframe : PassedDataType
        Pandas or Spark dataframe with split data.
    id_column : IdColumnNameType
        Name of id column which is used in hash split.
    groups_size : int
        Split size of groups.
    test_group_ids : PeriodColumnNamesType
        Ids of objects which are in B(test) group.
    fit_columns : PeriodColumnNamesType
        List of columns names used for metric split.
    strat_columns : PeriodColumnNamesType
        Stratification columns names.

    Examples
    --------
    Our development team decided to add onboarding to the mobile app.
    Already knowing the required group size, we would like to select
    users for groups A and B respectively. Using the splitter class,
    this task could be done in the following way:

    >>> splitter = Splitter(dataframe=dataframe)
    >>> splitter.run(group_size=1000, method='hash', salt='onboarding')

    Suppose now, we know that people of different ages and from several
    countries use our application, so we would like to take this into
    account during split. To do this, you might use stratification,
    which can be easily applied by passing only one additional parameter:

    >>> splitter = Splitter(data=dataframe, strat_columns=['age', 'country'])
    >>> splitter.run(group_size=1000, method='hash', salt='onboarding')

    If we have fixed users for the testing group,
    this can be specified as a parameter:

    >>> splitter = Splitter(data=dataframe, strat_columns=['age', 'country'])
    >>> splitter.run(method='hash',
    >>>              salt='onboarding',
    >>>              test_group_ids=B_group_id
    >>> )

    Notes
    -----

    Main methods for split:

    Simple:
        - Randomly chosen groups (via ``np.random.choice``).

    Hash:
        - Using hashing of identifiers and distribution
          by buckets, selects the desired buckets for groups formation.

    Metric:
        - For a fixed reference group or a randomly selected one,
          other groups are selected using the nearest neighbor method
          (for desired list of columns passed in ``fit_columns`` parameter).

    Constructors:

        >>> # Empty constructor
        >>> splitter = Splitter()
        >>> # Some data
        >>> splitter = Splitter(dataframe=df,
        >>>                     id_column='my_id_column',
        >>>                     strat_columns=['gender', 'age'],
        >>>                     test_group_ids=ids_for_B_group
        >>> )

    Setters:

        >>> splitter.set_dataframe(dataframe)
        >>> # You can pass string for pd.read_csv
        >>> splitter.set_dataframe('name_of_table.csv')
        >>> # Other setters
        >>> splitter.set_group_size(1000)
        >>> splitter.set_strat_columns(['age', 'region'])

    Run:

        >>> splitter.run(method='hash', groups_size=10000)
        >>> splitter.run(method='metric'
        >>>              test_group_ids=b_group,
        >>>              id_column='id',
        >>>              strat_columns=['age', 'city']
        >>>              fit_columns=['metric_history_column', 'other_metric']
        >>>              method_meric='fast', # It is used as kwarg
        >>>              norm='l2' # It is used as kwarg
        >>> )

    Load from yaml config:

    >>> config = '''
                !splitter # <--- this is yaml tag (important!)
                    groups_size:
                        1000
                    id_column:
                        id
                    strat_columns:
                        - age
                        - country
            '''
    >>> splitter = yaml.load(config)
    >>> # Or use the implmented function
    >>> splitter = load_from_config(config)
    """

    yaml_tag = "!splitter"

    @type_checks.check_type_decorator(type_checks.check_type_dataframe)
    def set_dataframe(self, dataframe: Optional[types.PassedDataType]) -> None:
        self.__df = dataframe

    @type_checks.check_type_decorator(type_checks.check_type_id_column)
    def set_id_column(self, id_column: Optional[str]) -> None:
        self.__id_column = id_column

    @type_checks.check_type_decorator(type_checks.check_type_group_size)
    def set_group_size(self, groups_size: Optional[int]) -> None:
        self.__groups_size = groups_size

    @type_checks.check_type_decorator(type_checks.check_type_test_group_ids)
    def set_test_group_ids(self, test_group_ids: types.IndicesType) -> None:
        self.__test_group_ids = test_group_ids

    @type_checks.check_type_decorator(type_checks.check_type_fit_columns)
    def set_fit_columns(self, fit_columns: types.ColumnNamesType) -> None:
        self.__fit_columns = fit_columns

    @type_checks.check_type_decorator(type_checks.check_type_strat_columns)
    def set_strat_columns(self, strat_columns: types.ColumnNamesType) -> None:
        self.__strat_columns = strat_columns

    def __init__(
        self,
        dataframe: Optional[types.PassedDataType] = None,
        id_column: Optional[types.ColumnNameType] = None,
        groups_size: Optional[int] = None,
        test_group_ids: Optional[types.IndicesType] = None,
        fit_columns: Optional[types.ColumnNamesType] = None,
        strat_columns: Optional[types.ColumnNamesType] = None,
    ):
        """
        Splitter class constructor to initialize the object.
        """
        self.set_dataframe(dataframe)
        self.set_id_column(id_column)
        self.set_group_size(groups_size)
        self.set_test_group_ids(test_group_ids)
        self.set_fit_columns(fit_columns)
        self.set_strat_columns(strat_columns)

    def __getstate__(self):
        """
        Get the state of the object to serialize.
        """
        return dict(
            id_column=self.__id_column,
            groups_size=self.__groups_size,
            fit_columns=self.__fit_columns,
            strat_columns=self.__strat_columns,
        )

    @classmethod
    def from_yaml(cls, loader: yaml.Loader, node: yaml.Node):
        kwargs = loader.construct_mapping(node)
        return cls(**kwargs)

    def run(
        self,
        method: str,
        dataframe: Optional[types.PassedDataType] = None,
        id_column: Optional[types.ColumnNameType] = None,
        groups_size: Optional[int] = None,
        part_of_table: Optional[float] = None,
        groups_number: int = 2,
        test_group_ids: Optional[types.IndicesType] = None,
        strat_columns: Optional[types.ColumnNamesType] = None,
        salt: Optional[str] = None,
        fit_columns: Optional[types.ColumnNamesType] = None,
        **kwargs,
    ) -> types.SplitterResult:
        """
        Perform a split into groups with selected or saved parameters.

        Parameters
        ----------
        method : str
            Split method, for example ``"hash"``.
        dataframe : PassedDataType, optional
            Dataframe or string name of .csv table which
            contains data used for groups split.
        id_column : IdColumnNameType, optional
             Name of id column which is used in hash split.
        groups_size : int, optional
            Size of the splitted groups.
        part_of_table: float, optional
            Split factor(for group A) for tasks of dataframe full split.
            If is not ``None``, then overrides ``groups_size`` parameter
            during the split.
        groups_number : int, default: ``2``
            Number of groups to be splitted.
        test_group_ids : PeriodColumnNamesType, optional
            Ids of objects which are in B(test) group.
            Used in tasks of post experiment A(control) group pick up.
        strat_columns : PeriodColumnNamesType, optional
            Columns for stratification.
            https://en.wikipedia.org/wiki/Stratified_sampling
        salt : str, optional
            Salt for hashing in hash-split.
        fit_columns : PeriodColumnNamesType, optional
            List of columns names which values will be interpreted as
            coordinates of points in multidimensional space during metric split.
        **kwargs : Dict
            Other keyword arguments.

        Returns
        -------
        groups : pd.DataFrame
            Returns a dataframe with groups and label column.
            Dataframe will contain all columns of the original dataframe.

        Other Parameters
        ----------------
        threads : int, default : ``1``
            Number of threads used for calculations.
        """
        method: str = type_checks.check_split_method_value(method)
        dataframe: types.PassedDataType = type_checks.check_type_dataframe(dataframe)
        id_column: types.ColumnNameType = type_checks.check_type_id_column(id_column)
        groups_size: int = type_checks.check_type_group_size(groups_size)
        test_group_ids: types.IndicesType = type_checks.check_type_test_group_ids(test_group_ids)
        fit_columns: types.ColumnNamesType = type_checks.check_type_fit_columns(fit_columns)
        strat_columns: types.ColumnNamesType = type_checks.check_type_strat_columns(strat_columns)

        arguments_choice: types._PrepareArgumentsType = {
            "dataframe": (self.__df, dataframe),
        }

        strat_columns: str = strat_columns if strat_columns is not None else self.__strat_columns
        test_group_ids = test_group_ids if test_group_ids is not None else self.__test_group_ids
        id_column = id_column if id_column is not None else self.__id_column

        if test_group_ids is not None:
            arguments_choice["group_b_indices"] = (None, test_group_ids)
        else:
            arguments_choice["groups_size"] = (self.__groups_size, groups_size)
        if part_of_table is not None:
            # Group size will be set later
            arguments_choice["groups_size"] = (self.__groups_size, 0)
            if groups_size is not None:
                log.info_log("Groups size variable ignored because part splitting variable set")
            if groups_number > 2:
                groups_number = 2
                log.info_log("Groups number was set to 2 because part splitting variable set")

        if method in ("metric", "dim_decrease"):
            # For methods use metric/cluster/unsupervised approach
            arguments_choice["fit_columns"] = (self.__fit_columns, fit_columns)

        chosen_args: types._UsageArgumentsType = Splitter._prepare_arguments(arguments_choice)

        if part_of_table is not None:
            split_part: float = part_of_table if (part_of_table <= SPLITTING_BOUND_CONST) else 1 - part_of_table
            chosen_args["groups_size"] = round(split_part * data_shape(chosen_args["dataframe"]))

        chosen_args["split_method"] = method
        chosen_args["id_column"] = id_column
        chosen_args["strat_columns"] = strat_columns
        chosen_args["salt"] = salt
        chosen_args["groups_number"] = groups_number
        groups: types.SplitterResult = split_data_handler(**chosen_args, **kwargs)

        if part_of_table is not None:
            return handle_full_split(chosen_args["dataframe"], groups, part_of_table, id_column)

        return groups


def load_from_config(yaml_config: str, loader: type = yaml.Loader) -> Splitter:
    """
    Restore a ``Splitter`` class instance from a yaml config.

    For yaml_config parameter you can pass file name with
    config, which must ends with .yaml, for example: "config.yaml".
    For loader you can choose SafeLoader.
    """
    if isinstance(yaml_config, str) and yaml_config.endswith(".yaml"):
        with open(yaml_config, "r", encoding="utf-8") as file:
            return yaml.load(file, Loader=loader)
    return yaml.load(yaml_config, Loader=loader)


def split(
    method: str,
    dataframe: Optional[types.PassedDataType] = None,
    id_column: Optional[types.ColumnNameType] = None,
    groups_size: Optional[int] = None,
    part_of_table: Optional[float] = None,
    groups_number: int = 2,
    test_group_ids: Optional[types.IndicesType] = None,
    strat_columns: Optional[types.ColumnNamesType] = None,
    salt: Optional[str] = None,
    fit_columns: Optional[types.ColumnNamesType] = None,
    threads: int = 1,
    **kwargs,
) -> types.SplitterResult:
    """
    Function wrapper around the ``Splitter`` class.

    Used to create splitted groups from the dataframe.

    Creates an instance of the ``Splitter`` class internally and execute
    run method with corresponding arguments.

    Parameters
    ----------
    method : str
        Split method, for example ``"hash"``.
    dataframe : PassedDataType, optional
        Dataframe or string name of .csv table which
        contains data used for groups split.
    id_column : IdColumnNameType, optional
            Name of id column which is used in hash split.
    groups_size : int, optional
        Size of the splitted groups.
    part_of_table: float, optional
        Split factor(for group A) for tasks of dataframe full split.
        If is not ``None``, then overrides ``groups_size`` parameter
        during the split.
    groups_number : int, default : ``2``
        Number of groups to be splitted.
    test_group_ids : PeriodColumnNamesType, optional
        Ids of objects which are in B(test) group.
        Used in tasks of post experiment A(control) group pick up.
    strat_columns : PeriodColumnNamesType, optional
        Columns for stratification.
        https://en.wikipedia.org/wiki/Stratified_sampling
    salt : str, optional
        Salt for hashing in hash-split.
    fit_columns : PeriodColumnNamesType, optional
        List of columns names which values will be interpreted as
        coordinates of points in multidimensional space during metric split.
    threads : int, default : ``1``
        Number of threads used for calculations.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    groups : pd.DataFrame
        Returns a dataframe with groups and label column.
        Dataframe will contain all columns of the original dataframe.
    """
    return Splitter(
        dataframe=dataframe,
        id_column=id_column,
        groups_size=groups_size,
        fit_columns=fit_columns,
        test_group_ids=test_group_ids,
        strat_columns=strat_columns,
    ).run(method, salt=salt, threads=threads, part_of_table=part_of_table, groups_number=groups_number, **kwargs)
