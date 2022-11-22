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
Experiment results evaluation and testing methods.

Module contains `Tester` core class and `test` methid which are
designed to evaluate statistical significance of the experiment results
and a magnitude of effect via large number of methods and criteria.

It is recommended to use for parameters such as test method and statistical
criterion the values that were chosen during the experiment design stage.

Currently, experimental results can only be processed and evaluated as
pandas DataFrames or .csv tables. Support for Spark dataframes is under
development and will be available soon.

"""
import itertools
from copy import deepcopy
from typing import Dict, List, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd

import ambrosia.tools.empirical_tools as empirical_pkg
import ambrosia.tools.pvalue_tools as pvalue_pkg
import ambrosia.tools.stat_criteria as criteria_pkg
from ambrosia import types
from ambrosia.tools.ab_abstract_component import ABStatCriterion, ABToolAbstract, DataframeHandler, StatCriterion

from .binary_result_evaluation import binary_absolute_result, binary_relative_result

BOOTSTRAP_SIZE: int = 1000
AVAILABLE: List[str] = ["pandas", "spark"]
AVAILABLE_AB_CRITERIA: Dict[str, ABStatCriterion] = {
    "ttest": criteria_pkg.TtestIndCriterion,
    "ttest_rel": criteria_pkg.TtestRelCriterion,
    "mw": criteria_pkg.MannWhitneyCriterion,
    "wilcoxon": criteria_pkg.WilcoxonCriterion,
}
AVAILABLE_MULTITEST_CORRECTIONS: List[str] = ["bonferroni"]


class Tester(ABToolAbstract):
    """
    Unit for experimental data test and evaluation.

    The experiment evaluation result contains:
        - Pvalue for the selected criterion
        - Point effect estimation
        - Corresponding confidence interval for the effect
        - Boolean result - presence / absence of the effect

    Parameters
    ----------
    dataframe : PassedDataType, optional
        Dataframe used with experiment results metrics.
    df_mapping : GroupsInfoType, optional
        Dataframe which contains group labels of objects.
    experiment_results : ExperimentResults, optional
        Dict with separate experiment results for each group.
        Dict keys are used as groups labels, values must be either
        pandas or Spark dataframes.
    column_groups : ColumnNameType, optional
        Column which contains groups label of objects.
    group_labels : GroupLabelsType, optional
        Labels for experimental groups. If ``column_groups`` contains
        at least two values, they will choose for labels.
    id_column : ColumnNameType, optional
        Name of column with objects ids in ``df_mapping`` dataframe.
    first_errors : StatErrorType, default: ``0.05``
        I type errors values. Fix P (detect difference for equal) to be less
        than threshold. Used to construct confidence intervals.
    metrics : MetricNameType, optional
        Metrics (columns of dataframe) which is used to calculate
        experiment result.

    Attributes
    ----------
    dataframe : PassedDataType
        Dataframe used with experiment results metrics.
    df_mapping : GroupsInfoType
        Dataframe which contains group labels of objects.
    experiment_results : ExperimentResults, optional
        Dict with separate experiment results for each group.
    column_groups : ColumnNameType
        Column which contains groups label of objects.
    group_labels : GroupLabelsType
        Labels for experimental groups.
    id_column : ColumnNameType
        Name of column with objects ids in ``df_mapping`` dataframe.
    first_errors : StatErrorType, default: ``0.05``
        I type errors values.
    metrics : MetricNameType
        Columns of dataframe with experiment results.

    Examples
    --------
    We've experimented with adding onboarding to our mobile app and
    would like to know about its results in terms of A/B testing.
    Suppose we have a loaded pandas dataframe with a column responsible
    for the groups in the testing and columns with metric values,
    such as retention. Then you can use the tester class the following way:

    >>> tester = Tester(
    >>>     dataframe=df,
    >>>     column_groups='groups',
    >>>     metrics='retention'
    >>> )
    >>> tester.run()
    >>> # Output
    >>> [{
    >>>     'first_type_error' : 0.05,
    >>>     'pvalue' : 0.03,
    >>>     'effect' : 1.05,
    >>>     'confidence_interval' : (1.01, 1.10),
    >>>     'metric name': 'retention',
    >>>     'group A label': 'A',
    >>>     'group B label': 'B'
    >>> }]

    Notes
    -----
    Basic mathematic methods for evaluating experiments:

        - Theory:
            - Absolute: Using ttest, mann-whitney, others and custom criteria
            - Relative: Using delta method

        - Empiric:
            - Absolute / Relative: Building empirical distribution for T(A, B)

        - Binary:
            - Absolute: Using special binary intervals and
              finding pvalue = inf_a {x : 0 not in interval(x)}
            - Relative: Not implemented yet :(

    Constructors:

    >>> # Empty constructor
    >>> tester = Tester()
    >>> # You can pass Iterable or single object for some parameters
    >>> tester = Tester(
    >>>     dataframe=df,
    >>>     columns_groups='groups',
    >>>     metrics=['ltv', 'retention']
    >>> )
    >>> tester = Tester(metrics='retention', first_errors=[0.01, 0.05])
    >>> # You can set a separate table containing information about
    >>> # the partitioning in the experiment
    >>> tester = tester = Tester(
    >>>     dataframe=df, # main dataframe with metrics
    >>>     df_mapping=groups, # table with information about groups
    >>>     metrics='metric', # Metric to be tested
    >>>     column_groups='group', # Column in df_mapping with labels
    >>>     id_column='id' # Column with ids in df and df_mapping (for join)
    >>> )

    Setters:

    >>> tester.set_metrics(['ltv', 'retention'])
    >>> tester.set_dataframe(dataframe=dataframe, column_groups='groups')
    >>> # You can set separate data of each group packed in special dict form
    >>> tester.set_experiment_results(experiment_results=experiment_results)

    Run:

    >>> # You can choose effect_type to estimate: relative / absolute
    >>> tester.run('absolute')
    >>> # Also you can choose method
    >>> tester.run('absolute', method='empriric') # emipiric for bootstrap
    >>> # One can pass arguments in run() method and they will have
    >>> # higher priority
    >>> tester.run(metrics='ltv', data_a_group=df_a)

    Use a function instead of a class:

    >>> test('absolute', dataframe=df, column_groups='groups', metrics='ltv')
    """

    # This is for avoiding warnings from pytest
    __test__ = False

    def set_experiment_results(self, experiment_results: types.ExperimentResults) -> None:
        self.__experiment_results = experiment_results

    def set_errors(self, first_errors: types.StatErrorType) -> None:
        if isinstance(first_errors, float):
            self.__alpha = np.array([first_errors])
        else:
            self.__alpha = np.array(first_errors)

    def set_metrics(self, metrics: types.MetricNamesType) -> None:
        if isinstance(metrics, types.MetricNameType):
            self.__metrics = [metrics]
        else:
            self.__metrics = metrics

    def set_dataframe(
        self,
        dataframe: types.PassedDataType,
        column_groups: types.MetricNameType,
        group_labels: types.GroupLabelsType = None,
        df_mapping: types.GroupsInfoType = None,
        id_column: types.MetricNameType = None,
    ) -> None:
        __filtering_kwargs = {
            "dataframe": dataframe,
            "df_mapping": df_mapping,
            "column_groups": column_groups,
            "group_labels": group_labels,
            "id_column": id_column,
        }
        self.__experiment_results = DataframeHandler()._handle_cases(
            Tester.__filter_data, Tester.__filter_spark_data, **__filtering_kwargs
        )

    def __init__(
        self,
        dataframe: Optional[types.PassedDataType] = None,
        df_mapping: Optional[types.GroupsInfoType] = None,
        experiment_results: Optional[types.ExperimentResults] = None,
        column_groups: Optional[types.ColumnNameType] = None,
        group_labels: Optional[types.GroupLabelsType] = None,
        id_column: Optional[types.ColumnNameType] = None,
        first_errors: types.StatErrorType = 0.05,
        metrics: Optional[types.MetricNamesType] = None,
    ):
        """
        Tester class constructor to initialize the object.
        """
        if dataframe is not None:
            self.set_dataframe(
                dataframe,
                column_groups,
                group_labels,
                df_mapping,
                id_column,
            )
        else:
            self.set_experiment_results(experiment_results=experiment_results)
        self.set_errors(first_errors)
        self.set_metrics(metrics)

    @staticmethod
    def __filter_data(
        dataframe: types.PassedDataType,
        df_mapping: types.GroupsInfoType,
        column_groups: types.ColumnNameType,
        group_labels: types.GroupLabelsType,
        id_column: types.ColumnNameType,
    ) -> types.TwoSamplesType:
        """
        Function to handle setting of pandas data.
        """
        if dataframe is None:
            return None

        if df_mapping is not None:
            if id_column not in dataframe:
                raise ValueError(f"Column {id_column}, is not in list of df columns")
            if id_column not in df_mapping:
                raise ValueError(f"Column {id_column}, is not in list of df_mapping columns")
            dataframe = dataframe.merge(df_mapping, how="left", on=id_column).dropna()
        if column_groups not in dataframe:
            raise ValueError(f"Column {column_groups}, is not in list of df columns")

        if group_labels is not None:
            if len(group_labels) < 2:
                raise ValueError(f"Group labels must be at least 2, given {group_labels}")
        else:
            group_labels = dataframe[column_groups].unique()
        experiment_results: types.ExperimentResults = {
            group_label: dataframe[dataframe[column_groups] == group_label] for group_label in group_labels
        }
        return experiment_results

    def __filter_spark_data(self):
        """
        Function to handle setting of Spark data.
        """

    @staticmethod
    def __bootstrap_result(
        group_a: types.GroupType,
        group_b: types.GroupType,
        alpha: np.ndarray,
        bootstrap_size: int = BOOTSTRAP_SIZE,
        effect_type: str = "absolute",
        **kwargs,
    ) -> types._SubResultType:
        """
        Function to handle the empirical approach to testing.
        """
        if effect_type == "absolute":
            metric = "mean"
            point_effect = np.mean(group_b) - np.mean(group_a)
        elif effect_type == "relative":
            metric = "fraction"
            point_effect = np.mean(group_b) / np.mean(group_a) - 1
        else:
            raise ValueError("Set effect_type as 'absolute' or 'relative'")
        bootstrap_handler = empirical_pkg.BootstrapStats(bootstrap_size=bootstrap_size, metric=metric)
        bootstrap_handler.fit(group_a, group_b)
        left_bounds, right_bounds = bootstrap_handler.confidence_interval(confidence_level=1 - alpha, **kwargs)
        pvalue = bootstrap_handler.pvalue_criterion(**kwargs)
        confidence_interval = list(zip(left_bounds, right_bounds))
        return {
            "first_type_error": alpha,
            "pvalue": pvalue,
            "effect": point_effect,
            "confidence_interval": confidence_interval,
        }

    @staticmethod
    def __binary_result(
        group_a: types.GroupType, group_b: types.GroupType, alpha: np.ndarray, effect_type: str = "absolute", **kwargs
    ) -> types._SubResultType:
        """
        Function to handle binary intervals for testing.
        """
        warning_message_values: str = "Values for metric is not binary, choose other method, for example ttest!"
        if not set(np.unique(group_a)).issubset({0, 1}) or not set(np.unique(group_b)).issubset({0, 1}):
            warn(warning_message_values)
        if effect_type == "absolute":
            return binary_absolute_result(group_a, group_b, alpha, **kwargs)
        elif effect_type == "relative":
            return binary_relative_result(group_a, group_b, alpha, **kwargs)
        else:
            raise ValueError(f"``effect_type`` variable could be only  'absolute' or 'relative, got {effect_type}.")

    @staticmethod
    def __theory_handler(
        group_a: types.GroupType,
        group_b: types.GroupType,
        alpha: np.ndarray,
        effect_type: str = "absolute",
        criterion: Optional[ABStatCriterion] = None,
        **kwargs,
    ) -> types._SubResultType:
        """
        Function to handle the theoretical approach to testing.
        """
        criterion: Union[str, StatCriterion] = criterion if criterion is not None else "ttest"
        if isinstance(criterion, str) & (criterion in AVAILABLE_AB_CRITERIA):
            criterion = AVAILABLE_AB_CRITERIA[criterion]
        elif not (hasattr(criterion, "get_results") and callable(criterion.get_results)):
            raise ValueError(
                f"Choose correct criterion name from {list(AVAILABLE_AB_CRITERIA)} or pass correct custom class"
            )
        return criterion().get_results(group_a=group_a, group_b=group_b, alpha=alpha, effect_type=effect_type, **kwargs)

    @staticmethod
    def __pre_run(method: str, args: types._UsageArgumentsType, **kwargs) -> types.TesterResult:
        """
        Function to handle run method on pandas dataframes.
        """
        accepted_methods: List[str] = ["theory", "empiric", "binary"]
        if method not in accepted_methods:
            raise ValueError(f'Choose method from {", ".join(accepted_methods)}')
        result: types.TesterResult = {}
        for metric in args["metrics"]:
            a_values: np.ndarray = args["data_a_group"][metric].values
            b_values: np.ndarray = args["data_b_group"][metric].values
            if method == "theory":
                sub_result = Tester.__theory_handler(
                    a_values,
                    b_values,
                    np.array(args["alpha"]),
                    effect_type=args["effect_type"],
                    criterion=args["criterion"],
                    **kwargs,
                )
            elif method == "empiric":
                sub_result = Tester.__bootstrap_result(
                    a_values, b_values, np.array(args["alpha"]), effect_type=args["effect_type"], **kwargs
                )
            elif method == "binary":
                sub_result = Tester.__binary_result(
                    a_values, b_values, np.array(args["alpha"]), effect_type=args["effect_type"], **kwargs
                )
            result[metric] = sub_result
        return result

    @staticmethod
    def __pre_run_spark():
        """
        Function to handle run method on Spark dataframes.
        """

    @staticmethod
    def __apply_first_stage_multitest_correction(
        alphas: types.StatErrorType, hypothesis_num: int, method: str = "bonferroni"
    ) -> types.StatErrorType:
        """
        Apply first stage of multitest correction for first type errors.
        """
        if method == "bonferroni":
            alphas /= hypothesis_num
        return alphas

    @staticmethod
    def __apply_second_stage_multitest_correction(
        result: types.TesterResult, hypothesis_num: int, method: str = "bonferroni"
    ):
        """
        Apply second stage of multitest correction.
        """
        if method == "bonferroni":
            result["pvalue"] *= hypothesis_num
            result["first_type_error"] *= hypothesis_num
        return result

    @staticmethod
    def as_table(dict_result: types.TesterResult) -> pd.DataFrame:
        """
        Transform dict type output result to pandas DataFrame format.

        Parameters
        ----------
        dict_result : TesterResult
           Tester result as a dictionary.

        Returns
        -------
        result_table : pd.DataFrame
           Table with results.
        """
        answer: List[pd.DataFrame] = []
        for single_test in dict_result:
            metrics_names = list(dict_result[single_test].keys())
            metrics_names.remove("group_a_label")
            metrics_names.remove("group_b_label")
            for metric_name in metrics_names:
                tmp = deepcopy(dict_result[single_test][metric_name])
                tmp["metric name"] = metric_name
                tmp["group A label"] = dict_result[single_test]["group_a_label"]
                tmp["group B label"] = dict_result[single_test]["group_b_label"]
                if tmp["confidence_interval"][0][0] is not None:
                    tmp["confidence_interval"] = [
                        (round(left, Tester._PRECISION_DIGITS), round(right, Tester._PRECISION_DIGITS))
                        for left, right in tmp["confidence_interval"]
                    ]
                answer.append(pd.DataFrame(tmp))
        result_table = pd.concat(answer).reset_index(drop=True)
        return result_table

    def run(
        self,
        effect_type: str = "absolute",
        method: str = "theory",
        dataframe: Optional[types.PassedDataType] = None,
        df_mapping: Optional[types.GroupsInfoType] = None,
        experiment_results: Optional[types.ExperimentResults] = None,
        id_column: Optional[str] = None,
        column_groups: Optional[str] = None,
        group_labels: Optional[types.GroupLabelsType] = None,
        metrics: Optional[types.MetricNamesType] = None,
        first_errors: Optional[types.StatErrorType] = None,
        criterion: Optional[ABStatCriterion] = None,
        correction_method: Union[str, None] = "bonferroni",
        as_table: bool = True,
        **kwargs,
    ) -> types.TesterResult:
        """
        The main method for testing and evaluating experimental results.

        Parameters
        ----------
        effect_type : str, default: ``"absolute"``
           Effect type to calculate.
           Could be ``"absolute"`` or ``"relative"``.
        method : str, default: ``"theory"``
           Type of testing approach.
           Can take the values ``"theory"``, ``"empiric"`` or ``"binary"``.
        dataframe : PassedDataType, optional
           Data used to calculate the results of an experiment.
        df_mapping : GroupsInfoType, optional
           Dataframe which contains group labels of objects.
        experiment_results : ExperimentResults
            Dict with separate experiment results for each group.
            Dict keys are used as groups labels, values must be either
            pandas or Spark dataframes.
        column_groups : ColumnNameType
            Column which contains groups label of objects.
        group_labels : GroupLabelsType
            Labels for experimental groups.
        id_column : ColumnNameType
            Name of column with objects ids in ``df_mapping`` dataframe.
        first_errors : StatErrorType, default: ``0.05``
            I type errors values.
        metrics : MetricNameType
            Columns of dataframe with experiment results.
        criterion : ABStatCriterion, optional
            Statistical criterion for hypotheses testing.
            If ``method`` is ``"theory"`` and no criterion provided,
            ttest for independent samples will be used.
        correction_method : Union[str, None], default: ``bonferroni``
            Method for pvalues and confidence intervals multitest correction.
        as_table : bool, default: ``True``
            Return the test results as a pandas dataframe.
            If ``False``, a list of dicts with results will be returned.
        **kwargs : Dict
            Other keyword arguments.

        Returns
        -------
        result : types.TesterResult
            Experiment results as pandas table or list of dicts for each metric
            and first type error.
        """
        if isinstance(metrics, types.MetricNameType):
            metrics = [metrics]
        if isinstance(first_errors, float):
            first_errors = [first_errors]
        if "alternative" in kwargs:
            pvalue_pkg.check_alternative(kwargs["alternative"])

        __filtering_kwargs = {
            "dataframe": dataframe,
            "df_mapping": df_mapping,
            "column_groups": column_groups,
            "group_labels": group_labels,
            "id_column": id_column,
        }
        if dataframe is not None:
            experiment_results = DataframeHandler()._handle_cases(
                Tester.__filter_data, Tester.__filter_spark_data, **__filtering_kwargs
            )

        arguments_choice: types._PrepareArgumentsType = {
            "experiment_results": (self.__experiment_results, experiment_results),
            "metrics": (self.__metrics, metrics),
            "alpha": (self.__alpha, first_errors),
        }
        chosen_args: types._UsageArgumentsType = Tester._prepare_arguments(arguments_choice)
        chosen_args["effect_type"] = effect_type
        chosen_args["criterion"] = criterion

        hypothesis_num: int = len(list(itertools.combinations(chosen_args["experiment_results"], 2)))
        correction_available: Optional[bool] = None
        if hypothesis_num > 1:
            if correction_method in AVAILABLE_MULTITEST_CORRECTIONS:
                correction_available = True
            else:
                raise ValueError(f"Choose correction method from {AVAILABLE_MULTITEST_CORRECTIONS}")

        if correction_available:
            chosen_args["alpha"] = Tester.__apply_first_stage_multitest_correction(
                chosen_args["alpha"], hypothesis_num, correction_method
            )

        result: types.TesterResult = {}
        # Variating over all pairs of groups - comb(n, 2)
        for group_a_label, group_b_label in itertools.combinations(chosen_args["experiment_results"], 2):
            test_name = f"group_{group_a_label}_vs_group_{group_b_label}"
            chosen_args["data_a_group"] = chosen_args["experiment_results"][group_a_label]
            chosen_args["data_b_group"] = chosen_args["experiment_results"][group_b_label]
            pre_run_args = (method, chosen_args)
            subresult: types.TesterResult = DataframeHandler()._handle_on_table(
                Tester.__pre_run, Tester.__pre_run_spark, chosen_args["data_a_group"], *pre_run_args, **kwargs
            )
            subresult["group_a_label"] = group_a_label
            subresult["group_b_label"] = group_b_label
            result[test_name] = subresult

        result = Tester.as_table(result)
        if correction_available:
            result = Tester.__apply_second_stage_multitest_correction(result, hypothesis_num, correction_method)
        if not as_table:
            result = result.to_dict(orient="records")
        return result


def test(
    effect_type: str = "absolute",
    method: str = "theory",
    dataframe: Optional[types.PassedDataType] = None,
    df_mapping: Optional[types.GroupsInfoType] = None,
    experiment_results: Optional[types.ExperimentResults] = None,
    id_column: Optional[str] = None,
    column_groups: Optional[str] = None,
    group_labels: Optional[types.GroupLabelsType] = None,
    metrics: Optional[types.MetricNamesType] = None,
    first_errors: Optional[types.StatErrorType] = None,
    criterion: Optional[ABStatCriterion] = None,
    correction_method: Union[str, None] = "bonferroni",
    as_table: bool = True,
    **kwargs,
) -> types.TesterResult:
    """
    Standalone function used to get the results of an experiment.

    Creates an instance of the ``Tester`` class internally and execute
    run method with corresponding arguments.

    Parameters
    ----------
    effect_type : str, default: ``"absolute"``
        Effect type to calculate.
        Could be ``"absolute"`` or ``"relative"``.
    method : str, default: ``"theory"``
        Type of testing approach.
        Can take the values ``"theory"``, ``"empiric"`` or ``"binary"``.
    dataframe : PassedDataType, optional
        Data used to calculate the results of an experiment.
    df_mapping : GroupsInfoType, optional
        Dataframe which contains group labels of objects.
    experiment_results : ExperimentResults
        Dict with separate experiment results for each group.
        Dict keys are used as groups labels, values must be either
        pandas or Spark dataframes.
    column_groups : ColumnNameType
        Column which contains groups label of objects.
    group_labels : GroupLabelsType
        Labels for experimental groups.
    id_column : ColumnNameType
        Name of column with objects ids in ``df_mapping`` dataframe.
    first_errors : StatErrorType, default: ``0.05``
        I type errors values.
    metrics : MetricNameType
        Columns of dataframe with experiment results.
    criterion : ABStatCriterion, optional
        Statistical criterion for hypotheses testing.
        If ``method`` is ``"theory"`` and no criterion provided,
        ttest for independent samples will be used.
    correction_method : Union[str, None], default: ``bonferroni``
        Method for pvalues and confidence intervals multitest correction.
    as_table : bool, default: ``True``
        Return the test results as a pandas dataframe.
        If ``False``, a list of dicts with results will be returned.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    result : types.TesterResult
        Experiment results as pandas table or list of dicts for each metric
        and first type error.
    """
    return Tester(
        dataframe=dataframe,
        df_mapping=df_mapping,
        id_column=id_column,
        column_groups=column_groups,
        group_labels=group_labels,
        metrics=metrics,
        first_errors=first_errors,
    ).run(
        effect_type=effect_type,
        method=method,
        experiment_results=experiment_results,
        criterion=criterion,
        correction_method=correction_method,
        as_table=as_table,
        **kwargs,
    )
