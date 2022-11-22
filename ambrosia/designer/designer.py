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
Experiment design methods.

Module contains `Designer` core class and `design` method which are
intended to conduct the experiment design for A/B/.. tests via different
methods.

Experiment design of the individual metric is based on its historical data
and could be done for any parameter from the self-dependent triplet:
group size, effect size and experiment power.

Currently, experiment design problem could be solved using data provided
in form of both pandas and Spark(with some restrictions) dataframes.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

import ambrosia.tools.bin_intervals as bin_pkg
from ambrosia import types
from ambrosia.tools.ab_abstract_component import ABMetaClass, ABToolAbstract, SimpleDesigner

from .handlers import EmpiricHandler, TheoryHandler, calc_prob_control_class

SIZE: str = "size"
EFFECT: str = "effect"
POWER: str = "power"


class Designer(yaml.YAMLObject, ABToolAbstract, metaclass=ABMetaClass):
    """
    Unit for experiments and pilots design.

    Enables to design missing experiment parameters using historical data.
    The main related to each other designable parameters for a single metric are:
        - Effect (Minimal Detectible Effect):
            old_mean_metric_value * effect_value = new_mean_metric_value
        - Sample size:
            Number of research objects in sample
            (for example number of users and their retention).
        - Errors (I type error, II type error):
            I error (alpha):
                Probability to detect presence of effect
                for equally distributed samples.
            II error (beta):
                Probability not to find effect
                for differently distributed samples.

    Parameters
    ----------
    dataframe : PassedDataType, optional
        DataFrame with metrics historical values.
    sizes : SampleSizeType, optional
        Values of research objects number in groups samples during
        the experiment.
    effects : EffectType, optional
        Effects values that are expected during the experiment.
    first_type_errors : StatErrorType, default: ``0.05``
        I type error bounds
        P (detect difference for equal) < alpha.
    second_type_errors : StatErrorType, default: ``0.2``
        II type error bounds
        P (suppose equality for different groups) < beta.
    metrics : MetricNamesType, optional
        Column names of metrics in dataframe to be designed.
    method : str, optional
        Method used for experiment design.
        Can be ``"theory"``, ``"empiric"`` or ``"binary"``.


    Attributes
    ----------
    dataframe : PassedDataType
        DataFrame with metrics historocal values.
    sizes : SampleSizeType
        Number of research objects in group samples.
    effects : EffectType
        Effects values in the experiment.
    first_type_errors : StatErrorType, default: ``0.05``
        I type errors.
    second_type_errors : StatErrorType, default: ``0.2``
        II type errors.
    metrics : MetricNamesType
        Column names of metrics in dataframe to be designed.
    method : str
        Method used for experiment design.

    Examples
    --------
    We have retention labels for users of mobile app for previous month.
    Suppose old_retention = ``0.3``, that is 30% of users returned to the app
    in a month after installation.

    Let us fix the following parameters:
        I type error (alpha) = ``0.05``
        (5% of equal samples we can suppose to be different).

        II type error (beta) = ``0.2``
        (20% of different sampels we can suppose to be equal).

    We add onboarding to our app and want to estimate an effect, by A/B testing
    and wish to increase retention value to 31% percents, so our effect
    parameter gets value of ``1.0(3)``. Now we want to find how much users we
    need in both groups to detect such effect.

    We can use ``Designer`` class in the following way:
        >>> designer = Designer(dataframe=df, metric='retention', effect=1.033)
        >>> designer.run("size")

    Note, that default values for errors are:
        ``first_type_error`` = ``0.05``

        ``second_type_error`` = ``0.2``

    Then we get dataframe that contains value of  sufficient number of users
    for our experiment.

    Notes
    -----

    Constructors:
        >>> designer = Designer()
        >>> # You can pass an Iterable or single object for some parameters
        >>> designer = Designer(
        >>>     dataframe=df,
        >>>     sizes=[100, 200],
        >>>     metrics='LTV',
        >>>     effects=1.05
        >>> )
        >>> designer = Desginer(sizes=1000, metrics=['retention', 'LTV'])
        >>> # You can use path to .csv table for pandas
        >>> designer = Designer('./data/table.csv')

    Setters:
        >>> designer.set_first_errors([0.05, 0.01])
        >>> desginer.set_dataframe(df)

    Run:
        >>> # One can pass arguments and they will have higher priority
        >>> designer.run('size', effects=1.1)
        >>> designer.run('effect', sizes=[500, 1000], metrics='retention')
        >>> # You can set method (watch below)
        >>> designer.run('effect', sizes=[500, 1000], metrics='retention', method='binary')

    Load from yaml config:
        >>> config = '''
                !splitter # <--- this is yaml tag (!important)
                    effects:
                        - 0.9
                        - 1.05
                    sizes:
                        - 1000
                    dataframe:
                        ./data/table.csv
            '''
        >>> designer = yaml.load(config)
        >>> # Or use the implmented function
        >>> designer = load_from_config(config)

    Use standalone function instead of a class:
        >>> design('size', dataframe=df, effects=1.05, metrics='retention')
    """

    # YAML tag for loading from configs
    yaml_tag = "!designer"

    def set_first_errors(self, first_type_errors: types.StatErrorType) -> None:
        if isinstance(first_type_errors, float):
            self.__alpha = [first_type_errors]
        else:
            self.__alpha = first_type_errors

    def set_second_errors(self, second_type_errors: types.StatErrorType) -> None:
        if isinstance(second_type_errors, float):
            self.__beta = [second_type_errors]
        else:
            self.__beta = second_type_errors

    def set_sizes(self, sizes: types.SampleSizeType) -> None:
        if isinstance(sizes, int):
            self.__size = [sizes]
        else:
            self.__size = sizes

    def set_effects(self, effects: types.EffectType) -> None:
        if isinstance(effects, float):
            self.__effect = [effects]
        else:
            self.__effect = effects

    def set_dataframe(self, dataframe: types.PassedDataType) -> None:
        if isinstance(dataframe, str):
            if dataframe.endswith(".csv"):
                self.__df = pd.read_csv(dataframe)
            else:
                raise ValueError("File name must ends with .csv")
        else:
            self.__df = dataframe

    def set_method(self, method: str) -> None:
        self.__method = method

    def set_metrics(self, metrics: str) -> None:
        if isinstance(metrics, types.MetricNameType):
            self.__metrics = [metrics]
        else:
            self.__metrics = metrics

    def __init__(
        self,
        dataframe: Optional[types.PassedDataType] = None,
        sizes: Optional[types.SampleSizeType] = None,
        effects: Optional[types.EffectType] = None,
        first_type_errors: types.StatErrorType = 0.05,
        second_type_errors: types.StatErrorType = 0.2,
        metrics: Optional[types.MetricNamesType] = None,
        method: str = "theory",
    ):
        """
        Designer class constructor to initialize the object.
        """
        self.set_first_errors(first_type_errors)
        self.set_second_errors(second_type_errors)
        self.set_sizes(sizes)
        self.set_effects(effects)
        self.set_metrics(metrics)
        self.set_dataframe(dataframe)
        self.set_method(method)

    @classmethod
    def from_yaml(cls, loader: yaml.Loader, node: yaml.Node):
        kwargs = loader.construct_mapping(node)
        return cls(**kwargs)

    @staticmethod
    def __dataframe_handler(handler: SimpleDesigner, parameter: str, **kwargs) -> pd.DataFrame:
        """
        Handles different dataframe types.
        Now pandas and spark are available.
        """
        if parameter == SIZE:
            return handler.size_design(**kwargs)
        elif parameter == EFFECT:
            return handler.effect_design(**kwargs)
        elif parameter == POWER:
            return handler.power_design(**kwargs)
        else:
            raise ValueError(f"Only {SIZE}, {EFFECT} and {POWER} parameters of the experiment could be designed.")

    @staticmethod
    def __theory_design(label: str, args: types._UsageArgumentsType, **kwargs) -> types.DesignerResult:
        """
        Designing an experiment, using a theoretical approach.
        """
        result: types.DesignerResult = {}
        for metric_name in args["metric"]:
            kwargs["dataframe"] = args["df"]
            kwargs["column"] = metric_name
            kwargs["first_errors"] = np.array(args["alpha"])
            if label == SIZE:
                kwargs["effects"] = args[EFFECT]
                kwargs["second_errors"] = np.array(args["beta"])
            elif label == EFFECT:
                kwargs["sample_sizes"] = args[SIZE]
                kwargs["second_errors"] = np.array(args["beta"])
            elif label == POWER:
                kwargs["sample_sizes"] = args[SIZE]
                kwargs["effects"] = args[EFFECT]
            result[metric_name] = Designer.__dataframe_handler(TheoryHandler(), label, **kwargs)
        if len(args["metric"]) == 1:
            return result[args["metric"][0]]
        else:
            return result

    @staticmethod
    def __empiric_design(label: str, args: types._UsageArgumentsType, **kwargs) -> types.DesignerResult:
        """
        Designing an experiment, using an empirical approach.
        """
        kwargs["dataframe"] = args["df"]
        kwargs["alphas"] = np.array(args["alpha"])
        kwargs["metrics"] = args["metric"]
        if label == SIZE:
            kwargs["effects"] = args[EFFECT]
            kwargs["betas"] = np.array(args["beta"])
        elif label == EFFECT:
            kwargs["group_sizes"] = args[SIZE]
            kwargs["betas"] = np.array(args["beta"])
        elif label == POWER:
            kwargs["sample_sizes_a"] = args[SIZE]
            kwargs["sample_sizes_b"] = args[SIZE]
            kwargs["effects"] = args[EFFECT]
        return Designer.__dataframe_handler(EmpiricHandler(), label, **kwargs)

    @staticmethod
    def __binary_design(label: str, args: types._UsageArgumentsType, **kwargs) -> types.DesignerResult:
        """
        Designing an experiment, using the approach for binary metrics.
        """
        result: types.DesignerResult = {}
        for metric_name in args["metric"]:
            kwargs["p_a"] = calc_prob_control_class(args["df"], metric_name)
            if label == SIZE:
                kwargs["delta_relative_values"] = args[EFFECT]
                kwargs["second_errors"] = args["beta"]
                result[metric_name] = bin_pkg.get_table_sample_size_on_effect(**kwargs)
            elif label == EFFECT:
                kwargs["second_errors"] = args["beta"]
                kwargs["sample_sizes"] = args[SIZE]
                result[metric_name] = bin_pkg.get_table_effect_on_sample_size(**kwargs)
            elif label == POWER:
                kwargs["delta_relative_values"] = args[EFFECT]
                kwargs["sample_sizes"] = args[SIZE]
                result[metric_name] = bin_pkg.get_table_power_on_size_and_delta(**kwargs)
        if len(args["metric"]) == 1:
            return result[args["metric"][0]]
        else:
            return result

    @staticmethod
    def __pre_design(label: str, args: types._UsageArgumentsType, **kwargs) -> types.DesignerResult:
        """
        Helper function for run() method logic.
        """
        admissible_methods: List[str] = ["theory", "empiric", "binary"]
        if args["method"] == "theory":
            return Designer.__theory_design(label, args, **kwargs)
        elif args["method"] == "empiric":
            return Designer.__empiric_design(label, args, **kwargs)
        elif args["method"] == "binary":
            return Designer.__binary_design(label, args, **kwargs)
        else:
            raise ValueError(f'Choose method from {", ".join(admissible_methods)}, got {args["method"]}')

    def run(
        self,
        to_design: str,
        method: Optional[str] = None,
        sizes: Optional[types.SampleSizeType] = None,
        effects: Optional[types.EffectType] = None,
        first_type_errors: Optional[types.StatErrorType] = None,
        second_type_errors: Optional[types.StatErrorType] = None,
        dataframe: Optional[types.PassedDataType] = None,
        metrics: Optional[types.MetricNamesType] = None,
        **kwargs,
    ) -> types.DesignerResult:
        """
        Perform an experiment design for chosen parameter and metrics
        using historical data.

        Parameters
        ----------
        to_design : str
           Parameter that will be designed using historical data.
           Can take the values of ``"size"``, ``"effect"`` or ``"power"``.
        method : str, optional
            Method used for experiment design.
            Can be ``"theory"``, ``"empiric"`` or ``"binary"``.
        sizes : SampleSizeType, optional
            Values of research objects number in groups samples during
            the experiment.
            If is not provided, must exist as proper class attribute.
        effects : EffectType, optional
            Effects for experiment
            If is not provided, must exist as proper class attribute.
        first_type_errors : StatErrorType, optional
            I type error bounds
            P (detect difference for equal) < alpha.
        second_type_errors : StatErrorType, optional
            II type error bounds
            P (suppose equality for different groups) < beta.
        dataframe : PassedDataType, optional
            DataFrame with metrics historical values.
            If is not provided, must exist as proper class attribute.
        metrics : MetricNamesType, optional
            Column names of metrics in dataframe to be designed.
            If not provided, must exist as proper class attribute.
        **kwargs : Dict
            Other keyword arguments.

        Other Parameters
        ----------------
        as_numeric : bool, default: ``False``
            The result of calculations can be obtained as a percentage string
            either as a number, this parameter could used to toggle.

        Returns
        -------
        result : DesignerResult
            Table or dictionary with the results of parameter design for each
            metric.
        """
        if isinstance(effects, float):
            effects = [effects]
        if isinstance(sizes, int):
            sizes = [sizes]
        if isinstance(first_type_errors, float):
            first_type_errors = [first_type_errors]
        if isinstance(second_type_errors, float):
            second_type_errors = [second_type_errors]
        if isinstance(metrics, types.MetricNameType):
            metrics = [metrics]

        arguments_choice: types._PrepareArgumentsType = {
            "df": (self.__df, dataframe),
            "alpha": (self.__alpha, first_type_errors),
            "metric": (self.__metrics, metrics),
            "method": (self.__method, method),
        }

        designable_parameters: List[str] = [SIZE, EFFECT, POWER]

        if to_design == SIZE:
            arguments_choice[EFFECT] = (self.__effect, effects)
            arguments_choice["beta"] = (self.__beta, second_type_errors)
            chosen_args: types._UsageArgumentsType = Designer._prepare_arguments(arguments_choice)
            return Designer.__pre_design(SIZE, chosen_args, **kwargs)
        elif to_design == EFFECT:
            arguments_choice[SIZE] = (self.__size, sizes)
            arguments_choice["beta"] = (self.__beta, second_type_errors)
            chosen_args: types._UsageArgumentsType = Designer._prepare_arguments(arguments_choice)
            return Designer.__pre_design(EFFECT, chosen_args, **kwargs)
        elif to_design == POWER:
            arguments_choice[SIZE] = (self.__size, sizes)
            arguments_choice[EFFECT] = (self.__effect, effects)
            chosen_args: types._UsageArgumentsType = Designer._prepare_arguments(arguments_choice)
            return Designer.__pre_design(POWER, chosen_args, **kwargs)
        else:
            raise ValueError(f'Incorrect parameter name to design, choose from {", ".join(designable_parameters)}')


def load_from_config(yaml_config: str, loader: type = yaml.Loader) -> Designer:
    """
    Create Designer class instance from yaml config.

    For yaml_config you can pass file name with config,
    it must ends with .yaml, for example: "config.yaml".

    For loader you can choose SafeLoader.
    """
    if isinstance(yaml_config, str) and yaml_config.endswith(".yaml"):
        with open(yaml_config, "r", encoding="utf8") as file:
            return yaml.load(file, Loader=loader)
    return yaml.load(yaml_config, Loader=loader)


def design(
    to_design,
    dataframe: types.PassedDataType,
    metrics: types.MetricNamesType,
    sizes: types.SampleSizeType = None,
    effects: types.EffectType = None,
    first_type_errors: types.StatErrorType = (0.05,),
    second_type_errors: types.StatErrorType = (0.2,),
    method: str = "theory",
    **kwargs,
) -> types.DesignerResult:
    """
    Make experiment design based on historical data using passed arguments.

    Parameters
    ----------
    to_design : str
        Parameter that will be designed using historical data.
        Can take the values of ``"size"``, ``"effect"`` or ``"power"``.
    dataframe : PassedDataType
        DataFrame with metrics historical values.
    metrics : MetricNamesType
        Column names of metrics in dataframe to be designed.
    sizes : SampleSizeType, optional
        Values of research objects number in groups samples during
        the experiment.
        If is not provided, ``effects`` value must be defined.
    effects : EffectType, optional
        Effects for experiment
        If is not provided, ``sizes`` value must be defined.
    first_type_errors : StatErrorType, default: ``(0.05,)``
        I type error bounds
        P (detect difference for equal) < alpha.
    second_type_errors : StatErrorType, default: ``(0.2,)``
        II type error bounds
        P (suppose equality for different groups) < beta.
    method : str, default: ``"theory"``
        Method used for experiment design.
        Can be ``"theory"``, ``"empiric"`` or ``"binary"``.
    **kwargs : Dict
        Other keyword arguments.

    Other Parameters
    ----------------
    as_numeric : bool, default: ``False``
        The result of calculations can be obtained as a percentage string
        either as a number, this parameter could used to toggle.

    Returns
    -------
    result : DesignerResult
        Table or dictionary with the results of parameter design for each
        metric.
    """
    return Designer(
        dataframe=dataframe,
        metrics=metrics,
        first_type_errors=first_type_errors,
        second_type_errors=second_type_errors,
        sizes=sizes,
        effects=effects,
        method=method,
    ).run(to_design, **kwargs)


def design_binary_size(
    prob_a: float,
    effects: types.EffectType,
    first_type_errors: types.StatErrorType = (0.05,),
    second_type_errors: types.StatErrorType = (0.2,),
    **kwargs,
) -> pd.DataFrame:
    """
    Design size for binary metrics.

    Parameters
    ----------
    prob_a : float
        Probability of success for the control group.
    effects : EffectType
        List or single value of relative effects.
        For example: ``1.05``, ``[1.05, 1.2]``.
    first_type_errors : StatErrorType, default: ``(0.05,)``
       I type error bounds
       P (detect difference for equal) < alpha.
    second_type_errors : StatErrorType, default: ``(0.2,)``
       II type error bounds
       P (suppose equality for different groups) < beta.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    result_table : pd.DataFrame
        Table with results of design.
    """
    if isinstance(effects, float):
        effects = [effects]
    if isinstance(first_type_errors, float):
        first_type_errors = [first_type_errors]
    if isinstance(second_type_errors, float):
        second_type_errors = [second_type_errors]
    return bin_pkg.get_table_sample_size_on_effect(
        p_a=prob_a,
        first_errors=first_type_errors,
        second_errors=second_type_errors,
        delta_relative_values=effects,
        **kwargs,
    )


def design_binary_effect(
    prob_a: float,
    sizes: types.SampleSizeType,
    first_type_errors: types.StatErrorType = (0.05,),
    second_type_errors: types.StatErrorType = (0.2,),
    **kwargs,
) -> pd.DataFrame:
    """
    Design effect for binary metrics.

    Parameters
    ----------
    prob_a : float
         Probability of success for the control group.
    sizes : SampleSizeType
        List or single value of group sizes.
        For example: ``100``, ``[100, 200]``.
    first_type_errors : StatErrorType, default: ``(0.05,)``
       I type error bounds
       P (detect difference for equal) < alpha.
    second_type_errors : StatErrorType, default: ``(0.2,)``
       II type error bounds
       P (suppose equality for different groups) < beta.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    result_table : pd.DataFrame
        Table with results of design.
    """
    if isinstance(sizes, int):
        sizes = [sizes]
    if isinstance(first_type_errors, float):
        first_type_errors = [first_type_errors]
    if isinstance(second_type_errors, float):
        second_type_errors = [second_type_errors]
    return bin_pkg.get_table_effect_on_sample_size(
        p_a=prob_a, sample_sizes=sizes, first_errors=first_type_errors, second_errors=second_type_errors, **kwargs
    )


def design_binary_power(
    prob_a: float,
    sizes: types.SampleSizeType,
    effects: types.EffectType,
    first_type_errors: types.StatErrorType = 0.05,
    **kwargs,
) -> pd.DataFrame:
    """
    Design power for binary metrics.

    Parameters
    ----------
    prob_a : float
       Probability of success for the control group.
    sizes : SampleSizeType
        List of single value of group sizes.
        For example: ``100``, ``[100, 200]``.
    effects : EffectType
        List or single value of relative effects.
        For example: ``1.05``, ``[1.05, 1.2]``.
    first_type_errors : StatErrorType, default: ``0.05``
       I type error bounds
       P (detect difference for equal) < alpha.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    result_table : pd.DataFrame
        Table with results of design.
    """
    if isinstance(effects, float):
        effects = [effects]
    if isinstance(sizes, int):
        sizes = [sizes]
    return bin_pkg.get_table_power_on_size_and_delta(
        p_a=prob_a, sample_sizes=sizes, confidence_level=1 - first_type_errors, delta_relative_values=effects, **kwargs
    )


def design_binary(
    to_design: str,
    prob_a: float,
    sizes: Optional[types.SampleSizeType] = None,
    effects: Optional[types.EffectType] = None,
    first_type_errors: types.StatErrorType = 0.05,
    second_type_errors: types.StatErrorType = (0.2,),
    **kwargs,
) -> pd.DataFrame:
    """
    Design desired parameter for binary metrics.

    Parameters
    ----------
    to_design : str
        Parameter to design.
    prob_a : float
        Probability of success for the control group.
    sizes : SampleSizeType, optional
        List or single value of group sizes.
        For example: ``100``, ``[100, 200]``.
    effects : EffectType, optional
        List of single value of relative effects.
        For example: 1.05, [1.05, 1.2].
    first_type_errors : StatErrorType, default: ``0.05``
       I type error bounds
       P (detect difference for equal) < alpha.
    second_type_errors : StatErrorType, default: ``(0.2,)``
       II type error bounds
       P (suppose equality for different groups) < beta.
    **kwargs : Dict
        Other keyword arguments.

    Returns
    -------
    result_table : pd.DataFrame
        Table with results of design.
    """
    if to_design == SIZE:
        return design_binary_size(prob_a, effects, first_type_errors, second_type_errors, **kwargs)
    elif to_design == EFFECT:
        return design_binary_effect(prob_a, sizes, first_type_errors, second_type_errors, **kwargs)
    elif to_design == POWER:
        return design_binary_power(prob_a, sizes, effects, first_type_errors, **kwargs)
    else:
        raise ValueError(f"Only {SIZE}, {EFFECT} and {POWER} parameters of the binary experiment could be designed.")
