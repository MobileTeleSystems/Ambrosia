:hide-toc:

.. role:: bolditalic
    :class: bolditalic

.. include:: ../../README.rst
    :end-before: shields end

.. include:: ../../README.rst
    :start-after: title
    :end-before: documentation

.. brief description 

A/B testing with *Ambrosia* in a Nutshell
-----------------------------------------

Imagine that you want to run your own A/B test, and after analyzing the product and forming ideas into a hypothesis,
you usually have to go through several routine calculation steps: from collecting and transforming raw data 
to measuring the statistical significance of the experiment result 
and construction of the corresponding confidence intervals.

In order to solve the problem of carrying out a large number of different calculations using various techniques,
in *Ambrosia*, we have identified the following stages of experiments and provide actionsand automation for them:

- :bolditalic:`Process`

Raw data aggregation, outlier removal, metric transformation
as well as various methods for accelerating experiments.
Storable data processing pipelines that can be reused.

- :bolditalic:`Design`

Experiment parameters such as effect uplift, groups size, 
and experiment statistical power are designed using metrics historical data 
by a theoretical or empirical approaches.

- :bolditalic:`Split`

Group split methods support different strategies and multi-group split, 
which allows to quickly create control and test groups of interest.
Currently, only batch data splitting methods are supported.

- :bolditalic:`Test`

Tools for the statistical inference are able to calculate relative and absolute effects,
construct corresponding confidence intervals for continious and binary variables. 
A significant number of statistical tests is supported, such as t-test, 
non-parametric, bootstrap, and others.

.. toctree::
    :maxdepth: 1
    :caption: Quickstart
    :name: quickstart
    :hidden:

    installation
    usage

.. toctree::
    :maxdepth: 1
    :caption: Core Functionality
    :name: mastertoc
    :hidden:

    ambrosia_elements/preprocessing
    ambrosia_elements/designer
    ambrosia_elements/splitter
    ambrosia_elements/tester

.. toctree::
    :maxdepth: 1
    :caption: Develop
    :name: develop
    :hidden:

    develop
    changelog
    authors

.. toctree::
    :maxdepth: 1
    :caption: Usage examples
    :name: usage examples
    :hidden:

    nb_pandas_examples
    nb_spark_examples
    ab_cases

.. toctree::
    :maxdepth: 1
    :caption: Project Links
    :name: links
    :hidden:

    GitHub Repository <https://github.com/MobileTeleSystems/Ambrosia>
    PyPI <https://pypi.org/project/ambrosia> 
    Telegram Chat <https://t.me/+Tkt43TNUUSAxNWNi>
