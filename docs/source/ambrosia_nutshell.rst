.. role:: bolditalic
    :class: bolditalic

.. brief description 

A/B testing with *Ambrosia* in a Nutshell
-----------------------------------------

Imagine that you want to run your own A/B test, and after the product analysis and gathering ideas 
into a hypothesis, you usually have to go through several routine calculation steps: from collecting 
and transforming raw data to measuring the statistical significance of the experiment result 
and confidence intervals construction.

In order to solve the problem of carrying out a large number of calculations using various techniques,
in *Ambrosia*, we have identified the following stages of experiments and provide tools and automation for them:

- :bolditalic:`Process`

Raw data aggregation, outliers removal, metric transformation
as well as various methods for experiments acceleration.
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