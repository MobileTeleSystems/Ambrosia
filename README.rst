.. shields start

Ambrosia
========

|PyPI| |PyPI License| |ReadTheDocs| |Tests| |Coverage| |Black| |Python Versions|

.. |PyPI| image:: https://img.shields.io/pypi/v/ambrosia
    :target: https://pypi.org/project/ambrosia
.. |PyPI License| image:: https://img.shields.io/pypi/l/ambrosia.svg
    :target: https://github.com/MobileTeleSystems/Ambrosia/blob/main/LICENSE
.. |ReadTheDocs| image:: https://img.shields.io/readthedocs/ambrosia.svg
    :target: https://ambrosia.readthedocs.io
.. |Tests| image:: https://img.shields.io/github/workflow/status/MobileTeleSystems/RecTools/Test/main?label=tests
    :target: https://github.com/MobileTeleSystems/Ambrosia/actions/workflows/test.yaml?query=branch%3Amain+
.. |Coverage| image:: https://codecov.io/gh/MobileTeleSystems/Ambrosia/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/MobileTeleSystems/Ambrosia
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/ambrosia.svg
    :target: https://pypi.org/project/ambrosia  

.. shields end

.. image:: https://raw.githubusercontent.com/MobileTeleSystems/Ambrosia/main/docs/source/_static/ambrosia.png

.. title

**Ambrosia** is a Python library for A/B tests design, split and effect 
measurement. It provides rich set of methods for conducting full 
A/B test pipeline. 

An experiment design stage is performed using metrics historical data 
which could be processed in both forms of pandas and spark dataframes 
with either theoretical or empirical approach. 

Group split methods support different strategies and multi-group split, 
which allows to quickly create control and test groups of interest. 

Final effect measurement stage is conducted via testing tools that 
are able to return relative and absolute effects and construct corresponding 
confidence intervalsfor continious and binary variables. 
Testing tools as well as design ones support significant number of 
statistical criteria, like t-test, non-parametric, and bootstrap. 

For additional A/B tests support library provides features and tools 
for data preproccesing and experiment acceleration.

.. functional

Key functionality
-----------------

* Pilots design ✈
* Multi-group split 🎳
* Matching of new control group to the existing pilot 🎏
* Getting the experiments result evaluation as p-value, point estimate of effect and confidence interval 🎞
* Experiments acceleration 🎢

.. documentation

Documentation
-------------

For more details, see the `Documentation <https://ambrosia.readthedocs.io/>`_ 
and `Tutorials <https://github.com/MobileTeleSystems/Ambrosia/tree/main/examples>`_.

.. install

Installation
------------

Stable version is released on every tag to ``main`` branch. 

.. code:: bash
    
    pip install ambrosia 

**Ambrosia requires Python 3.7+**

.. usage

Usage
-----

Designer 
~~~~~~~~

.. code:: python

    from ambrosia.designer import Designer
    designer = Designer(dataframe=df, effects=1.2, metrics='portfel_clc') # 20% effect, and loaded data frame df
    designer.run('size') 


Splitter
~~~~~~~~

.. code:: python

    from ambrosia.splitter import Splitter
    splitter = Splitter(dataframe=df, id_column='id') # loaded data frame df with column with id - 'id'
    splitter.run(groups_size=500, method='simple') 


Tester 
~~~~~~

.. code:: python

    from ambrosia.tester import Tester
    tester = Tester(dataframe=df, column_groups='group') # loaded data frame df with groups info 'group'
    tester.run(metrics='retention', method='theory', criterion='ttest')

.. develop

Development
-----------

To install all requirements run

.. code:: bash

    make install

You must have ``python3`` and ``poetry`` installed.

For autoformatting run

.. code:: bash

    make autoformat

For linters check run

.. code:: bash

    make lint

For tests run

.. code:: bash

    make test

For coverage run

.. code:: bash

    make coverage

To remove virtual environment run

.. code:: bash

    make clean

.. contributors

Communication
-------------

**Developers and evangelists**:

* `Bayramkulov Aslan <https://github.com/aslanbm>`_
* `Khakimov Artem <https://github.com/xandaau>`_
* `Vasin Artem <https://github.com/VictorFromChoback>`_
