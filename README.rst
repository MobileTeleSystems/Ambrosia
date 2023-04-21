.. shields start

Ambrosia
========

|PyPI| |PyPI License| |ReadTheDocs| |Tests| |Coverage| |Black| |Python Versions| |Telegram Channel|

.. |PyPI| image:: https://img.shields.io/pypi/v/ambrosia
    :target: https://pypi.org/project/ambrosia
.. |PyPI License| image:: https://img.shields.io/pypi/l/ambrosia.svg
    :target: https://github.com/MobileTeleSystems/Ambrosia/blob/main/LICENSE
.. |ReadTheDocs| image:: https://img.shields.io/readthedocs/ambrosia.svg
    :target: https://ambrosia.readthedocs.io
.. |Tests| image:: https://img.shields.io/github/actions/workflow/status/MobileTeleSystems/Ambrosia/test.yaml?branch=main
    :target: https://github.com/MobileTeleSystems/Ambrosia/actions/workflows/test.yaml?query=branch%3Amain+
.. |Coverage| image:: https://codecov.io/gh/MobileTeleSystems/Ambrosia/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/MobileTeleSystems/Ambrosia
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/ambrosia.svg
    :target: https://pypi.org/project/ambrosia  
.. |Telegram Channel| image:: https://img.shields.io/badge/telegram-Ambrosia-blueviolet.svg?logo=telegram
    :target: https://t.me/+Tkt43TNUUSAxNWNi

.. shields end

.. image:: https://raw.githubusercontent.com/MobileTeleSystems/Ambrosia/main/docs/source/_static/ambrosia.png
   :height: 320 px
   :width: 320 px
   :align: center

.. title

*Ambrosia* is a Python library for A/B tests design, split and effect measurement. 
It provides rich set of methods for conducting full A/B testing pipeline. 

The project is intended for use in research and production environments 
based on data in pandas and Spark format.

.. functional

Key functionality
-----------------

* Pilots design 🛫
* Multi-group split 🎳
* Matching of new control group to the existing pilot 🎏
* Experiments result evaluation as p-value, point estimate of effect and confidence interval 🎞
* Data preprocessing ✂️
* Experiments acceleration 🎢

.. documentation

Documentation
-------------

For more details, see the `Documentation <https://ambrosia.readthedocs.io/>`_ 
and `Tutorials <https://github.com/MobileTeleSystems/Ambrosia/tree/main/examples>`_.

.. install

Installation
------------

You can always get the newest *Ambrosia* release using ``pip``.
Stable version is released on every tag to ``main`` branch. 

.. code:: bash
    
    pip install ambrosia 

Starting from version ``0.4.0``, the ability to process PySpark data is optional and can be enabled 
using ``pip`` extras during the installation.

.. code:: bash
    
    pip install ambrosia[pyspark]

.. usage

Usage
-----

The main functionality of *Ambrosia* is contained in several core classes and methods, 
which are autonomic for each stage of an experiment and have very intuitive interface. 

|

Below is a brief overview example of using a set of three classes to conduct some simple experiment.

**Designer**

.. code:: python

    from ambrosia.designer import Designer
    designer = Designer(dataframe=df, effects=1.2, metrics='portfel_clc') # 20% effect, and loaded data frame df
    designer.run('size') 


**Splitter**

.. code:: python

    from ambrosia.splitter import Splitter
    splitter = Splitter(dataframe=df, id_column='id') # loaded data frame df with column with id - 'id'
    splitter.run(groups_size=500, method='simple') 


**Tester**

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

Authors
-------

**Developers and evangelists**:

* `Bayramkulov Aslan <https://github.com/aslanbm>`_
* `Khakimov Artem <https://github.com/xandaau>`_
* `Vasin Artem <https://github.com/VictorFromChoback>`_
