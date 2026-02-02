==================
Effect Measurement
==================

Tools for assessing the statistical significance of completed experiments
and calculating the experimental uplift value with corresponding confidence intervals.

.. admonition:: Multiple testing correction
   :class: caution

   Currently, if multiple hypothesis(number of variants combinations * number of metrics passed) are tested, 
   these groups are compared in pairs and Bonferroni correction is applied to all p-values and confidence intervals.


.. currentmodule:: ambrosia.tester

.. autosummary::
    :nosignatures:

    Tester
    test

----

.. autoclass:: Tester
   :members: run
.. autofunction:: test

Examples of using testing tools
-------------------------------

.. toctree::
    :maxdepth: 1

    /pandas_examples/06_pandas_tester
    /spark_examples/09_spark_tester