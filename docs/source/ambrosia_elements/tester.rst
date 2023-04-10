==================
Effect measurement
==================

Tools for assessing the statistical significance of completed experiments
and calculating the experimental uplift value with corresponding confidence intervals.

.. admonition:: Multiple testing correction
   :class: caution

   Currently, if multiple groups are tested, these groups are compared in pairs and automatic 
   Bonferroni correction is applied to all alpha values and confidence intervals.


.. currentmodule:: ambrosia.tester

.. autosummary::
    :nosignatures:

    Tester
    test

----

.. autoclass:: Tester
   :members: run
.. autofunction:: test

Examples using testing tools
----------------------------

.. nblinkgallery::
    :name: tester-examples

    /pandas_examples/design_binary
    /spark_examples/spark_api