=================
Experiment design
=================

*Ambrosia* offers tools for calculating A/B test parameters such as effect uplift,
groups size, and experiment statistical power, based on historical metrics values.

.. admonition:: Choice of design approach
   :class: Tip

   The theoretical approach to designing experimental parameters is much faster than the empirical one.

.. currentmodule:: ambrosia.designer
    
.. autosummary::
    :nosignatures:

    Designer
    load_from_config
    design
    design_binary

----

.. autoclass:: Designer
   :members: run
.. autofunction:: load_from_config
.. autofunction:: design
.. autofunction:: design_binary


Examples using experiment design tools
--------------------------------------

.. nblinkgallery::
    :name: designer-examples

    /pandas_examples/design_binary
    /spark_examples/spark_api