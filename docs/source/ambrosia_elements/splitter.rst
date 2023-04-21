================
Groups Splitting
================

The following classes and functions helps to split batch data into
experimental groups using different approaches.

.. admonition:: Real-time Splitter availability
   :class: caution

   The real-time splitting tools are under development. This functionality is intended to be applied to batch data only.

.. currentmodule:: ambrosia.splitter

.. autosummary::
    :nosignatures:

    Splitter
    load_from_config
    split

----

.. autoclass:: Splitter
   :members: run
.. autofunction:: load_from_config
.. autofunction:: split

Examples of using groups splitting tools
----------------------------------------

.. nblinkgallery::
    :name: splitter-examples

    /pandas_examples/05_pandas_splitter
    /spark_examples/08_spark_splitter