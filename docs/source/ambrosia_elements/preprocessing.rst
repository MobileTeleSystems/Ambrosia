==================
Data Preprocessing
==================

The tools from this subsection allow to automatically perform various stages of processing experimental data 
and save the specified configurations for repeated data transformations.

Data preprocessing tools:

.. toctree::
   :maxdepth: 1

   aggregation
   robust
   simple_transformation
   advanced_transformations
   processor

.. admonition:: Chain preprocessing
   :class: Tip

   Almost all separate data transformations are available as sequential methods of the ``Preprocessor`` class.


Examples of using data transformation tools
-------------------------------------------

.. nblinkgallery::
    :name: preprocessing-examples

   /pandas_examples/00_preprocessing
   /pandas_examples/01_vr_transformations
   /pandas_examples/02_preprocessor
   /pandas_examples/11_cuped_example