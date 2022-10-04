Preprocessing
=============


.. currentmodule:: ambrozia.preprocessing

.. autosummary::
    :nosignatures:

    Preprocessor
    AggregatePreprocessor
    RobustPreprocessor
    Cuped
    MultiCuped

.. autoclass:: ambrozia.preprocessing.Preprocessor
   :members: aggregate, robust, cuped, data, transformations

.. autoclass:: ambrozia.preprocessing.AggregatePreprocessor
   :members: run, transform, get_params_dict

.. autoclass:: ambrozia.preprocessing.RobustPreprocessor
   :members: run

.. autoclass:: ambrozia.preprocessing.Cuped
   :members: fit, transform, fit_transform

.. autoclass:: ambrozia.preprocessing.MultiCuped
   :members: fit, transform, fit_transform