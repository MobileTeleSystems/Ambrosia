Preprocessing
=============


.. currentmodule:: ambrosia.preprocessing

.. autosummary::
    :nosignatures:

    Preprocessor
    AggregatePreprocessor
    RobustPreprocessor
    Cuped
    MultiCuped

.. autoclass:: ambrosia.preprocessing.Preprocessor
   :members: aggregate, robust, cuped, data, transformations

.. autoclass:: ambrosia.preprocessing.AggregatePreprocessor
   :members: run, transform, get_params_dict

.. autoclass:: ambrosia.preprocessing.RobustPreprocessor
   :members: run

.. autoclass:: ambrosia.preprocessing.Cuped
   :members: fit, transform, fit_transform

.. autoclass:: ambrosia.preprocessing.MultiCuped
   :members: fit, transform, fit_transform