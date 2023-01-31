Preprocessing
=============


.. currentmodule:: ambrosia.preprocessing

.. autosummary::
    :nosignatures:

    Preprocessor
    AggregatePreprocessor
    RobustPreprocessor
    IQRPreprocessor
    BoxCoxTransformer
    LogTransformer
    Cuped
    MultiCuped
    MLVarianceReducer

.. autoclass:: ambrosia.preprocessing.Preprocessor
   :members: aggregate, robust, iqr, boxcox, log, cuped, data, transformations

.. autoclass:: ambrosia.preprocessing.AggregatePreprocessor
   :members: run, transform, get_params_dict

.. autoclass:: ambrosia.preprocessing.RobustPreprocessor
   :members: fit, transform, fit_transform, store_params, load_params

.. autoclass:: ambrosia.preprocessing.IQRPreprocessor
   :members: fit, transform, fit_transform, store_params, load_params

.. autoclass:: ambrosia.preprocessing.BoxCoxTransformer
   :members: fit, transform, fit_transform, store_params, load_params

.. autoclass:: ambrosia.preprocessing.LogTransformer
   :members: fit, transform, fit_transform, store_params, load_params

.. autoclass:: ambrosia.preprocessing.Cuped
   :members: fit, transform, fit_transform, store_params, load_params

.. autoclass:: ambrosia.preprocessing.MultiCuped
   :members: fit, transform, fit_transform, store_params, load_params

.. autoclass:: ambrosia.preprocessing.MLVarianceReducer
   :members: fit, transform, fit_transform, store_params, load_params