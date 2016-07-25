import datetime as dt

import numpy as np
import pandas as pd

from ..compat.python import add_metaclass
from ..utility.array import (DocStringInheritor, date_to_index, ensure1d,
                             ensure2d)


@add_metaclass(DocStringInheritor)
class ARCHModel(object):
    """
    Abstract base class for mean models with ARCH processes.
    Specifies the conditional mean process.

    All public methods that raise NotImplementedError should be overridden by
    any subclass.  Private methods that raise NotImplementedError are optional
    to override but recommended where applicable.
    """

    def __init__(self, y=None, volatility=None, distribution=None,
                 hold_back=None, last_obs=None):

        self._is_pandas = isinstance(y, (pd.DataFrame, pd.Series))

        ndim = y.ndim
        self._y_original = y
        self._y = np.asarray(self._y_original)
        if ndim == 1:
            self._y_pd = ensure1d(y, 'y', series=True)
        else:
            ensure2d(y, 'y', dataframe=True)

        self.hold_back = hold_back
        if isinstance(hold_back, (str, dt.datetime, np.datetime64)):
            date_index = self._y_pd.index
            _first_obs_index = date_to_index(hold_back, date_index)
            self.first_obs = date_index[_first_obs_index]
        elif hold_back is None:
            self.first_obs = _first_obs_index = 0
        else:
            _first_obs_index = hold_back
            self.first_obs = self._y_pd.index[_first_obs_index]

        self.last_obs = _last_obs_index = last_obs
        if isinstance(last_obs, (str, dt.datetime, np.datetime64)):
            date_index = self._y_pd.index
            _last_obs_index = date_to_index(last_obs, date_index)
            self.last_obs = date_index[_last_obs_index]
        elif last_obs is None:
            self.last_obs = _last_obs_index = self._y.shape[0]
        else:
            self.last_obs = self._y_pd.index[last_obs]

        self.nobs = _last_obs_index - _first_obs_index
        self._indices = (_first_obs_index, _last_obs_index)

        self._volatility = volatility
        self._distribution = distribution
        self._backcast = None
