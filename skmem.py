""" skmem.MemReducer
    Smart memory reduction for pandas.

    Pandas is very flexible when loading data. You can set dtypes by column
    and even use lists of columns or dictionaries. But it can get tedious when
    working with many columns. And, it's not always clear how types should be
    changed until you load the data and look at it. This transformer provides
    a way to quickly reduce dataframe memory by converting memory-hungry
    dtypes to ones needing less memory. Advantages include:
        - Fully compatible with scikit-learn. Combine with other transformers
          and pipelines with ease.
        - Preserves data integrity. Set simple parameters to control
          treatment of floats and objects.
        - Easy to customize. Use class inheritance or directly change modular
          functions as needed.
        - Efficient. Save time with vectorized functions that process data
          faster than most parallelized solutions.
        - Fixes dataframe column names. Use an optional function to convert
          column names to snake_case.
"""


import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation


class MemReducer(BaseEstimator, TransformerMixin):
    """ Reduce dataframe memory by converting dataframe columns to dtypes
        requiring less memory. Returns a dataframe with memory-efficient
        dtypes where possible.

        Integers, 64-bit floats and objects/strings can be converted.
        Parameters provide control for treatment of floats and objects.

        Parameters
        ___________
        max_unique_pct : float, optional, default=0.5
        Sets maximum threshold for converting object columns to categoricals.
        Threshold is compared to the number of unique values as a percent of
        column length. 0.0 prevents all conversions and 1.0 allows all
        conversions.

        snake_case : boolean, optional, default=False
        If True, converts dataframe column names to snake_case.

        Example
        --------
        >>> import skmem
        >>> df = pd.DataFrame({'Cats': np.tile(['a', 'b'], 500_000),
                    'trueInts': np.tile(np.arange(-5, 5), 100_000),
                    'floats': np.arange(0., 1_000_000.)
                    })
        >>> print(df.dtypes)
        |Cats      object
        |trueInts    int64
        |floats    float64
        |dtype: object
        >>> mr = skmem.MemReducer(max_unique_pct=0.8, snake_case=True)
        >>> df_small = mr.fit_transform(df, float_cols=['floats'])
        |Memory in: 0.08 GB
        |Starting integers.
        |Starting floats.
        |Starting objects.
        |Memory out: 0.01 GB
        |Reduction: 92.7%
        >>> print(df_small.dtypes)
        |cats      category
        |true_ints    int8
        |floats     float32
        |dtype: object

        Notes
        -----
        Downcasting to float dtypes below 32-bits (np.float16, np.float8)
        is not supported.
        """

    def __init__(self, max_unique_pct=0.5, snake_case=False):
        self.max_unique_pct = max_unique_pct
        self.snake_case = snake_case

    def fit(self, df, float_cols=None):
        """ Identify dataframe and any float columns to be reduced.

        Parameters
        ----------
        df : pandas DataFrame
            The dataframe used as the basis for conversion.

        float_cols : list, optional, default=None
            A list of column names to be converted from np.float64 to
            np.float32.
        """
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"'{type(df).__name__}' object is not a pandas \
                    dataframe.")
        
        self.float_candidates = float_cols
        return self

    # Helper functions for .transform()
    def reduce_ints(self, df):
        int_cols = df.select_dtypes('integer').columns
        if len(int_cols) > 0:
            print("Starting integers.")
            mins = df[int_cols].min()
            unsigneds = mins.index[mins >= 0]
            df[unsigneds] = df[unsigneds].apply(pd.to_numeric,
                                                downcast='unsigned')
            signeds = mins.index[mins < 0]
            df[signeds] = df[signeds].apply(pd.to_numeric,
                                            downcast='signed')
        return df

    def reduce_floats(self, df, float_cols):
        print("Starting floats.")
        if not isinstance(float_cols, list):
            print(f"'{type(float_cols).__name__}' object is not a list,\
                    skipping floats.")
        else:
            true_float_cols = df.select_dtypes(np.float64).columns.tolist()
            non_float64s = [f for f in float_cols if f not in true_float_cols]
            if len(non_float64s) > 0:
                print("Skipping columns that are not np.float64")
            convertibles = [f for f in float_cols if f in true_float_cols]
            if len(convertibles) > 0:
                df[convertibles] = df[convertibles].astype(np.float32)
        return df

    def reduce_objs(self, df, max_pct):
        if (max_pct < 0.) or (max_pct > 1.):
            raise ValueError("max_unique_pct must be between 0 and 1")
        obj_cols = df.select_dtypes('object').columns
        if len(obj_cols) > 0:
            category_mask = df[obj_cols].nunique().values/len(df) <= max_pct
            cat_cols = obj_cols[category_mask]
            if len(cat_cols) > 0:
                print("Starting objects.")
                df[cat_cols] = df[cat_cols].astype('category')
        return df

    def snakify(self, df):
        col_list = []
        for c in df.columns.tolist():
            underscored = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', c)
            refined = re.sub('([a-z0-9])([A-Z])', r'\1_\2',
                             underscored).lower()
            col_list.append(refined)
        df.columns = col_list
        return df

    def transform(self, df):
        """ Convert dataframe columns to dtypes requiring lower memory.

        Parameters
        ----------
        df : pandas DataFrame
            The dataframe to be converted.
        """

        validation.check_is_fitted(self, 'float_candidates')

        memory_GB_in = df.memory_usage(deep=True).sum()/(1024**3)
        print(f"Memory in: {memory_GB_in:.2f} GB")

        df = self.reduce_ints(df)
        if self.float_candidates is not None:
            df = self.reduce_floats(df, self.float_candidates)
        df = self.reduce_objs(df, self.max_unique_pct)
        if self.snake_case:
            df = self.snakify(df)

        memory_GB_out = df.memory_usage(deep=True).sum()/(1024**3)
        print(f'Memory out: {memory_GB_out:.2f} GB',
              f'Reduction: {1 - memory_GB_out/memory_GB_in:.1%}',
              sep='\n')

        return df
