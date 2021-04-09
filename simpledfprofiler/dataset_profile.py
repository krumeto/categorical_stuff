import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_categorical, infer_dtype

def dataset_profile(data: pd.DataFrame):
    """A simple function to get you a simple dataset variables overview

    Args:
        data (pd.DataFrame): the dataset to be profiled

    Returns:
        pd.DataFrame: containing the report
    """

    report = {}
    for col in data.columns:
        col_dict = {}
        col_dict['feature_name'] = col
        col_dict['inferred_type'] = infer_dtype(data[col])
        col_dict['current_type'] = data[col].dtype
        col_dict['missing_values_sum'] = data[col].isna().sum()
        col_dict['missing_values_perc'] = data[col].isna().mean()
        if infer_dtype(data[col]) in ["string", 'categorical', 'mixed']:
            col_dict['n_unique_values'] = data[col].nunique()
            col_dict['biggest_category'] = data[col].value_counts(normalize=True).nlargest(1).index[0]
            col_dict['biggest_category_perc'] = data[col].value_counts(normalize=True, dropna=False).nlargest(1).values[0]
            col_dict['smallest_category'] = data[col].value_counts(normalize=True).nsmallest(1).index[0]
            col_dict['smallest_category_perc'] = data[col].value_counts(normalize=True, dropna=False).nsmallest(1).values[0]
            
        else:
            col_dict['n_unique_values'] = np.nan
            col_dict['biggest_category'] = np.nan
            col_dict['biggest_category_perc'] = np.nan
            col_dict['smallest_category'] = np.nan
            col_dict['smallest_category_perc'] = np.nan
        
        if infer_dtype(data[col]) in ['floating', 'integer', 'mixed-integer', 'mixed-integer-float', 'decimal']:
            col_dict['mean'] = pd.to_numeric(data[col], errors='coerce').mean()
            col_dict['std'] =  pd.to_numeric(data[col], errors='coerce').std()
            col_dict['min'] =  pd.to_numeric(data[col], errors='coerce').min()
            col_dict['max'] =  pd.to_numeric(data[col], errors='coerce').max()
        
        else:
            col_dict['mean'] = np.nan
            col_dict['std'] = np.nan
            col_dict['min'] = np.nan
            col_dict['max'] = np.nan
            
        report[col] = col_dict
    
    results = pd.DataFrame.from_dict(report).T
    results['duplicates'] = results.drop('feature_name', axis=1).duplicated(keep=False)
    results['categorical_duplicate'] = results[['n_unique_values', 'biggest_category_perc', 'smallest_category_perc', 'smallest_category']].dropna().duplicated(subset=['n_unique_values', 'biggest_category_perc', 'smallest_category_perc', 'smallest_category'],keep=False)
    results['numerical_duplicate'] = results[['mean', 'std', 'min', 'max']].dropna().duplicated(subset=['mean', 'std', 'min', 'max'],keep=False)
    
    return results