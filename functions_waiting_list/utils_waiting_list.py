from functools import wraps
import numpy as np
import datetime as dt

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical, infer_dtype, is_object_dtype, is_string_dtype

from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline

#TODO - create a simple class to dummify date columns
def dummify_date_cols(df):
  if 'giadmd' in df.columns:
    df['giadmd'] = pd.to_datetime(df['giadmd'], errors='coerce')
    df['giadmd_year'] = df['giadmd'].dt.year.astype('Int64').astype('object')
    df['giadmd_month'] = df['giadmd'].dt.month.astype('Int64').astype('object')
    df = df.drop('giadmd', axis=1)
  
  if 'girefs' in df.columns:
    df['girefs'] = pd.to_datetime(df['girefs'], errors='coerce')
    df['girefs_year'] = df['girefs'].dt.year.astype('Int64').astype('object')
    df['girefs_month'] = df['girefs'].dt.month.astype('Int64').astype('object')
    df = df.drop('girefs', axis=1)

  if 'gidscd' in df.columns:
    df['gidscd'] = pd.to_datetime(df['gidscd'], errors='coerce')
    df['gidscd_year'] = df['gidscd'].dt.year.astype('Int64').astype('object')
    df['gidscd_month'] = df['gidscd'].dt.month.astype('Int64').astype('object')
    df = df.drop('gidscd', axis=1)
  
  print("Shape after dummify:", df.shape)
    
  return df


def format_missings(df):
  for column in df.columns:
    if is_numeric_dtype(df[column]):
      fill_value = df[column].mean()
      df[column] = df[column].fillna(fill_value, downcast=False)
    elif is_object_dtype(df[column]) or is_string_dtype(df[column]):
      df[column] = df[column].fillna('MISSING', downcast=False)
  print("Shape after format_missing:", df.shape)
  return df

def remove_features_with_missing_values(df, na_thres):
  return df.loc[:, df.isna().mean() < na_thres]


def clean_floats(x):
  if pd.isnull(x):
    return x
  elif type(x) is float:
    return str(int(x))
  else: 
    return x
  
def clean_up_floats(df):
  for col in df.columns:
    if is_object_dtype(df[col]) or is_string_dtype(df[col]):
      df[col] = df[col].apply(clean_floats)
  print('Shape after clean_floats:', df.shape)
  return df


#Decorator to log information on functions 
def log_pipe_step(func):
    """Decorator to log information about functions.
    Use function.unwrapped to turn the decorator off.
    
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        print(f"Ran {func.__name__} DF shape={result.shape} took {time_taken}s")
        return result
    wrapper.unwrapped = func
    
    return wrapper


@log_pipe_step   
def rev_codes_one_hot(df, n_codes=50):
  """Takes a df and the n_codes, returns a one_hot df.
  
  Usage Example: df.pipe(rev_codes_one_hot, 10)
  
  """
  df_copy = df.copy()
#  single_code_map = df_copy.rev_codes.str.contains(';')
#  top_codes = df_copy.loc[~single_code_map].rev_codes.value_counts(normalize=True).nlargest(n_codes).index
  top_codes = ['300', '403', '320', '510', '402', '450', '420', '761', '981',
       'MISSING', '972', '921', '480', '352', '511', '483', '333', '610',
       '612', '943', '310', '740', '920', '430', '942', '401', '540', '351',
       '324', '456', '521', '440', '350', '301', '730', '311', '300LA', '964',
       '611', '987', '360', '361', '460', '731', '424', '510CL', '306', '413',
       '940', '948', '482', '985', '320RA', '305', '983', '922', '450ER',
       '434', '614', '780', '982', '410', '918', '636', '619', '469', '912',
       '250', '444', '420PT']
  
  
  for code in top_codes[:n_codes]:
    df_copy[f'rev_code_{code}'] = df_copy.rev_codes.str.contains(code).astype('int')
  
  df_copy = df_copy.drop('rev_codes', axis=1)
  return df_copy


def rev_codes_nmf(df, n_components=10):
  """Takes a df and the n_codes, returns a nmf df.
  
  Usage Example: df.pipe(rev_codes_nmf, 10)
  
  """
  df_copy = df.copy()
#  single_code_map = df_copy.rev_codes.str.contains(';')
#  top_codes = df_copy.loc[~single_code_map].rev_codes.value_counts(normalize=True).nlargest(60).index
  
  top_codes = ['300', '403', '320', '510', '402', '450', '420', '761', '981',
       'MISSING', '972', '921', '480', '352', '511', '483', '333', '610',
       '612', '943', '310', '740', '920', '430', '942', '401', '540', '351',
       '324', '456', '521', '440', '350', '301', '730', '311', '300LA', '964',
       '611', '987', '360', '361', '460', '731', '424', '510CL', '306', '413',
       '940', '948', '482', '985', '320RA', '305', '983', '922', '450ER',
       '434', '614', '780', '982', '410', '918', '636', '619', '469', '912',
       '250', '444', '420PT']
  
  codes_df = pd.DataFrame()
  for code in top_codes:
    codes_df[f'rev_codes_{code}'] = df_copy.rev_codes.str.contains(code).astype('int')
  print('Starting NMF')
  nmf = NMF(n_components=n_components)
  W = nmf.fit_transform(codes_df)
  
  col_names = [f"rev_component_{i}" for i in range(n_components)]
  for i, name in enumerate(col_names):
    df_copy[name] = W[:,i]
  
  df_copy = df_copy.drop('rev_codes', axis=1)
  return df_copy


def transform_diagnosis(df):
  """Transform the text diagnosis to features for classification. 
  The HashingVectorizer converts text to matrics, while the TfidfTransformer provides inverse
  document frequencies, resulting in a sparse matrix. Last, the SVD reduces dimensions to improve 
  the work of Tree-based models.
  
  Inspired by https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
  
  Usage: df.pipe(transform_diagnosis)
  
  """
  n_features = 10000
  n_components=25
  dataset = df.fddiagtx
  hasher = HashingVectorizer(n_features=n_features,
                              stop_words='english', 
                             alternate_sign=False,
                                   norm=None)
  
  vectorizer = make_pipeline(hasher, TfidfTransformer())
  sparse_matrix = vectorizer.fit_transform(dataset)
  
  svd = TruncatedSVD(n_components=n_components)
  regr = svd.fit_transform(sparse_matrix)

  col_names = [f"diag_component_{i}" for i in range(n_components)]
  for i, name in enumerate(col_names):
    df[name] = regr[:,i]
  
  df = df.drop('fddiagtx', axis=1)
  return df


def clean_floats(x):
  if pd.isnull(x):
    return x
  elif type(x) is float:
    return str(int(x))
  else: 
    return x
  
def clean_up_floats(df):
  for col in df.columns:
    if is_object_dtype(df[col]) or is_string_dtype(df[col]):
      df[col] = df[col].apply(clean_floats)
  print('Shape after clean_floats:', df.shape)
  return df