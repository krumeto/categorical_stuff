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


def remove_rows_with_many_nas(df, portion_nonna):
  thresh = portion_nonna*df.shape[1]
  return df.dropna(thresh=thresh)
  
  
def fillna_with_missing(df, subset):
  if not isinstance(subset, list):
    subset = [subset]
  
  print('Mean NAs Before filling with MISSING')
  print(df.loc[:, subset].isna().mean())
  df.loc[:, subset] = df.loc[:, subset].loc[:, subset].fillna('MISSING')
  
  return df

def get_query(query):
  sql_query =""
  with open(query, 'r') as fh:

      for line in fh:
        line = line.replace("\n",' ')
        sql_query= sql_query + line
        #print(line)      
  return sql_query

def most_common_token(s, n):
  if is_numeric_dtype(s):
    s = s.astype('string')
  long_string = s.str.cat(sep=' ')
  
  c = Counter(long_string.split(' '))
  del c['MISSING']
  
  for k in c.keys():
    c[k] = round(c[k]/len(s), 3)
    
  return c.most_common(n)

def get_zero_variance(df):
  
  features = ['giclnt_string', 'giatyp_string', 'gicfac_string',
       'rev_string', 'APC_string', 'appaynam_string', 'applan__string',
       'mue_string', 'rarc_string', 'ub4bx67_string', 'sum(trpadj)', 'dbantp',
       'dbaqtr', 'dbaday', 'dbctyp', 'ud4ubseq', 'unit', 'modifier', 'lgmid',
       'er_flag', 'ddstatus', 'ddrcause', 'rddesc', 'ddcode', 'pass_thru_flag',
       'lsat_flag']
  
  mapper = df.loc[:, features].nunique().loc[df.loc[:, features].nunique() == 1]
  
  filters = [col for col in mapper.index.values if col in features]
  
  if len(filters) > 0:
    df_results = df.loc[:, filters].mode().T.to_dict()[0]
    df_results = dict(sorted(df_results.items()))
    result = str(df_results)
    result = result.replace("{", "").replace("}", "")
    return result
  
  else:
    return np.nan


def get_clusters_summary(df,cluster_columns, cols_to_summarize):
  
  cluster_grouping = df.groupby(cluster_columns)
  
  common_features_per_cluster = cluster_grouping.apply(get_zero_variance).reset_index(name='cluster_common_features')
  
  for col in cols_to_summarize:
    result_ = cluster_grouping[col].apply(most_common_token,3).reset_index(name=f'top_most_frequent_{col}')
    
    common_features_per_cluster = common_features_per_cluster.merge(result_,
                                                                    how='left',
                                                                    on = cluster_columns
                                                                   )
  
  common_features_per_cluster.columns = [col.replace("_string", "") for col in common_features_per_cluster.columns]

  return common_features_per_cluster

def pull_account_number_and_fac_id(data, chunksize=9999):
  """Takes the cluster/novelty report as input. 
  Returns a pandas DF with Account dbmid, dbref1 and dbcfac for the dbmid in the report"""
  list_of_ids = data['dbmid'].astype('string').to_list()
  
  n_chunks = len(list_of_ids)//chunksize
  
  final_result = pd.DataFrame()
  for i in range(n_chunks+1):
    id_chunk = list_of_ids[i*chunksize: (1+i)*chunksize]
    
    query = f"""SELECT DISTINCT dbmid, dbref1, dbcfac FROM acedta.dbinfo WHERE dbmid in ({','.join(id_chunk)})"""
    
    interim_result = pd.read_sql(query, con=process())
    
    final_result = pd.concat([final_result, interim_result])

  return final_result.reset_index(drop=True)


def add_nthrive(data, chunksize=9999):
  """Takes the cluster/novelty report as input. 
  Returns a pandas DF with nThrive report for accounts with the same 
  concat(_FAC, '_', accountnumber) as combined_key in the report """
  list_of_ids = data['combined_key'].astype('string').to_list()
  
  n_chunks = len(list_of_ids)//chunksize
  
  final_result = pd.DataFrame()
  for i in range(n_chunks+1):
    id_chunk = list_of_ids[i*chunksize: (1+i)*chunksize]
    
    query = f"""SELECT *, concat(_FAC, '_', accountnumber) as combined_key FROM  datascience.nTrive_dataset WHERE concat(_FAC, '_', accountnumber) in ({",".join(f"'{w}'" for w in id_chunk)})""" 
    
    interim_result = pd.read_sql(query, con=process())
    
    final_result = pd.concat([final_result, interim_result])

  return final_result.reset_index(drop=True)

def mask_the_last_character(s):
  return s[:3].ljust(len(s), 'X')

def get_codes_from_string(s):
  return re.findall(r"'([\w\s]+)'", s)

def get_values_from_string(s):
  return re.findall("\d+\.\d+", s)

def zip_codes_values(s):
  if isinstance(s, str):
    return dict(zip(get_codes_from_string(s), get_values_from_string(s)))
  
  elif isinstance(s, list):
    return dict(s)

  else:
    return np.NaN
  
  
def get_rev_code_descriptions(dict_like_s, dictionary_to_use):
  outcome_dict = {}
  for k, v in dict_like_s.items():
    general_string = dictionary_to_use['Description2'].get(mask_the_last_character(k), 'No Info Available')
    specific_string = dictionary_to_use['Description'].get(k, 'No Info Available')
    full_string = k + ": " + general_string + ' - ' + specific_string
    
    outcome_dict[full_string] = v
    
  return outcome_dict
  
def add_description_to_main_data(s, dictionary_to_use):
  
  list_of_codes = s.split(' ')
  
  result_string = ''
  for code in list_of_codes:
    try:
      result_string = result_string + code + ': ' + dictionary_to_use[code] + ';'
    except:
      result_string = result_string + code + ': ' + 'No description found' + ';'
      
  return result_string

def get_patient_type_description(dict_like_s, dictionary_to_use):
  
  outcome_dict = {}
  for k, v in dict_like_s.items():
    patient_type_string = dictionary_to_use.get(k, 'No Info Available')
    full_string = "Patient Type " + k + " : " + patient_type_string
    
    outcome_dict[full_string] = v
    
  return outcome_dict

def get_plan_id_descriptions(dict_like_s, dictionary_to_use):
  
  outcome_dict = {}
  for k, v in dict_like_s.items():
    plan_name_string = dictionary_to_use.get(k, 'No Info Available')['plan name']
    facility_string = dictionary_to_use.get(k, 'No Info Available')['facility']
    full_string = k + ": " + 'Plan ' + plan_name_string + ' for facility ' + facility_string
    
    outcome_dict[full_string] = v
    
  return outcome_dict

def get_rarc_code_descriptions(dict_like_s, dictionary_to_use):
  
  outcome_dict = {}
  for k, v in dict_like_s.items():
    
    #Check if key in main dictionary
    if k in rarc_dict.keys():
      explanation = dictionary_to_use[k]['dim_remark_name']
      prevention_category = dictionary_to_use[k]['dim_prevention_category_name']
      is_denial = dictionary_to_use[k]['is_denial']
      is_foster = dictionary_to_use[k]['is_foster']
      is_preventable = dictionary_to_use[k]['is_preventable']
      
      
      full_string = explanation + " Prevention Category: " + prevention_category + \
      ' Denial: ' + is_denial + ' Preventable: ' + is_preventable
    
      outcome_dict[full_string] = v
    
    else:
      outcome_dict[k + ": " "Unknown Code"] = v
  return outcome_dict




def get_zero_variance(df):
  mapper = df.nunique().loc[df.nunique() == 1]
  
  features = ['giclnt_string', 'giatyp_string', 'gicfac_string',
       'rev_string', 'APC_string', 'appaynam_string', 'applan__string',
       'mue_string', 'rarc_string', 'ub4bx67_string', 'sum(trpadj)', 'dbantp',
       'dbaqtr', 'dbaday', 'dbctyp', 'ud4ubseq', 'unit', 'modifier', 'lgmid',
       'er_flag', 'ddstatus', 'ddrcause', 'rddesc', 'ddcode', 'pass_thru_flag',
       'lsat_flag']
  
  filters = [col for col in mapper.index.values if col in features]
  
  if len(filters) > 0:
    df_results = df.loc[:, filters].mode().T.to_dict()[0]
    df_results = dict(sorted(df_results.items()))
    result = str(df_results)
    result = result.replace("{", "").replace("}", "")
    return result
  
  else:
    return np.nan


from pandas.api.types import is_numeric_dtype

def most_common_token(s, n):
  if is_numeric_dtype(s):
    s = s.astype('string')
  long_string = s.str.cat(sep=' ')
  
  c = Counter(long_string.split(' '))
  del c['MISSING']
  
  for k in c.keys():
    c[k] = round(c[k]/len(s), 3)
    
  return c.most_common(n)



def get_clusters_summary(df,cluster_columns, cols_to_summarize):
  
  cluster_grouping = df.groupby(cluster_columns)
  
  common_features_per_cluster = cluster_grouping.apply(get_zero_variance).reset_index(name='cluster_common_features')
  
  for col in cols_to_summarize:
    result_ = cluster_grouping[col].apply(most_common_token,3).reset_index(name=f'top_most_frequent_{col}')
    
    common_features_per_cluster = common_features_per_cluster.merge(result_,
                                                                    how='left',
                                                                    on = cluster_columns
                                                                   )
    
  common_features_per_cluster.columns = [col.replace("_string", "") for col in common_features_per_cluster.columns]
    
  return common_features_per_cluster