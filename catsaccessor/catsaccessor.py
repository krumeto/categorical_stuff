import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical, infer_dtype
from functools import reduce
import warnings
import weakref

from itertools import combinations
from scipy.stats import chi2_contingency
import numpy as np

from collections import Counter

@pd.api.extensions.register_dataframe_accessor("cats")
class CatsAccessor:
    """A class of useful categorical stuff to add to pandas
    """
    def __init__(self, pandas_obj):
        self._finalizer = weakref.finalize(self, self._cleanup)
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._categorical_columns = None
        

    def _cleanup(self):
        del self._obj

    def remove(self):
        self._finalizer()

    @staticmethod
    def _validate(obj):
        # verify this is a DataFrame
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")
    
    def _get_categorical_columns(self):
      result = [col for col in self._obj.columns if infer_dtype(self._obj[col]) in ['object', 'string', 'category', 'categorical']]
      self._categorical_columns = result
      return result

    
    def _cramers_corrected_stat(self, confusion_matrix, correction: bool) -> float:
      """Calculate the Cramer's V corrected stat for two variables.
      Function from pandas-profiling.github.io
      Args:
          confusion_matrix: Crosstab between two variables.
          correction: Should the correction be applied?

      Returns:
          The Cramer's V corrected stat for the two variables.
      """
      chi2 = chi2_contingency(confusion_matrix, correction=correction)[0]
      n = confusion_matrix.sum().sum()
      phi2 = chi2 / n
      r, k = confusion_matrix.shape

      # Deal with NaNs later on
      with np.errstate(divide="ignore", invalid="ignore"):
          phi2corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / (n - 1.0))
          rcorr = r - ((r - 1.0) ** 2.0) / (n - 1.0)
          kcorr = k - ((k - 1.0) ** 2.0) / (n - 1.0)
          corr = np.sqrt(phi2corr / min((kcorr - 1.0), (rcorr - 1.0)))
      return corr

    
    def corr(self, correction = True):
      self._get_categorical_columns()
      results = pd.DataFrame()
      combos = combinations(self._categorical_columns , 2)
      for combo in list(combos):
        print(combo)
        cat_matrix = pd.crosstab(self._obj[combo[0]], self._obj[combo[1]])
        corr_coef = self._cramers_corrected_stat(cat_matrix, correction)
        results_series = pd.Series([combo[0], combo[1], corr_coef])
        results = pd.concat([results, results_series], axis=1)
      
      if correction:
        test_type = 'corrected_cramers'
      else:
        test_type = 'cramers'
      results = results.T.reset_index(drop=True)
      results.columns = ['feature_1', 'feature_2', test_type]
      return results.sort_values(by=test_type, ascending=False)

    def propose_removal(self, threshold, method = 'naive'):

      if method == 'naive':
        matrix = self.corr()

        test_type = [col for col in matrix.columns if col not in ['feature_1', 'feature_2']][0]
        print(test_type)
        corr_feature_list = (matrix.
                             loc[matrix[test_type] > threshold][['feature_1', 'feature_2']].
                             values)
        print(corr_feature_list)
        deletion_list = []
        for pair in corr_feature_list:
          feature = pair[0]
          filter1 = matrix.feature_1 != feature
          filter2 = matrix.feature_2 != feature
          corr_zero = matrix[filter1 & filter2][test_type].mean()
          print(feature, corr_zero)

          feature = pair[1]
          filter1 = matrix.feature_1 != feature
          filter2 = matrix.feature_2 != feature
          corr_one = matrix[filter1 & filter2][test_type].mean()
          print(feature, corr_one)

          if corr_zero > corr_one:
            deletion_list.append(pair[0])
          else:
            deletion_list.append(pair[1])

        return deletion_list  


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