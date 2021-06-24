from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline



class SubstringOnehotEncoder(TransformerMixin, BaseEstimator):
  """A class to encode just a subset of top codes as one hots. The class is able to
  search for a match within a string.
  
  For example if a list_of_strings is ['Boston', 'Chicago'] both 'Boston Celtics' and
  'Chicago Bulls' are going to be encoded.
  
  """
  
  def __init__(self, drop_original = False, method='onehot', n_components = None):
    self.drop_original = drop_original
    self.method = method
    self.n_components = n_components
    
    if self.n_components is None and self.method == 'nmf':
      raise ValueError("Please define n_components for method nmf")
    
  def fit(self, df):
    self.df = df.copy()
    return self

  def transform(self, column , list_of_strings):
    allowed_methods = ['onehot', 'nmf']
    if self.method not in allowed_methods:
      raise ValueError(f"Please choose one of the following methods: {allowed_methods}")
    
    
    
    if self.method == 'onehot':
      check_is_fitted(self, ['df'])
      transformed = self.df
      
      for substring in list_of_strings:
        transformed[f"{column}_{substring}"] = transformed[column].str.contains(str(substring), case=False, na=0).astype('Int64')
    
    if self.method == 'nmf':
      print('NMF method does not retain state. Please beware in production.')
      check_is_fitted(self, ['df', 'n_components'])

      codes_df = pd.DataFrame()
      transformed = self.df
      
      for substring in list_of_strings:
        codes_df[f"{substring}"] = self.df[column].str.contains(str(substring), case=False, na=0).astype('Int64')
          
      print('Starting NMF')
      nmf = NMF(n_components=self.n_components)
      W = nmf.fit_transform(codes_df)
  
      col_names = [f"{column}_component_{i}" for i in range(self.n_components)]
      for i, name in enumerate(col_names):
        transformed[name] = W[:,i]


    
    if self.drop_original:
      transformed = transformed.drop(column, axis=1)
      
    return transformed
  
  
