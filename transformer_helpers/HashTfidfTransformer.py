from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline


class HashTfidfSvdTransformer(TransformerMixin, BaseEstimator):
  """A class to encode high cardinality categorical variables into a
  TfIdf Matrix.
  n_components: the number of max_features for the CountVectorizer
  hashing_kwargs: parameters to use for the CountVectorizer
  """
  
  def __init__(self, n_components, hashing_kwargs):
    self.n_components = n_components
    self.hashing_kwargs = hashing_kwargs
    
  
  def fit(self, df_orig, col_orig):
    self.df = df_orig.copy()
    
    self.hasher = CountVectorizer(
      max_features=self.n_components,
      **self.hashing_kwargs)
    
    self.vectorizer = make_pipeline(self.hasher, TfidfTransformer())
    self.vectorizer.fit(self.df[col_orig])

    return self
  
  def transform(self,X, col_to_encode):
    check_is_fitted(self, ['df', "n_components", 
                       #    'vectorizer', 'sparce_matrix',
                       #   'svd', 'regr'
                       ])
    self.dataset = X[col_to_encode]
    
    self.sparse_matrix = self.vectorizer.transform(self.dataset).toarray()
    print(self.sparse_matrix.shape)
    print(self.n_components)
  
#    self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
#    self.regr = self.svd.fit_transform(self.sparse_matrix)
    
    
    col_names = [f"{col_to_encode}_diag_component_{i}" for i in range(self.sparse_matrix.shape[1])]
    for i, name in enumerate(col_names):
      X[name] = self.sparse_matrix[:,i]
      
    return X
  