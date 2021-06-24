import pandas as pd


class CollectionToStringTransformer:
  
  """A class to encode features with multiple string entries into one single string
  
  df_orig is the original dataframe
  index_col:  string or a list,  is the column to group the df by
  col_to_transform: a string or a list, columns to combine into a string
  
  Initiate with df_orig in the init method and transform with index_col and col_to_transform
  """
  
  def __init__(self, df_orig):
      self.df = df_orig.copy()
      
  def get_collection_per_index(self, index_col, col_to_transform):
    
    if not isinstance(col_to_transform, str):
      index_col = list(index_col)
    
    if isinstance(col_to_transform, str):
      mapping = self.df.groupby(index_col)[col_to_transform].apply(set).reset_index(name=f'{col_to_transform}_set')
      mapping[f'{col_to_transform}_set'] = mapping[f'{col_to_transform}_set'].apply(sorted)
    
    if isinstance(col_to_transform, list):
      mapping = self.df.groupby(index_col)[col_to_transform].agg(set)
      mapping.columns = [f'{col}_set' for col in mapping.columns]
      
      for col in mapping.columns:
        mapping[col] = mapping[col].apply(sorted)
        
      mapping = mapping.reset_index()
      
    return mapping
    
  def transform(self, index_col, col_to_transform, drop_set = True, sep = ' '):
    
    if not isinstance(col_to_transform, str):
      index_col = list(index_col)
      
    if isinstance(col_to_transform, str):
      transformed = self.get_collection_per_index(index_col, col_to_transform)
      transformed[f"{col_to_transform}_string"] = transformed[f'{col_to_transform}_set'].apply(lambda x: f'{sep}'.join([str(i) for i in x]))
      
      if drop_set:
        transformed = transformed.drop([f'{col_to_transform}_set'], axis=1)
    
    
    if isinstance(col_to_transform, list):
      transformed = self.get_collection_per_index(index_col, col_to_transform)
      for col in col_to_transform:
        transformed[f"{col}_string"] = transformed[f'{col}_set'].apply(lambda x: ' '.join([str(i) for i in x]))
        
        if drop_set:
          transformed = transformed.drop([f'{col}_set'], axis=1)
        
    return transformed
