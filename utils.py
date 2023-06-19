import copy
import numpy as np
import argparse as args

class Remove_correlateds():
  def __init__(self, threshold):
    self.threshold = threshold
    self.is_trained = False
  def fit(self, df_train):
    self.is_trained = True
    data = copy.deepcopy(df_train)
    # Create correlation matrix
    corr_matrix = data.corr()
    corr_matrix = corr_matrix.abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) #lower half is NaN
    # Find features with correlation greater than [threshold]
    self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
    data.drop(self.to_drop, axis=1, inplace=True)
    return data
  def transform(self, df_test):
    # Drop features 
    data = copy.deepcopy(df_test)
    data.drop(self.to_drop, axis=1, inplace=True)
    return data