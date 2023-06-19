# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import copy
import argparse
from utils import Remove_correlateds

def main(): 
  #parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--threshold", type=float, default=0.9)
  parser.add_argument("--k_features", type=str, default="parsimonious")
  parser.add_argument("--path_to_data", type=str, default="stats/ml_dataframe.csv")
  parser.add_argument("--kernel", type=str, default="linear")
  parser.add_argument("--C", type=float, default=0.1)
  args = parser.parse_args()
  
  # create dataframes
  ml_dataframe = pd.read_csv(args.path_to_data)
  ml_dataframe = ml_dataframe.drop(columns=['id', 'eTIV'])
  ml_dataframe['diagnosis'] = [1 if dx =="MS-TN" else 0 for dx in ml_dataframe['diagnosis']]

  # set condition
  condition = list(ml_dataframe["diagnosis"])
  condition = np.array(condition)

  # drop unnecessary columns
  gm_data = ml_dataframe.drop(columns=['age', 'sex', 'diagnosis', 'duration_of_ms', 'duration_of_pain', 'side_of_pain'])

  # stratify
  model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel=args.kernel, C=args.C))])
  kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  rc = Remove_correlateds(threshold=args.threshold)
  x = rc.fit(gm_data).to_numpy()
  y = condition

  # test
  for nest_index, test_index in kf.split(x,y):
      x_nest, x_test = x[nest_index], x[test_index]
      y_nest, y_test = y[nest_index], y[test_index]
      kf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
      split = kf2.split(x_nest, y_nest)
      featureselector = SFS(model, k_features=args.k_features, forward=False, floating=False, scoring="accuracy", cv=list(split), n_jobs=4, verbose=1)
      featureselector.fit(x_nest, y_nest)
      print('Best accuracy score: %.4f' % featureselector.k_score_)
      print('Best subset (indices):', featureselector.k_feature_idx_)
      print('Number of features:', len(featureselector.k_feature_idx_))
      print('Best subset (corresponding names):', list(featureselector.k_feature_names_))

if __name__ == "__main__":
  main()

#TODO
# 1. follow main() function structure - modify this script so that it can be called from main()
# 2. Look at argparse library to make it easier to pass arguments to the script (options - threshold, k_features, path to data, C for SVM, kernel type for SVM, etc.)
# 3. Organize the custom classes in this file into a separate file (e.g. utils.py) and import them into this file
#install MLxtent