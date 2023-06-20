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
from utils.utils import Remove_correlateds
from tqdm import tqdm

def main(): 
  #parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--threshold", type=float, default=0.9)
  parser.add_argument("--k_features", type=str, default="parsimonious")
  parser.add_argument("--path_to_data", type=str, default="stats/ml_dataframe.csv")
  parser.add_argument("--kernel", type=str, default="linear")
  parser.add_argument("--C", type=float, default=0.1)
  parser.add_argument("--n_splits", type=int, default=10)
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
  gm_data = gm_data[np.random.default_rng(seed=42).permutation(gm_data.columns.values)]


  # stratify
  model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel=args.kernel, C=args.C))])
  kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  rc = Remove_correlateds(threshold=args.threshold)
  x = rc.fit(gm_data)
  gm_cols = list(x.columns)
  x = np.array(x)
  y = condition
  features = []
  train_acc = []
  test_acc = []
  model_probe = None

  # test
  progress_bar = tqdm(enumerate(kf.split(x, y)), desc='Progress', total = kf.get_n_splits(x, y))
  for fold, (nest_index, test_index) in progress_bar:
    x_nest, x_test = x[nest_index], x[test_index]
    y_nest, y_test = y[nest_index], y[test_index]
    kf2 = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    split = kf2.split(x_nest, y_nest)
    featureselector = SFS(model, k_features=args.k_features, forward=False, floating=False, scoring="accuracy", cv=list(split), n_jobs=4, verbose=0)
    featureselector.fit(x_nest, y_nest)
    #print('Best accuracy score: %.4f' % featureselector.k_score_)
    #print('Best subset (indices):', featureselector.k_feature_idx_)
    #print('Number of features:', len(featureselector.k_feature_idx_))
    fold_features = []
    for i in featureselector.k_feature_idx_:
        features.append(gm_cols[i])
        fold_features.append(gm_cols[i])
    x_nest = featureselector.transform(x_nest)
    x_test = featureselector.transform(x_test)
    model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel=args.kernel, C=args.C))])
    model.fit(x_nest, y_nest)
    model_feature_weights = model['svm'].coef_

    y_pred_train = model.predict(x_nest)
    y_pred_test = model.predict(x_test)
    acc_train = accuracy_score(y_nest, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    test_acc.append(acc_test)
    train_acc.append(acc_train)
    # confusion matrix
    if model_probe is None:
        model_probe = model.predict_proba(x_test)
    else:
        model_probe = np.append(model_probe, model.predict_proba(x_test), axis=0)

    progress_bar.set_postfix({'Train': acc_train, 'Test': acc_test})

    # save fold_features as csv
    feature_weights = pd.DataFrame(model_feature_weights, columns=fold_features)
    feature_weights.to_csv('out/{}_feature_weights.csv'.format(fold))

  feature_set = pd.DataFrame([])
  feature_set['feature'] = features
  # overall results
  print("Mean train accuracy: %.4f" % np.mean(train_acc))
  print("Mean test accuracy: %.4f" % np.mean(test_acc))
  feature_set = feature_set.feature.value_counts()
  feature_set = feature_set.to_frame()
  feature_set.to_csv('out/important_features.csv')


if __name__ == "__main__":
  main()  
  
#TODO
# 0. Figuring out what is going on with tsne 
# 1. Add feature weights to the model script - save weights to a csv file for each fold

# 2. Make a jupyter notebook that does the data visualization of the feature weights
# 3. ROC curve for the model and confusion matrix - on the model script

