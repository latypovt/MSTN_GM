# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse
from utils.utils import Remove_correlateds
from utils import confusion_matrix as plot_cm
from tqdm import tqdm

def main(): 
  #parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--threshold", type=float, default=0.9)
  parser.add_argument("--k_features", type=str, default="parsimonious")
  parser.add_argument("--path_to_data", type=str, default="stats/ml_dataframe.csv")
  parser.add_argument("--kernel", type=str, default="linear")
  parser.add_argument("--C", type=float, default=0.01)
  parser.add_argument("--n_splits", type=int, default=9)
  args = parser.parse_args()
  
  # create dataframes
  ml_dataframe = pd.read_csv(args.path_to_data)
  ml_dataframe = ml_dataframe.drop(columns=['id', 'eTIV'])
  ml_dataframe['diagnosis'] = [1 if dx =="MS-TN" else 0 for dx in ml_dataframe['diagnosis']]
  print(ml_dataframe.shape)
  # drop subjects with missing data
  ml_dataframe = ml_dataframe.dropna(axis=0)
  print(ml_dataframe.shape)

  # set condition
  condition = list(ml_dataframe["diagnosis"])
  condition = np.array(condition)

  # drop unnecessary columns, prepare data for stratification
  gm_data = ml_dataframe.drop(columns=['age', 'sex', 'diagnosis', 'duration_of_ms', 'duration_of_pain', 'side_of_pain', 'edss'])
  gm_data = gm_data[np.random.default_rng(seed=42).permutation(gm_data.columns.values)]


  # stratify
  
  kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
  rc = Remove_correlateds(threshold=args.threshold)
  x = rc.fit(gm_data)
  print(x.shape)
  gm_cols = list(x.columns)
  x = np.array(x)
  y = condition
  features = []
  train_acc = []
  test_acc = []
  model_proba = None
  y_true = None
  model_pred = None


  # test
  progress_bar = tqdm(enumerate(kf.split(x, y)), desc='Progress', total = kf.get_n_splits(x, y), ascii=" ▖▘▝▗▚▞█", colour="#42cbf5")
  for fold, (nest_index, test_index) in progress_bar:
    x_nest, x_test = x[nest_index], x[test_index]
    y_nest, y_test = y[nest_index], y[test_index]
    kf2 = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    split = kf2.split(x_nest, y_nest)
    model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel=args.kernel, C=args.C, probability=True))])
    featureselector = SFS(model, k_features=args.k_features, forward=False, floating=False, scoring="accuracy", cv=list(split), n_jobs=20, verbose=0)
    featureselector.fit(x_nest, y_nest)
    fold_features = []
    for i in featureselector.k_feature_idx_:
        features.append(gm_cols[i])
        fold_features.append(gm_cols[i])
    x_nest = featureselector.transform(x_nest)
    x_test = featureselector.transform(x_test)
    final_model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel=args.kernel, C=args.C, probability=True))])
    final_model.fit(x_nest, y_nest)
    model_feature_weights = final_model['svm'].coef_
    y_pred_train = final_model.predict(x_nest)
    y_pred_test = final_model.predict(x_test)
    acc_train = accuracy_score(y_nest, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    test_acc.append(acc_test)
    train_acc.append(acc_train)
   
   # confusion matrix
    if model_proba is None:
        model_proba = final_model.predict_proba(x_test)
        y_true = y_test
        model_pred = y_pred_test
    else:
        model_proba = np.append(model_proba, final_model.predict_proba(x_test), axis=0)
        y_true = np.append(y_true, y_test, axis=0)
        model_pred = np.append(model_pred, y_pred_test, axis=0)

    #progress bar
    progress_bar.set_postfix({'Train': np.mean(train_acc), 'Test': np.mean(test_acc)})

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

  # save model probe
  cm = plot_cm.confusion_matrix(y_true, model_pred, normalize='true')
  plot_cm.plot_confusion_matrix(cm, class_names=['MS', 'MS-TN'], savefig='out/cm.png')
  plot_cm.plot_roc_auc(y_true, model_proba, class_names=['MS', 'MS-TN'], colors=['#B1C8E7', '#E6BA97'], savefig='out/roc.png')

if __name__ == "__main__":
  main()  
  

