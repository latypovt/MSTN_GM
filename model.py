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

# create dataframes
ml_dataframe = pd.read_csv('stats/ml_dataframe.csv')
ml_dataframe = ml_dataframe.drop(columns=['id', 'eTIV'])
ml_dataframe['diagnosis'] = [1 if dx =="MS-TN" else 0 for dx in ml_dataframe['diagnosis']]

# set condition
condition = list(ml_dataframe["diagnosis"])
condition = np.array(condition)

# drop unnecessary columns
gm_data = ml_dataframe.drop(columns=['age', 'sex', 'diagnosis', 'duration_of_ms', 'duration_of_pain',
       'side_of_pain'])
gm_data = gm_data[np.random.default_rng(seed=42).permutation(gm_data.columns.values)]

# class for removing highly correlated features 
class Remove_correlateds():
  def __init__(self, threshold=0.9):
    self.threshold = threshold
    self.is_trained = False
  def fit(self, df_train):
    self.is_trained = True
    data = copy.deepcopy(df_train)
    # Create correlation matrix
    corr_matrix = data.corr()
    corr_matrix = corr_matrix.abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) #lower half is NaN
    # Find features with correlation greater than 0.95
    self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
    data.drop(self.to_drop, axis=1, inplace=True)
    return data
  def transform(self, df_test):
    # Drop features 
    data = copy.deepcopy(df_test)
    data.drop(self.to_drop, axis=1, inplace=True)
    return data
  
# stratify
model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", C=0.1))])
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rc = Remove_correlateds()
x = rc.fit(gm_data)
gm_cols = list(x.columns)
x = x.to_numpy()
y = condition

features = []
train_acc = []
test_acc = []
# test
for nest_index, test_index in kf.split(x,y):
    x_nest, x_test = x[nest_index], x[test_index]
    y_nest, y_test = y[nest_index], y[test_index]
    kf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    split = kf2.split(x_nest, y_nest)
    featureselector = SFS(model, k_features="parsimonious", forward=False, floating=False, scoring="accuracy", cv=list(split), n_jobs=18, verbose=0)
    featureselector.fit(x_nest, y_nest)
    print('Best accuracy score: %.4f' % featureselector.k_score_)
    print('Best subset (indices):', featureselector.k_feature_idx_)
    print('Number of features:', len(featureselector.k_feature_idx_))
    for i in featureselector.k_feature_idx_:
        features.append(gm_cols[i])
    x_nest = featureselector.transform(x_nest)
    x_test = featureselector.transform(x_test)
    model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", C=0.1))])
    model.fit(x_nest, y_nest)
    y_pred_train = model.predict(x_nest)
    y_pred_test = model.predict(x_test)
    acc_train = accuracy_score(y_nest, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    print("Train accuracy: %.4f" % acc_train)
    print("Test accuracy: %.4f" % acc_test)
    test_acc.append(acc_test)
    train_acc.append(acc_train)
feature_set = pd.DataFrame([])
feature_set['feature'] = features
print("Mean train accuracy: %.4f" % np.mean(train_acc))
print("Mean test accuracy: %.4f" % np.mean(test_acc))
feature_set = feature_set.feature.value_counts()
feature_set = feature_set.to_frame()
feature_set.to_csv('important_features.csv')


#TODO
# 1. follow main() function structure - modify this script so that it can be called from main()
# 2. Look at argparse library to make it easier to pass arguments to the script (options - threshold, k_features, path to data, C for SVM, kernel type for SVM, etc.)
# 3. Organize the custom classes in this file into a separate file (e.g. utils.py) and import them into this file