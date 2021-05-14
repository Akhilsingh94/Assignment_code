import pickle as pkl

import numpy as np
import pandas as pd
# import sklearn packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

# Creating numpy arrays
num_sample = 1000
num_feat = 5
arr_feat = np.random.binomial(1, 0.55, size=[num_sample, num_feat])
arr_target = np.random.binomial(1, 0.32, size=[num_sample, 1])
arr_list = [arr_feat, arr_target]
# Saving list of array in .pkl file
with open('data.pkl', 'wb') as f:
    pkl.dump(arr_list, f)

# loading data
with open('data.pkl', 'rb') as f:
    arr_list_load = pkl.load(f)

featColNames = ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5']
feat_df = pd.DataFrame(arr_list_load[0], columns=featColNames)
target_df = pd.DataFrame(arr_list_load[1], columns=['target'])
# Creating dataframe
df = pd.concat([feat_df, target_df], axis=1)

"""As the data is theoretically created hence not performing any EDA"""
df.describe()

# Declare feature vector and target variable
X = df.drop(['target'], axis=1)
Y = df['target']

# Split into training and testing data
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1, random_state=50)
print("Training features shape: ", train_x.shape)
print("Testing features shape: ", test_x.shape)

# First create the base model to tune
rf = RandomForestClassifier()
# Cross validation set
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# create random search
random_search = dict(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap)
# search across 20 different combinations
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_search, n_iter=20,
                               cv=cv, verbose=2, random_state=41, n_jobs=-1)
# Fit the random search model
rf_random.fit(train_x, train_y)

# print(rf_random.best_params_)

# Create the parameter grid based on the results of random search 
grid_search = {'n_estimators': [100, 150, 200],
               'min_samples_split': [8, 10, 15],
               'min_samples_leaf': [1, 2, 3],
               'max_features': ['auto'],
               'max_depth': [50, 75],
               'bootstrap': [False]}

# Instantiate grid search
rf_grid = GridSearchCV(estimator=rf, param_grid=grid_search,
                       cv=cv, n_jobs=-1, verbose=2)
# fit grid
grid_model = rf_grid.fit(train_x, train_y)


# print(grid_model.best_params_)

# evaluating the model
def evaluate(model, test_features, test_labels):
    y_pred = model.predict(test_features)
    accuracy = accuracy_score(test_labels, y_pred)
    model_f1_score = f1_score(test_labels, y_pred)
    print(f" Model accuracy: {accuracy}")
    print(f"Model f1 score: {model_f1_score}")
    print(f"confusion_matrix: \n {confusion_matrix(test_labels, y_pred)}")


best_fr_model = grid_model.best_estimator_

evaluate(best_fr_model, test_x, test_y)
