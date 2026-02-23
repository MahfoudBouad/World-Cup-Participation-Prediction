import numpy as np
import pandas as pd #implements dataframes, better data rep method
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from sklearn import tree #scikit tree learning method
from sklearn import preprocessing #scikit preprocessing library
from sklearn.model_selection import train_test_split # Import train_test_split function
import matplotlib.pyplot as plt

#################################################################
# read in the data from faces.csv, if I don't indicate what the line the header is on it uses line 0
file_path = r'C:\Users\mahfo\Documents\UMD-Spring 2024\CS 5232 (002) Machine Learning & Data Mining\Labs\Lab1\Data\MyData1.csv'
df = pd.read_csv(file_path)
dfcopy = df #gonna make a copy of the df
df.dtypes

###############################################################################
# Preparing the features (X) and target variable (y)
X = df.drop('PalyedQWorldCup22', axis=1)
y = df['PalyedQWorldCup22']
# One-hot encoding categorical variables
# Identifying categorical columns (excluding 'PalyedQWorldCup22' since it's the target)
categorical_cols = X.select_dtypes(include=['object']).columns

#######################################################################################################################
# Apply OneHotEncoder to the categorical columns - this returns a sparse matrix
encoder = OneHotEncoder(sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
# OneHotEncoder removes index; put it back
X_encoded.index = X.index
# Remove categorical columns from X
num_X = X.drop(categorical_cols, axis=1)
X_encoded.index
# Remove categorical columns from X
num_X = X.drop(categorical_cols, axis=1)
# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
categorical_cols

# Apply OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])
X_encoded

# Get the feature names after one-hot encoding
feature_names = encoder.get_feature_names_out(categorical_cols)

# Create a DataFrame for the encoded features
X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)
X_encoded_df.head()
# Drop the original categorical columns from X
X = X.drop(categorical_cols, axis=1)
# combin the original numerical features with the encoded categorical features
X_processed = pd.concat([X.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)
X_processed.dtypes


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


#######################################################
# Encode the target variable y using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

#######################################################################################################################

# Initialize and fit the Decision Tree Classifier
clf = DecisionTreeClassifier( random_state=42)
#too crowded to print
clf.fit(X_train, y_train)
#######################################################################################################################
#Plot Tree
# encode target
numerical_cols = df.select_dtypes(include=['number']).columns
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols.tolist() + encoded_feature_names.tolist()
numerical_cols
#**************************************************************
plt.figure(figsize=(20, 10))
plot_tree(clf, max_depth=3, filled=True, feature_names=all_feature_names, class_names=['Qualified', 'Not Qualified'], rounded=True, fontsize=12)
plt.show()
#######################################################################################################################
#Evaluate a descision tree model using 10 by 10-fold CV
from sklearn.model_selection import RepeatedKFold, cross_val_score

# create a RepeatedKFold object
rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
# perform the 10-fold cross-validation repeated 10 times
scores = cross_val_score(clf, X, y_encoded, cv=rkf)
print('Mean Accuracy: %.3f' % np.mean(scores))
print('Standard Deviation of Accuracy: %.3f' % np.std(scores))
# print the accuracy for each fold
for i, score in enumerate(scores, 1):
    print(f'Accuracy for fold {i}: {score:.3f}')

import os
print(os.getcwd())
