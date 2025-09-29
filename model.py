import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

## Importing the dataset
data = pd.read_csv('ClassicHit.csv')
#print(data.head())

## Checking for missing values
##print(data.isnull().sum())

## Dropping rows with missing values
data = data.dropna()

## Dropping irrelevant columns/rows
data = data.drop(columns=['Artist'])
data = data[~data.isin(['World', 'Today']).any(axis=1)]

#print list of genres and count of unique genres
print(data['Genre'].unique())
print(len(data['Genre'].unique()))

## Calculate the scatterplot matrix for each genre
# genres = data['Genre'].unique()
# for genre in genres:
#     genre_data = data[data['Genre'] == genre]

## Get the K-fold training, test, and validation sets
train, test = train_test_split(data, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.25, random_state=42)
print(train.shape, val.shape, test.shape)
print(train.head())
print(val.head())
print(test.head())

## Save the datasets to CSV files
train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False)
test.to_csv('test.csv', index=False)
# print(data.describe())
#print(data.info())

##Make a machine learning model decision tree classifier with the genres as the target variable features as the characteristics used to determine likely genre
print("LOOK HERE")
print(data.head())
X = data.drop(columns=['Genre', 'Year', 'Track'])
y = data['Genre']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

# ---- Feature Scaling ----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- Logistic Regression Model ----
# clf = LogisticRegression(multi_class='multinomial',
#     solver='lbfgs', 
#     max_iter=3000, 
#     random_state=42
#     )
# clf.fit(X_train, y_train)

# ---- Evaluate ----
# accuracy = clf.score(X_test, y_test)
# print(f"Logistic Regression Accuracy: {accuracy:.4f}")

## Tune hyperparameters using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [1000, 2000, 3000]
}
grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best parameters: {grid.best_params_}")
best_clf = grid.best_estimator_
best_accuracy = best_clf.score(X_test, y_test)
print(f"Tuned Logistic Regression Accuracy: {best_accuracy:.4f}")
