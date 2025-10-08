import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

## Importing the dataset
data = pd.read_csv('ClassicHit.csv')
data = data.dropna()

## Dropping irrelevant columns/rows
data = data.drop(columns=['Artist'])
data = data[~data.isin(['World', 'Today', 'Disco', 'Folk', 'Reggae', 'Funk', 'SKA', 'Gospel']).any(axis=1)]
# data = data[~data.isin(['World', 'Today', 'Disco', 'Folk', 'Reggae', 'Funk', 'SKA', 'Gospel', 'Rock', 'R&B', 'Punk', 'Metal', 'EDM', 'Country', 'Blues', 'Alt. Rock']).any(axis=1)]

## Get the K-fold training, test, and validation sets
train, test = train_test_split(data, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.25, random_state=42)

## Save the datasets to CSV files
train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False)
test.to_csv('test.csv', index=False)

x = data.drop(columns=['Genre', 'Year', 'Track', 'Time_Signature', 'Valence'])
x = x.apply(lambda x: np.log1p(x) if np.issubdtype(x.dtype, np.number) else x)
y = data['Genre']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": (
        RandomForestClassifier(class_weight='balanced', random_state=42),
        {
            "clf__n_estimators": [400],
            "clf__max_depth": [None],
            "clf__min_samples_split": [5],
            "clf__min_samples_leaf": [1],
            "clf__max_features": ["log2"],
            "clf__bootstrap": [True]
        }
    ),

    "Neural Network (MLP)": (
        MLPClassifier(max_iter=2000, random_state=42),
        {
            "clf__hidden_layer_sizes": [(50,)],
            "clf__activation": ["relu"],
            "clf__solver": ["adam"],
            "clf__alpha": [0.0001],
            "clf__learning_rate": ["constant"]
        }
    )
}

results = {}

for name, (clf, param_grid) in models.items():
    print(f"\n===== Training {name} =====")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=1,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
        verbose=2,
        random_state=42,
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nBest parameters for {name}: {random_search.best_params_}")
    print(f"Accuracy for {name}: {acc:.4f}")
    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")
    
    results[name] = acc

print("\n=== Summary of Results ===")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")