import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

X = df.drop(columns=['Target_15m_Return', 'Target_Class'])
y = df['Target_Class']

# chronological train/test split (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# define the mathematical imbalance
imbalance_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# create a blank baseline model
base_model = xgb.XGBClassifier(
    scale_pos_weight=imbalance_ratio,
    random_state=42
)

# define the Grid (the settings we want to test)
# max_depth: How many consecutive IF/THEN rules the tree can make
# learning_rate: How aggressively it corrects its mistakes
# subsample: What % of the data it looks at per tree (prevents overfitting)
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'n_estimators': [100]
}

# initialize grid search
# cv=3 means it will test each combination 3 times to ensure it wasn't a fluke
# scoring='precision' forces it to strictly optimize for high-quality Buy signals
grid_search = GridSearchCV(
    estimator=base_model, 
    param_grid=param_grid, 
    scoring='precision', 
    cv=3, 
    verbose=1
)

# running the search
grid_search.fit(X_train, y_train)

# find the ultimate model
print(f"Best Settings Found: {grid_search.best_params_}")

best_model = grid_search.best_estimator_

# making predicition using the best model
y_pred = best_model.predict(X_test)

# evaluation of the best model
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))
