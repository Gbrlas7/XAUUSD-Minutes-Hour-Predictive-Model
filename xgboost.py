import xgboost as xgb
from sklearn.metrics import classification_report

# define Features (X) and Target (y)
# We drop both Target columns from X so the model cannot cheat
X = df.drop(columns=['Target_15m_Return', 'Target_Class'])
y = df['Target_Class']

# chronological train/test split (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# initialize XGBoost Classifier
# parameters:
# - n_estimators=100: The forest will have 100 decision trees.
# - learning_rate=0.05: Slower learning prevents the model from overreacting to noise.
# - max_depth=4: Keeps the trees shallow to prevent overfitting (memorizing the training data).
# Calculate the ratio of negative to positive samples
imbalance_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=imbalance_ratio, # Forces the model to care about minor class
    random_state=42
)

# Training XGBoost model
model.fit(X_train, y_train)

# make predictions on the unseen test data
y_pred = model.predict(X_test)

# evaluate model
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# feature importance (this covers the raw features and the engineered feature made above)
# this reveals which mathematical indicators actually drove the model's decisions
importance = pd.Series(model.feature_importances_, index=X.columns)
top_features = importance.sort_values(ascending=False).head(5)

print("\n--- Top 5 Most Important Features ---")
print(top_features)
