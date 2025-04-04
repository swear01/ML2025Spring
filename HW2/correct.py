import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the datasets
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# Define the useful columns for each day
useful_columns = ['wnohh_cmnty_cli', 'wbelief_masking_effective', 'wbelief_distancing_effective', 
                  'wcovid_vaccinated_friends', 'wlarge_event_indoors', 'wothers_masked_public', 
                  'wothers_distanced_public', 'wshop_indoors', 'wrestaurant_indoors', 'wworried_catch_covid']

# Prepare the training data
X_train = pd.DataFrame()
for day in ['day1', 'day2']:
    X_train = pd.concat([X_train, train_df[[f'{col}_{day}' for col in useful_columns]]], axis=1)

y_train = train_df['tested_positive_day3']

# Prepare the test data
X_test = pd.DataFrame()
for day in ['day1', 'day2']:
    X_test = pd.concat([X_test, test_df[[f'{col}_{day}' for col in useful_columns]]], axis=1)

# Handle missing values (if any)
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train_split, y_train_split)

# Best model
best_model = grid_search.best_estimator_

# Evaluate the model on the validation set
y_val_pred = best_model.predict(X_val_split)
mse = mean_squared_error(y_val_split, y_val_pred)
print(f'Validation MSE: {mse}')

# Make predictions on the test set
y_test_pred = best_model.predict(X_test)

# Prepare the submission file
submission_df = pd.DataFrame({
    'id': np.arange(len(y_test_pred)),
    'tested_positive_day3': y_test_pred
})

# Save the predictions
submission_df.to_csv('./submission.csv', index=False)

print("Submission file saved.")