import pandas as pd
import os
from .load import load_file
import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
import joblib
import xgboost
xgboost.set_config(verbosity=0)

def handle_categorical_data(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def load_and_prepare_data(file_path):
    df = load_file(file_path)
    y = df['Price']
    X = df.drop(columns=['Price'])
    X = handle_categorical_data(X)
    
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X = poly.fit_transform(X)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=69)

def find_latest_model_path(base_dir):
    segments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    segments.sort(reverse=True)
    for segment in segments:
        model_path = os.path.join(base_dir, segment, 'model.pkl')
        if os.path.exists(model_path):
            return model_path
    return None

def train_model(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [100, 300, 500, 700, 900],
        'learning_rate': [0.01, 0.03, 0.1, 0.15, 0.2],
        'max_depth': [3, 5, 7, 10, 12],
        'subsample': [0.5, 0.7, 0.9],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

    model = XGBRegressor(tree_method='hist', random_state=69)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=400,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=69
    )
    random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False, early_stopping_rounds=10)
    
    return random_search.best_estimator_

def save_model(model, base_dir):
    new_segment_dir = os.path.join(base_dir, f"model_segment_{len(os.listdir(base_dir)) + 1}")
    os.makedirs(new_segment_dir, exist_ok=True)
    joblib.dump(model, os.path.join(new_segment_dir, 'model.pkl'))
    print(f"Model saved at {new_segment_dir}.")
    return new_segment_dir

def save_model_info(model, new_segment_dir, y_test, y_pred):
    info_path = os.path.join(new_segment_dir, 'info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Model segment: {len(os.listdir(new_segment_dir))}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}\n")
        f.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}\n")
        f.write(f"Best Params: {model.get_params()}\n")
    print(f"Model information stored at '{info_path}'")
