import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from utils.clean import clean
from utils.load import load_file
from utils.model import load_and_prepare_data, find_latest_model_path, save_model, save_model_info, handle_categorical_data, randomized_search, grid_search
import xgboost as xgb
xgb.set_config(verbosity=0)

def prepare_data(input_data, scaler, categorical_columns):
    df = pd.DataFrame([input_data])
    df = handle_categorical_data(df)
    
    for col in categorical_columns:
        if col not in df.columns:
            df[col] = 0

    df = df.reindex(columns=categorical_columns, fill_value=0)

    return scaler.transform(df)

def test_model(input_data, model_path, scaler, categorical_columns):
    model = joblib.load(model_path)
    X_test = prepare_data(input_data, scaler, categorical_columns)
    y_pred = model.predict(X_test)
    return y_pred

def model_stats(model_path, X_test, y_test):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model path: {model_path}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

def main():
    print("Choose an option:")
    print("1. Clean data set")
    print("2. Train model")
    print("3. Model statistics")
    print("4. Test model")
    choice = input("Enter 1, 2, 3, or 4: ")

    if choice == "1":
        raw_data_path = "./data/final_dataset.json"
        raw_df = load_file(raw_data_path)
        if raw_df is False:
            print("Failed to load raw data file.")
            return

        cleaned_df = clean(raw_df)
        
        cleaned_data_path = "./data/cleaned_data.csv"
        cleaned_df.to_csv(cleaned_data_path, index=False)
        print(f"Cleaned data saved to {cleaned_data_path}")

    elif choice == "2":
        cleaned_data_path = "./data/cleaned_data.csv"
        cleaned_df = load_file(cleaned_data_path)
        if cleaned_df is False:
            print("Failed to load cleaned data file.")
            return

        base_dir = "./data/models"
        X_train, X_test, y_train, y_test = load_and_prepare_data(cleaned_data_path)

        latest_model_path = find_latest_model_path(base_dir)
        if latest_model_path:
            model = joblib.load(latest_model_path)
            print(f"Existing model loaded from {latest_model_path}.")
        else:
            best_model, best_params = randomized_search(X_train, y_train, X_test, y_test)
            
            model = grid_search(X_train, y_train, best_params)
            
            new_segment_dir = save_model(model, base_dir)
            save_model_info(model, new_segment_dir, y_test, model.predict(X_test))

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"XGBoost Mean Absolute Error: {mae:.2f}")
        print(f"XGBoost R2 Score: {r2:.2f}")

    elif choice == "3":
        cleaned_data_path = "./data/cleaned_data.csv"
        df = load_file(cleaned_data_path)
        if df is False:
            print("Failed to load cleaned data file.")
            return

        y = df['Price']
        X = df.drop(columns=['Price'])
        X = handle_categorical_data(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

        model_path = find_latest_model_path("./data/models")
        if not model_path:
            print("No model found.")
            return

        model_stats(model_path, X_test, y_test)

    elif choice == "4":
        input_data = {
            "PostalCode": 4651,
            "BathroomCount": 2,
            "BedroomCount": 3,
            "ConstructionYear": 2011,
            "NumberOfFacades": 4,
            "PEB": 'B',
            "SurfaceOfPlot": 1044,
            "LivingArea": 200,
            "GardenArea": 948,
            "StateOfBuilding": 'Excellent',
            "SwimmingPool": 1,
            "Terrace": 1,
            "ToiletCount": 2,
            "RoomCount": 13
        }

        cleaned_data_path = "./data/cleaned_data.csv"
        df = load_file(cleaned_data_path)
        if df is False:
            print("Failed to load cleaned data file.")
            return

        X = df.drop(columns=['Price'])
        X = handle_categorical_data(X)
        categorical_columns = X.columns

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model_path = find_latest_model_path("./data/models")
        if not model_path:
            print("No model found.")
            return

        y_pred = test_model(input_data, model_path, scaler, categorical_columns)
        print(f"Predicted Price: {y_pred[0]:.2f}")

if __name__ == "__main__":
    main()
