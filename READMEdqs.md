
# 🏡 Immo Eliza Regression 🏡

Welcome to the Immo Eliza Regression project! This repository contains all the necessary code to clean data, train a model, and predict real estate prices based on various property features. 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict real estate prices using machine learning models. The pipeline includes data cleaning, feature selection, model training, and price prediction. The code is structured into different modules for easy maintenance and scalability.

![Real Estate](https://images.squarespace-cdn.com/content/v1/5feb53185d3dab691b47361b/1609930650139-9NRI63XUJ29Y7E9LEA9G/12eca-machine-learning.gif)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/immo-eliza-regression.git
cd immo-eliza-regression
pip install -r requirements.txt
```

## Usage

Follow these steps to clean data, train the model, and make predictions:

1. **🧹 Clean Data**: Load and clean the raw dataset.
    ```bash
    python main.py
    ```

2. **🎓 Train Model**: Train the model using the cleaned dataset.
    ```bash
    python main.py
    ```

3. **📊 Model Statistics**: Get statistics of the trained model.
    ```bash
    python main.py
    ```

4. **🔍 Test Model**: Test the model with new input data.
    ```bash
    python main.py
    ```

### Detailed Explanation

- **Data Cleaning**: 
    - Remove non-sale data
    - Strip whitespace
    - Remove duplicates
    - Select relevant features
    - Handle missing values using KNN imputation

- **Model Training**:
    - Use `XGBoost` for training
    - Hyperparameter tuning with `RandomizedSearchCV`
    - Save the best model

- **Prediction**:
    - Prepare new data
    - Load the saved model
    - Predict the price

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License

This project is licensed under the MIT License.

---

Happy coding! 😊🏡

### Creator: Atome ✨

## Code Overview

### Load Module

Handles loading of JSON and CSV files into DataFrames.

```python
import pandas as pd

def get_file_extension(file_path):
    return file_path.split("/")[-1].split(".")[-1]

def load_file(file_path: str):
    if get_file_extension(file_path) == "json":
        return pd.DataFrame(pd.read_json(file_path))
    elif get_file_extension(file_path) == "csv":
        return pd.DataFrame(pd.read_csv(file_path))
    else:
        return False
```

### Clean Module

Cleans and preprocesses the data.

```python
import pandas as pd
import os
from .load import load_file
from sklearn.impute import KNNImputer
import datetime
from sklearn.preprocessing import LabelEncoder

# Data cleaning functions here...
```

### Model Module

Trains and saves the model.

```python
import pandas as pd
import os
from .load import load_file
import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
import joblib
import xgboost
xgboost.set_config(verbosity=0)

# Model training functions here...
```

### Main Module

Main script to clean data, train model, and predict prices.

```python
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

# Main script functions here...
```

Feel free to explore the individual modules for more details on each function.

