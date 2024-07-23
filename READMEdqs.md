
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

![Real Estate](https://miro.medium.com/v2/resize:fit:1000/1*1C3GnoY-FzhqzL0MzTlWyQ.gif)

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
   
    ```
python main.py
    ```

2. **🎓 Train Model**: Train the model using the cleaned dataset.
   
    ```
python main.py
    ```

3. **📊 Model Statistics**: Get statistics of the trained model.
   
    ```
python main.py
    ```

4. **🔍 Test Model**: Test the model with new input data.
   
    ```
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

Creator Atome
