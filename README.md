
# 🏡 Immo Eliza Regression
<p align="center">
  <a href="https://www.youtube.com/embed/dQw4w9WgXcQ?autoplay=1">
      <img src="https://miro.medium.com/v2/resize:fit:1000/1*1C3GnoY-FzhqzL0MzTlWyQ.gif" alt="Click me !" width="600" />
  </a>
</p>

## 📑 Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Detailed Explanation](#detailed-explanation)
- [Author](#authors)
- [Project Structure](#-project-structure)

## Introduction

Welcome to the **Immo Eliza Regression** project! This repository contains all the necessary code to **clean data**, **train a model**, and **predict** real estate prices based on various property features. The project aims to predict real estate prices using machine learning models. The pipeline includes data cleaning, feature selection, model training, and price prediction. The code is structured into different modules for easy maintenance and scalability.

## 💻 Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone git@github.com:Atome1212/immo_eliza_regression.git
cd immo-eliza-regression
pip install -r requirements.txt
```

Download and place the CSV file into the `./data/csvfile.csv` directory:

[Download CSV](https://drive.google.com/file/d/1OUcp06JicyPdSeqTWDDuHNjrLxAOHFeR/view?usp=sharing)

## 🏃‍♂️ Usage

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

## 📖 Detailed Explanation

### Data Cleaning
   - Remove non-sale data
   - Strip whitespace
   - Remove duplicates
   - Select relevant features
   - Handle missing values using KNN imputation

### Model Training
   - Use `XGBoost` for training
   - Hyperparameter tuning with `RandomizedSearchCV`
   - Save the best model

### Prediction
   - Prepare new data
   - Load the saved model
   - Predict the price

## Author

- **👷‍♂️ [Atome1212](https://github.com/Atome1212)**: Data Engineer


## Project Structure

```
/immo_eliza_regression
├── README.md
├── data
│   ├── add_here_the_csv.txt
│   └── models
│       └── your_created_models_are_here.txt
├── main.py
├── requirements.txt
├── tree.py
└── utils
    ├── clean.py
    ├── load.py
    └── model.py
```

This tree provides an overview of the project structure, showing where each file and directory is located.

