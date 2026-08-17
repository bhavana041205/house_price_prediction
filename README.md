# 🏠 House Price Prediction

A basic Machine Learning project that analyzes **20,640 California housing records** and predicts house prices using features such as median income, house age, number of rooms, and population.

## 📖 Project Overview

The project follows a simple process:

1. **Analyze** the housing dataset using Pandas.
2. **Visualize** relationships between housing features and prices.
3. **Split** the data into training and testing sets.
4. **Train** a Linear Regression model to predict house prices.
5. **Evaluate** the model using MAE and R² score.

## 🔑 Key Features

* **Data Analysis:** Used Pandas and NumPy to load, inspect, and prepare the housing data.
* **Data Visualization:** Created histograms, scatter plots, and correlation heatmaps using Matplotlib and Seaborn.
* **Train-Test Split:** Divided the dataset into training and testing data to evaluate the model on unseen records.
* **Price Prediction:** Used Linear Regression to predict house prices based on different housing features.
* **Model Evaluation:** Used **Mean Absolute Error (MAE)** and **R² score** to measure prediction performance.

## 🛠️ Tech Stack

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

## 📂 Project Structure

```text
HousePricePrediction/
│
├── data/
│   └── california_housing.csv
│
├── House_Price_Prediction.ipynb
├── README.md
└── requirements.txt
```

## 🚀 How It Works

### 1. Load the Dataset

The California Housing dataset is loaded using Scikit-learn and converted into a Pandas DataFrame.

### 2. Analyze the Data

The dataset is checked for:

* Number of records
* Missing values
* Feature types
* Basic statistics
* Relationship between features and house prices

### 3. Visualize the Data

Graphs are created to understand how features such as median income, house age, and number of rooms are related to house prices.

### 4. Train the Model

The data is divided into:

* **80% Training Data**
* **20% Testing Data**

A Linear Regression model is trained using the training data.

### 5. Evaluate the Model

The model is evaluated using:

* **MAE:** Average difference between predicted and actual prices.
* **R² Score:** Shows how well the model explains changes in house prices.

## 📊 Expected Output

The project generates:

* Feature distribution plots
* Correlation heatmap
* Actual vs. predicted price plot
* MAE score
* R² score

## 💡 Simple Project Explanation

> “I used the California Housing dataset to predict house prices. I first analyzed and visualized the data, then divided it into training and testing sets. I trained a Linear Regression model using features such as income, house age, and number of rooms. Finally, I compared the predicted prices with the actual prices using MAE and R² score.”


