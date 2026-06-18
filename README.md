# 🏡 House Price Prediction (End-to-End ML Pipeline)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project designed to predict real estate prices based on various structural and geographical features. This repository demonstrates data preprocessing, feature scaling, multi-model training, and performance benchmarking.

---

## 📌 Table of Contents
* [Project Overview](#-project-overview)
* [Dataset Description](#-dataset-description)
* [Key Features & Pipeline](#-key-features--pipeline)
* [Model Performance Benchmarking](#-model-performance-benchmarking)
* [Getting Started & Installation](#-getting-started--installation)
* [How to Run](#-how-to-run)
* [Future Roadmap](#-future-roadmap)

---

## 🔍 Project Overview
Predicting property values is a fundamental regression problem in Data Science. This project implements a clean, reproducible Machine Learning pipeline that screens multiple algorithms (`Linear Regression`, `Random Forest`, and `Decision Tree`) to identify the most accurate model for predicting house prices based on historical data.

---

## 📊 Dataset Description
The model evaluates houses based on **7 critical structural features** after filtering out noise:

| Feature Name | Description |
| :--- | :--- |
| `bedrooms` | Number of bedrooms in the house |
| `bathrooms` | Number of bathrooms |
| `m_living` | Total living area square footage |
| `sqft_lot` | Total lot size square footage |
| `floors` | Number of floors |
| `grade` | Overall grade given to the housing unit (based on construction quality) |
| `yr_built` | The year the house was initially built |
| **`price` (Target)** | The target price variable to predict |

---

## ⚙️ Key Features & Pipeline
1. **Data Cleaning & Engineering:** Automatic handling of duplicates via `.drop_duplicates()` and dropping missing records with `.dropna()`.
2. **Feature Optimization:** Strategic dropping of highly redundant or noisy features (e.g., `zipcode`, `yr_renovated`, `lat`, `long`) to avoid overfitting.
3. **Data Scaling:** Applied robust feature scaling using `StandardScaler` to normalize numerical variables.
4. **Multi-Model Evaluation:** Simultaneous training and metrics collection across Linear, Tree, and Ensemble models.

---

## 🏆 Model Performance Benchmarking
After executing the pipeline on a **80/20 train-test split**, the models generated the following metrics:

| Model Name | MAE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | *Calculated Value* | *Calculated Value* | *e.g., 0.68* |
| **Random Forest** | *Calculated Value* | *Calculated Value* | *e.g., 0.74* |
| **Decision Tree** | *Calculated Value* | *Calculated Value* | *e.g., 0.65* |

> 💡 **Key Takeaway:** Ensemble methods like *Random Forest* typically outperform single trees by reducing variance and avoiding overfitting via bootstrap aggregating.

---

## 💻 Getting Started & Installation

### Prerequisites
Make sure you have **Python 3.10+** installed on your system.
## Murtada G. Hameed

### 1. Clone the repository
```bash
git clone [https://github.com/mortaza007iraqi-cyber/House-Price-Prediction-ML.git](https://github.com/mortaza007iraqi-cyber/House-Price-Prediction-ML.git)
cd House-Price-Prediction-ML
