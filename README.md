# Nutrifacts ML
This repository is the Capstone Project in [Bangkit Academy](https://grow.google/intl/id_id/bangkit/?tab=machine-learning) 2023. By applying machine learning techniques in Nutrifacts, we can construct a model that estimates food grade from nutritional information.

![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg) ![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg) 

This repository contains the code and documentation for a machine learning model that predicts food grade based on nutritional information. The model is designed to help individuals make more informed choices about their food intake.

## Overview

This project utilizes a linear Support Vector Machine (SVM) to predict food grade based on the following nutritional features:

- Calories
- Proteins
- Fat
- Carbohydrates

The model was trained on a dataset of food items with corresponding nutritional information and assigned food grades.

## Installation

1. Clone Repository
```sh
git clone https://github.com/dzakyadlh/nutrifacts-ml.git
```
2. Install Library
```sh
pip install -r requirements.txt
```
Library:
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- joblib

Example How to use Model:
```sh
import numpy as np
from joblib import load

# Load the model
model = load('svm_model_linear.joblib')

# Prepare input data
input_data = np.array([[150, 5, 8, 25]])

# Make a prediction
prediction = model.predict(input_data)

print(prediction)
```
Input data must be a NumPy array with 4 dimensions, with the following feature order:
- calories
- proteins
- fat
- carbohydrate

## Features

Dillinger is currently extended with the following plugins.
Instructions on how to use them in your own application are linked below.

| Features | Function |
| ------ | ------ |
| Numpy | Provides efficient numerical computations with multi-dimensional arrays and matrices. |
| Pandas | Offers high-performance data structures and tools for data analysis and manipulation. |
| Seaborn | A high-level Python visualization library built on Matplotlib, specializing in statistical plots. |
| Matplotlib | A comprehensive Python plotting library for creating static, animated, and interactive visualizations. |
| Scikit-learn | A powerful machine learning library with a wide range of algorithms for supervised and unsupervised learning. |
| Joblib | A Python library for efficient serialization and persistence of Python objects, often used for saving and loading machine learning models. |

## License

MIT
