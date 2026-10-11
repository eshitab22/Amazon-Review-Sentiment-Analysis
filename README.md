# Amazon Review Sentiment Analysis

## Overview
This project explores sentiment classification of Amazon reviews using **TF-IDF vectorization and Logistic Regression**. It investigates the effect of different class-weight settings on model performance when dealing with imbalanced data.

## Objectives
- Analyze and preprocess Amazon review data.
- Convert text reviews into numerical features using TF-IDF.
- Train Logistic Regression models with different class-weight settings.
- Compare model performance to understand the impact of class imbalance.

## Methodology
1. Data loading and preprocessing
2. Text feature extraction using TF-IDF
3. Train-test splitting with stratification
4. Logistic Regression model training
5. Comparison of class-weight ratios: **1:3, 1:5, and 1:8**
6. Model evaluation using performance metrics and confusion matrices

## Tools & Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Matplotlib and Seaborn

## Repository Contents
- `Amazon_Reviews_Logistic_Regression_Weight_Comparison.ipynb` — Jupyter Notebook containing the analysis, model training, and evaluation.

## How to Run
1. Clone or download this repository.
2. Install the required Python libraries.
3. Open the notebook in Jupyter Notebook or Google Colab.
4. Ensure the dataset is available at the path specified in the notebook.
5. Run the cells sequentially.

## Key Learning
This project demonstrates how class weighting can be investigated as a strategy for handling imbalanced classification problems and why model evaluation should consider more than overall accuracy.

---

**Author:** Eshita Basak
