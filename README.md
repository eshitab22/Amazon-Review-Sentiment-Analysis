# Amazon Review Sentiment Analysis

## Overview

This project investigates sentiment classification of Amazon Echo product reviews using **TF-IDF vectorization and Logistic Regression**. It compares three class-weight settings to evaluate how class weighting affects model performance on an imbalanced dataset.

The objective is to improve the identification of negative reviews while maintaining strong classification performance on positive reviews.

## Objectives

- Analyze and preprocess Amazon review data.
- Convert textual reviews into numerical features using TF-IDF.
- Investigate class imbalance in the dataset.
- Compare Logistic Regression models with different class-weight settings.
- Evaluate model performance using multiple metrics and confusion matrices.

## Dataset

The analysis uses the following columns:

- `verified_reviews`: Text of the customer review.
- `feedback`: Sentiment label, where `0` represents negative sentiment and `1` represents positive sentiment.

### Dataset Summary

| Description | Count |
|---|---:|
| Original records | 3,150 |
| Records excluded due to empty or missing review text | 80 |
| Final records used | 3,070 |
| Negative reviews | 237 (7.72%) |
| Positive reviews | 2,833 (92.28%) |

The dataset exhibits substantial class imbalance, with positive reviews accounting for 92.28% of the observations.

## Methodology

### 1. Data Preprocessing
- Selected the review text and sentiment label.
- Removed records with missing or invalid target labels.
- Cleaned missing and empty review text.

### 2. Train-Test Split
- Used an 80:20 stratified train-test split.
- Fixed the random state at 42 for reproducibility.
- Training set: 2,456 reviews.
- Test set: 614 reviews.

### 3. TF-IDF Feature Extraction

Used `TfidfVectorizer` with:
- Lowercase text conversion.
- English stop-word removal.
- Unigrams and bigrams.
- Minimum document frequency of 2.
- Maximum of 50,000 features.
- Sublinear term-frequency scaling.

### 4. Logistic Regression and Class Weighting

Trained three Logistic Regression models using the same train-test split.

| Setting (positive:negative) | Negative class weight | Positive class weight |
|---|---:|---:|
| 1:3 | 3 | 1 |
| 1:5 | 5 | 1 |
| 1:8 | 8 | 1 |

Increasing the weight assigned to negative reviews penalizes misclassification of negative examples more heavily during training.

## Model Evaluation

The models were evaluated using accuracy, balanced accuracy, ROC-AUC, negative-class precision and recall, negative-class F1-score, and macro F1-score.

### Results

| Metric | 1:3 | 1:5 | 1:8 |
|---|---:|---:|---:|
| Accuracy | 94.46% | 94.95% | 95.44% |
| Balanced Accuracy | 64.81% | 74.83% | 81.92% |
| ROC-AUC | 0.9680 | 0.9690 | 0.9697 |
| Negative Precision | 93.33% | 75.00% | 72.09% |
| Negative Recall | 29.79% | 51.06% | 65.96% |
| Negative F1-score | 0.4516 | 0.6076 | 0.6889 |
| Macro F1-score | 0.7112 | 0.7903 | 0.8321 |

*Results are from the notebook's fixed stratified test split.*

### Key Findings

- **The 1:8 setting achieved the highest accuracy (95.44%), balanced accuracy (81.92%), and macro F1-score (0.8321)** among the three tested settings.
- Negative-class recall increased from 29.79% to 65.96% as the negative-class weight increased from 3 to 8.
- Negative-class precision decreased from 93.33% to 72.09%, illustrating a trade-off between identifying more negative reviews and incorrectly flagging positive reviews.
- ROC-AUC remained high across all three settings, ranging from 0.9680 to 0.9697.

Overall, the 1:8 setting provided the strongest balance between the two classes among the configurations tested, based on balanced accuracy and macro F1-score. The preferred setting ultimately depends on the relative costs of missed negative reviews and false negative-sentiment predictions.

## Evaluation Techniques

- Confusion matrices
- Classification reports
- Accuracy and balanced accuracy
- Precision, recall, and F1-score
- ROC-AUC

## Tools and Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Matplotlib
- Seaborn
- Google Colab / Jupyter Notebook

## Repository Contents

- `Amazon_Echo_Review_Sentiment__Classifier.ipynb` — notebook containing data preprocessing, class imbalance analysis, model training, evaluation, and conclusions.

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Ensure that the dataset is available as `amazon_reviews.csv` in the working directory, or update `DATA_PATH` in the notebook.
4. Open the notebook in Google Colab or Jupyter Notebook.
5. Run the cells sequentially to reproduce the analysis.

### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Limitations and Future Improvements

- The comparison uses a single fixed train-test split.
- Stratified cross-validation could provide a more robust estimate of performance.
- Additional classification algorithms and text representations could be evaluated.
- Decision thresholds could be tuned based on the relative costs of different classification errors.

## Author

**Eshita Basak**

GitHub: [eshitab22](https://github.com/eshitab22)
