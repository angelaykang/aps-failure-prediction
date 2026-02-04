# APS Failure Prediction with Tree-Based Methods

A machine learning project that predicts Air Pressure System (APS) failures in Scania trucks using tree-based classification methods. This project addresses the critical challenge of class imbalance in failure prediction scenarios.

## Overview

The Air Pressure System (APS) is critical for truck operations, generating pressurized air for essential functions like braking and gear changes. This project implements predictive models to identify component failures before they cause breakdowns, helping prevent costly maintenance issues and improve vehicle reliability.

## Key Features

- **Handles Highly Imbalanced Data**: The dataset has only 1.67% positive class (failures), requiring specialized techniques
- **Multiple ML Approaches**: Implements Random Forest and XGBoost classifiers
- **Class Imbalance Techniques**: 
  - SMOTE (Synthetic Minority Oversampling Technique) for synthetic oversampling
  - Balanced class weights in Random Forest
- **Feature Engineering**: 
  - Missing value imputation using median strategy
  - Feature selection using coefficient of variation
  - Comprehensive exploratory data analysis
- **Model Evaluation**: 
  - ROC curves and AUC metrics
  - Confusion matrices
  - Cross-validation with proper SMOTE pipeline implementation
  - Out-of-bag (OOB) error analysis

## Dataset

The dataset consists of:
- **Training set**: 60,000 examples (59,000 negative, 1,000 positive)
- **Test set**: 16,000 examples
- **Features**: 170 anonymized attributes from truck sensor data
- **Target**: Binary classification (negative=0, positive=1)

**Dataset Source**: APS Failure and Operational Data for Scania Trucks
- Original dataset from Scania CV AB
- Used in the Industrial Challenge 2016 at IDA (International Symposium on Intelligent Data Analysis)
- Licensed under GNU General Public License v3

## Getting Started

### Prerequisites

- Python 3.7+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/angelaykang/aps-failure-prediction.git
cd aps-failure-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Place the dataset files in the `data/` directory:
   - `aps_failure_training_set.csv`
   - `aps_failure_test_set.csv`
   - `aps_failure_description.txt`

2. Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/aps_failure_prediction.ipynb
```

## Project Structure

```
aps-failure-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── aps_failure_training_set.csv
│   ├── aps_failure_test_set.csv
│   └── aps_failure_description.txt
└── notebooks/
    └── aps_failure_prediction.ipynb
```

## Methodology

### 1. Data Preprocessing
- **Missing Value Imputation**: Median imputation (robust to outliers)
- **Feature Selection**: Coefficient of variation analysis to identify most variable features
- **Data Visualization**: Correlation matrices, scatter plots, and box plots

### 2. Model Training

#### Random Forest
- Baseline model without class balancing
- Balanced model with `class_weight='balanced'`
- Out-of-bag (OOB) score evaluation

#### XGBoost
- Hyperparameter tuning for L1 regularization (alpha)
- Grid search with 5-fold cross-validation
- Uncompensated and SMOTE-compensated versions

### 3. Class Imbalance Handling

**SMOTE Implementation**:
- Properly integrated using `imblearn.pipeline` to ensure SMOTE is applied only to training folds during cross-validation
- Prevents data leakage and provides unbiased performance estimates

## Results

### Best Model Performance
- **Model**: XGBoost with SMOTE
- **Test Accuracy**: 99.31%
- **AUC**: 0.9946
- **Test Error**: 0.69%

### Model Comparison

| Model | Accuracy | AUC | False Negatives | False Positives |
|-------|----------|-----|-----------------|-----------------|
| Random Forest (Uncompensated) | 99.24% | 0.9928 | 106 | 15 |
| Random Forest (Balanced) | 98.91% | 0.9919 | 160 | 14 |
| XGBoost (Uncompensated) | 99.22% | 0.9922 | 98 | 26 |
| **XGBoost (SMOTE)** | **99.31%** | **0.9946** | **66** | **45** |

**Key Findings**:
- SMOTE with XGBoost achieves the best balance between accuracy and failure detection
- Significantly reduces false negatives (66 vs 98-160), which is critical for failure prediction
- Slight increase in false positives is acceptable given the cost structure (missing a failure is much more costly)

## Technologies Used

- **Python 3.7+**
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **XGBoost**: Gradient boosting framework
- **imbalanced-learn**: SMOTE implementation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib & seaborn**: Data visualization

## Key Insights

1. **Class Imbalance is Critical**: The 1.67% positive class rate requires specialized handling techniques
2. **SMOTE Effectiveness**: Synthetic oversampling significantly improves model performance for minority class detection
3. **Pipeline Importance**: Proper SMOTE integration in cross-validation prevents data leakage
4. **Model Selection**: XGBoost with SMOTE provides the best trade-off for failure detection scenarios

## Acknowledgments

- Scania CV AB for providing the APS Failure dataset
- The IDA 2016 Industrial Challenge organizers
- Contributors to scikit-learn, XGBoost, and imbalanced-learn libraries
