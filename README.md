# Customer Churn Risk

## Problem Statement

Customer churn — subscribers ending their relationship with a business — is one of the most expensive problems a subscription-based company can face: acquiring a new customer typically costs far more than retaining an existing one. Without a way to identify which customers are at risk *before* they leave, retention efforts (discounts, outreach, support) are applied blindly, wasting resources on customers who were never going to churn while missing the ones who were.

This project builds a baseline predictive model that estimates a customer's probability of churning from their account and usage data, so that retention efforts can be targeted rather than applied at random.

## Objective

Walk through a complete machine learning pipeline — from raw data to an interpretable model — using Logistic Regression as a baseline classifier, and to:

- Understand what drives churn risk in the dataset through exploratory data analysis.
- Build a clean, reproducible preprocessing pipeline (scaling, encoding, train/test split) that avoids data leakage.
- Train and evaluate a Logistic Regression model using standard classification metrics.
- Interpret the model's coefficients in terms of practical, business-relevant meaning.
- Honestly assess the model's limitations and outline concrete next steps for improvement.

## Features

The notebook (`churn_risk.ipynb`) follows a standard small-ML-project pipeline, end to end:

- **Data extraction** — loads the raw dataset from `data/`.
- **Exploratory Data Analysis** — missing values, target class balance, distribution plots (overall and split by churn class), a correlation heatmap, and a synthesized set of conclusions.
- **Data cleaning** — duplicate checks and validity checks (negative values, non-positive charges, out-of-range values, invalid target labels).
- **Feature engineering** — derives `friction_score` (`support_tickets + late_payments`), tested and kept because it correlates more strongly with churn than either raw component.
- **Feature selection** — correlation-based ranking of every feature (raw and engineered) against the target.
- **Leakage-safe preprocessing** — `train_test_split` runs *before* any fitting; `StandardScaler` is fit on the training set only and applied to the test set; categorical encoding (in place for future categorical columns) is done per split with test columns reindexed to match train.
- **Model training** — a Logistic Regression classifier, with its exact configuration (solver, regularization, etc.) explicitly recorded.
- **Model evaluation** — classification report, ROC-AUC, confusion matrix with a business-facing interpretation of true/false positives and negatives, train-vs-test comparison, and a learning curve to diagnose over/underfitting.
- **Coefficient interpretation** — fitted coefficients and their corresponding odds ratios, with a discussion of what each feature's direction and magnitude imply.

### Dataset

`data/dataset_churn_risk.csv` — 1,000 rows, one row per customer:

| Column | Description |
|---|---|
| `tenure_months` | How long the customer has been subscribed (months). |
| `monthly_charges` | The customer's monthly billed amount. |
| `support_tickets` | Number of support tickets filed. |
| `avg_session_minutes` | Average usage/session time. |
| `late_payments` | Count of late or missed payments. |
| `contract_months` | Length of the customer's contract term (months). |
| `target` | Label: 1 if the customer churned, 0 otherwise. |

## Technologies Used

- Python 3.10
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## Installation/Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/cristian0831/customer-churn-risk.git
   cd customer-churn-risk
   ```
2. (Recommended) create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required packages:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

## How to Run the Project

1. Launch Jupyter from the project root (so the notebook's relative path to `data/` resolves correctly):
   ```
   jupyter notebook churn_risk.ipynb
   ```

## Limitations

- All metrics come from a single 80/20 train/test split; there is no cross-validated confidence interval, so reported point estimates (e.g. 68% accuracy) carry unquantified sampling uncertainty.
- Logistic Regression is a linear model — it cannot capture feature interactions or non-linear effects. The learning curve shows the model is underfitting (bias-limited), not short on data.
- `friction_score` is a linear combination of `support_tickets` and `late_payments`, so those three coefficients are estimated on collinear inputs — their individual magnitudes should be read with some caution.
- No hyperparameter tuning was performed; regularization strength, penalty type, and solver were left at scikit-learn defaults.

## Future Improvements

- Try a model that can capture non-linear structure (e.g. Random Forest, XGBoost) to test whether the current ROC-AUC ceiling is a data-quality limit or a model-capacity one.
- Use k-fold cross-validation (or a bootstrap confidence interval) to quantify uncertainty around the reported metrics instead of relying on a single split.
- Validate the pipeline against real historical churn data before trusting any of its directional conclusions operationally.
