# ChurnSense — AI  Churn Prediction Platform

ChurnSense is a Streamlit-based customer churn prediction application that integrates a trained scikit-learn machine learning pipeline to estimate the likelihood of customer churn.

The project covers the complete machine learning workflow: data cleaning, exploratory analysis, preprocessing, model comparison, hyperparameter tuning, final model selection, model serialization, and deployment through a Streamlit interface.

## 🚀 Live Demo

**Live App:** [ChurnSense — Live Demo](https://churnsense-churn-prediction.streamlit.app/)

---

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are more likely to leave can help retention teams take proactive action.

ChurnSense takes **19 customer attributes** and uses a trained classification pipeline to:

* Predict whether a customer is likely to churn
* Calculate churn probability
* Classify the customer into a risk level
* Display retention probability and model confidence
* Provide a rule-based business recommendation

The application is designed around the Telco Customer Churn dataset and uses a telecom-style customer feature set.

---

## 🧠 Machine Learning Workflow

The project follows these main steps:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Preparation
   ↓
Train/Test Split
   ↓
Multiple Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final Model Selection
   ↓
Model Serialization
   ↓
Streamlit Application
```

### Train/Test Split

The dataset was divided using:

* **80% training data**
* **20% test data**
* `random_state = 42`
* `stratify = y`

Stratification was used to preserve the class distribution between training and test sets.

---

## 🤖 Model Comparison

Six classification algorithms were evaluated:

1. Logistic Regression
2. SVM
3. Random Forest
4. XGBoost
5. KNN
6. Decision Tree

The models were compared using accuracy, churn precision, and churn recall.

| Model                   |   Accuracy | Precision (Yes) | Recall (Yes) |
| ----------------------- | ---------: | --------------: | -----------: |
| **Logistic Regression** | **80.38%** |      **64.85%** |   **57.22%** |
| SVM                     |     79.18% |          64.01% |       49.47% |
| Random Forest           |     78.75% |          63.07% |       48.40% |
| XGBoost                 |     76.90% |          57.10% |       52.67% |
| KNN                     |     76.12% |          54.80% |   **58.02%** |
| Decision Tree           |     72.28% |          48.01% |       51.60% |

Logistic Regression achieved the highest accuracy in the initial comparison and was selected for further optimization along with Random Forest.

---

## ⚙️ Hyperparameter Tuning

Logistic Regression and Random Forest were further optimized using hyperparameter search.

### Logistic Regression

Best parameters:

```text
C = 0.01
solver = liblinear
```

Best cross-validation score:

```text
80.48%
```

### Random Forest

Best parameters:

```text
n_estimators = 300
max_depth = 10
min_samples_split = 10
```

Best cross-validation score:

```text
80.39%
```

### Tuned Model Comparison

| Model                         |   Accuracy | Precision (Yes) | Recall (Yes) | F1-Score (Yes) |
| ----------------------------- | ---------: | --------------: | -----------: | -------------: |
| **Tuned Logistic Regression** | **79.67%** |         **65%** |      **51%** |        **57%** |
| Tuned Random Forest           |     78.89% |             63% |          51% |            56% |

Based on the final test-set results, the tuned **Logistic Regression pipeline** was selected as the final model.

---

## 🔧 Preprocessing

The final pipeline handles preprocessing before classification.

### Numerical Features

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

These numerical features are standardized using **StandardScaler**.

### Categorical Features

The categorical features are transformed using **OneHotEncoder**.

A **ColumnTransformer** applies the appropriate preprocessing to numerical and categorical columns.

The preprocessing and Logistic Regression model are combined into a single scikit-learn **Pipeline**.

This allows the application to pass raw customer information directly to the trained pipeline.

---

## 📊 Features

The application collects 19 customer attributes:

| Feature           | Type        |
| ----------------- | ----------- |
| Gender            | Categorical |
| Senior Citizen    | Binary      |
| Partner           | Categorical |
| Dependents        | Categorical |
| Tenure            | Numerical   |
| Phone Service     | Categorical |
| Multiple Lines    | Categorical |
| Internet Service  | Categorical |
| Online Security   | Categorical |
| Online Backup     | Categorical |
| Device Protection | Categorical |
| Tech Support      | Categorical |
| Streaming TV      | Categorical |
| Streaming Movies  | Categorical |
| Contract          | Categorical |
| Paperless Billing | Categorical |
| Payment Method    | Categorical |
| Monthly Charges   | Numerical   |
| Total Charges     | Numerical   |

---

## 🎯 Churn Risk Classification

The application converts the predicted churn probability into four risk bands:

| Risk        | Probability     |
| ----------- | --------------- |
| 🟢 Low      | Below 25%       |
| 🟡 Medium   | 25% – below 50% |
| 🟠 High     | 50% – below 75% |
| 🔴 Critical | 75% and above   |

These thresholds are rule-based and are currently hardcoded in the application.

---

## 💼 Business Recommendation

ChurnSense also provides a simple rule-based recommendation based on the predicted risk.

* **Critical:** Immediate retention intervention
* **High:** Proactive customer outreach
* **Medium:** Engagement and usage support
* **Low:** Standard engagement and upsell opportunities

There is also a specific recommendation for low-risk customers who are both new and on month-to-month contracts.

This recommendation component is **rule-based**, not a machine learning recommendation engine.

---

## 🖥️ Application

The Streamlit application provides:

* Customer information form
* Churn probability
* Retention probability
* Churn/Retain prediction
* Risk classification
* Model confidence
* Business recommendation
* Custom dark-themed interface

The UI combines native Streamlit components with custom HTML/CSS styling.

---

## 🛠️ Technology Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **HTML/CSS**
* **Jupyter Notebook**

---

## 📁 Project Structure

```text
churnsense/
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   ├── logo
│   └── favicon
│
├── styles/
│   └── main.css
│
├── app.py
├── customer_churn.csv
├── Customer_Churn.ipynb
├── customer_churn_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

### Important Files

**`Customer_Churn.ipynb`**
Contains the machine learning workflow including data analysis, preprocessing, model comparison, tuning, evaluation, and model development.

**`customer_churn_model.pkl`**
Serialized final Logistic Regression pipeline used by the Streamlit application.

**`app.py`**
Streamlit application responsible for collecting customer inputs, loading the trained pipeline, generating predictions, and displaying results.

**`styles/main.css`**
Custom styling for the Streamlit interface.

---

## ▶️ How to Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd churnsense
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📓 Dataset

This project uses the ** Telco Customer Churn dataset**, which contains customer demographic, service, contract, and billing information.

The target variable is:

```text
Churn
```

where customers are classified as either:

```text
Yes
No
```

---

## ⚠️ Limitations

* The application predicts one customer profile at a time.
* Risk thresholds are manually defined rather than optimized from business costs.
* The recommendation system is rule-based.
* There is no database, authentication, monitoring, or automated model retraining.
* Model performance depends on the underlying training dataset.
* The final model is a serialized pre-trained pipeline and is not retrained by the Streamlit application.

---

## 🔮 Future Improvements

Potential improvements include:

* Threshold optimization based on business costs
* Model calibration
* Explainable AI using SHAP or similar techniques
* Batch prediction for multiple customers
* Model monitoring
* Automated retraining
* Database integration
* Authentication and user management
* More advanced retention recommendations

---

## 👤 Project Note

The machine learning workflow in the notebook includes data preparation, exploratory analysis, model comparison, hyperparameter tuning, evaluation, and final model selection.

The Streamlit application integrates the resulting trained pipeline and provides a user-facing prediction interface.

AI assistance was used during the development and refinement of the application interface and implementation.

---

## 📄 License

This project is intended for educational and portfolio purposes.
