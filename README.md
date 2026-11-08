# AI Tool Usage Classification using Machine Learning

[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://ai-tool-usage-classification.streamlit.app/)

## 📌 Problem Statement

The objective of this project is to predict whether a college student is willing to pay for AI tool access based on their AI usage habits, academic background, awareness level, internet access, preferred AI tools, and other related factors.

This is a binary classification problem where:
* **0** = Not Willing to Pay
* **1** = Willing to Pay

The project compares the performance of five machine learning classification algorithms and evaluates them using multiple performance metrics.

---

## 📊 Dataset

* **Dataset Name:** AI Tool Usage by Indian College Students 2025
* **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/rakeshkapilavai/ai-tool-usage-by-indian-college-students-2025)
* **Number of Records:** 3,614
* **Target Variable:** `Willing_to_Pay_for_Access`

### Key Features
* Stream
* Year_of_Study
* AI_Tools_Used
* Daily_Usage_Hours
* Use_Cases
* Trust_in_AI_Tools
* Impact_on_Grades
* Do_Professors_Allow_Use
* Preferred_AI_Tool
* Awareness_Level
* State
* Device_Used
* Internet_Access

### Removed Features
* `Student_Name`
* `College_Name`  
*(These columns were removed because they do not contribute to classification and may introduce unnecessary noise.)*

---

## 🛠️ Approach

### 1. Data Preprocessing
* Loaded dataset using Pandas.
* Removed irrelevant and identifying columns.
* Handled and removed missing values.
* Applied Label Encoding to categorical features.
* Split dataset into **80% Training Data** and **20% Testing Data**.
* Applied `StandardScaler` for feature standardization.

### 2. Models Implemented
* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* Support Vector Machine (SVM)

### 3. Evaluation Metrics
The following metrics were calculated for all models:
* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## 📈 Results

### Performance on the Common Stratified Holdout

| ML Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.5519** | 0.5330 | **0.5507** | **0.6552** | **0.5984** | **0.1037** |
| **Decision Tree** | 0.5394 | 0.5388 | 0.5429 | 0.5328 | 0.5378 | 0.0787 |
| **K-Nearest Neighbors (KNN)** | 0.5366 | 0.5326 | 0.5403 | 0.5188 | 0.5294 | 0.0731 |
| **Gaussian Naive Bayes** | 0.5159 | 0.5186 | 0.5192 | 0.5000 | 0.5094 | 0.0318 |
| **Support Vector Machine (SVM)** | 0.5035 | **0.5414** | 0.5048 | 0.3607 | 0.4207 | 0.0072 |

---

## 💡 Observations

1. **Support Vector Machine (SVM):** Produced the highest AUC score (0.5414), but classification performance was weaker than Logistic Regression with lower Recall and F1 Score.
2. **Logistic Regression (Winner):** Achieved the highest Accuracy (55.19%), Recall (65.52%), F1 Score (59.84%), and MCC Score (0.1037), delivering the most balanced overall performance.
3. **Decision Tree:** Captured non-linear relationships effectively, producing competitive accuracy just below Logistic Regression.
4. **KNN:** Benefited from data normalization, with results comparable to the Decision Tree.
5. **Gaussian Naive Bayes:** The fastest and simplest model, though affected by feature independence assumptions.

---

## 🚀 Local Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/bhargav020103-07/AI-Tool-Usage-Classification.git](https://github.com/bhargav020103-07/AI-Tool-Usage-Classification.git)
cd AI-Tool-Usage-Classification
