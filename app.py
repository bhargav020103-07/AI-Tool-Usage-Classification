import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Tool Usage Classification",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Tool Usage Classification")
st.markdown(
"""
### Machine Learning Model Comparison

Default Dataset:
**AI Tool Usage by Indian College Students 2025**
"""
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Dataset Options")

uploaded_file = st.sidebar.file_uploader(
    "Upload Your Own Dataset (Optional)",
    type=["csv"]
)

# =====================================================
# DEFAULT OR CUSTOM DATASET
# =====================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.sidebar.success(
        "Using Uploaded Dataset"
    )

    custom_dataset = True

else:

    try:
        df = pd.read_csv("Students.csv")

        st.sidebar.success(
            "Using Default Dataset: Students.csv"
        )

        custom_dataset = False

    except Exception as e:

        st.error(f"Dataset Error: {e}")
        st.stop()

# =====================================================
# DATA PREVIEW
# =====================================================

st.subheader("Dataset Preview")

st.dataframe(df.head())

col1, col2 = st.columns(2)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])

# =====================================================
# CLEAN DATA
# =====================================================

drop_cols = [
    "Student_Name",
    "College_Name"
]

for col in drop_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

df.dropna(inplace=True)

# =====================================================
# TARGET COLUMN
# =====================================================

if custom_dataset:

    target_col = st.sidebar.selectbox(
        "Select Target Column",
        df.columns
    )

else:

    target_col = "Willing_to_Pay_for_Access"

    st.info(
        f"Target Column: {target_col}"
    )

# =====================================================
# ENCODE DATA
# =====================================================

for col in df.columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

# =====================================================
# FEATURES/TARGET
# =====================================================

X = df.drop(
    target_col,
    axis=1
)

y = df[target_col]

# =====================================================
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# SCALE
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# =====================================================
# MODEL SELECTION
# =====================================================

st.subheader("Choose Model")

model_name = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]
)

# =====================================================
# MODEL CREATION
# =====================================================

if model_name == "Logistic Regression":

    model = LogisticRegression(
        max_iter=1000
    )

elif model_name == "Decision Tree":

    model = DecisionTreeClassifier(
        random_state=42
    )

elif model_name == "KNN":

    model = KNeighborsClassifier(
        n_neighbors=5
    )

elif model_name == "Naive Bayes":

    model = GaussianNB()

else:

    model = SVC(
        probability=True,
        random_state=42
    )

# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(
    X_train,
    y_train
)

# =====================================================
# PREDICTION
# =====================================================

y_pred = model.predict(
    X_test
)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)

# =====================================================
# AUC
# =====================================================

try:

    y_prob = model.predict_proba(
        X_test
    )

    if len(y.unique()) == 2:

        auc = roc_auc_score(
            y_test,
            y_prob[:, 1]
        )

    else:

        auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr"
        )

except:

    auc = None

# =====================================================
# SHOW METRICS
# =====================================================

st.subheader("Evaluation Metrics")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

c2.metric(
    "Precision",
    f"{precision:.4f}"
)

c3.metric(
    "Recall",
    f"{recall:.4f}"
)

c4, c5, c6 = st.columns(3)

c4.metric(
    "F1 Score",
    f"{f1:.4f}"
)

c5.metric(
    "MCC",
    f"{mcc:.4f}"
)

if auc is not None:

    c6.metric(
        "AUC",
        f"{auc:.4f}"
    )

else:

    c6.metric(
        "AUC",
        "N/A"
    )

# =====================================================
# CONFUSION MATRIX
# =====================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(6, 4)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(
"""
Built for Machine Learning Assignment 2

Models Implemented:
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- SVM
"""
)
