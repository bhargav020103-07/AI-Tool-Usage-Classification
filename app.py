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

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Classification Model Comparison",
    layout="wide"
)

st.title("🚀 Classification Model Comparison Tool")

st.write(
    """
    Upload a CSV dataset,
    select the target column,
    choose a machine learning model,
    and evaluate performance.
    """
)

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    # ==========================================
    # TARGET SELECTION
    # ==========================================

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # ==========================================
    # HANDLE MISSING VALUES
    # ==========================================

    df = df.dropna()

    # ==========================================
    # LABEL ENCODING
    # ==========================================

    for col in df.columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

    # ==========================================
    # FEATURES + TARGET
    # ==========================================

    X = df.drop(
        target_column,
        axis=1
    )

    y = df[target_column]

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ==========================================
    # SCALING
    # ==========================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ==========================================
    # MODEL SELECTION
    # ==========================================

    model_name = st.selectbox(
        "Choose Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "SVM"
        ]
    )

    # ==========================================
    # CREATE MODEL
    # ==========================================

    if model_name == "Logistic Regression":

        model = LogisticRegression(max_iter=1000)

    elif model_name == "Decision Tree":

        model = DecisionTreeClassifier()

    elif model_name == "KNN":

        model = KNeighborsClassifier()

    elif model_name == "Naive Bayes":

        model = GaussianNB()

    else:

        model = SVC(
            probability=True
        )

    # ==========================================
    # TRAIN MODEL
    # ==========================================

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    try:
        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        auc = roc_auc_score(
            y_test,
            y_prob
        )

    except:
        auc = "Not Available"

    # ==========================================
    # METRICS
    # ==========================================

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

    # ==========================================
    # DISPLAY METRICS
    # ==========================================

    st.subheader("Performance Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        round(accuracy, 4)
    )

    c2.metric(
        "Precision",
        round(precision, 4)
    )

    c3.metric(
        "Recall",
        round(recall, 4)
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "F1 Score",
        round(f1, 4)
    )

    c5.metric(
        "MCC",
        round(mcc, 4)
    )

    c6.metric(
        "AUC",
        auc if isinstance(auc, str)
        else round(auc, 4)
    )

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

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

    st.pyplot(fig)

    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================

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
