# AI-Tool-Usage-Classification using Machine Learning

**a. Problem Statement**
The objective of this project is to predict whether a student is willing to pay for AI tool access based on academic background, Al usage habits, awareness level, internet access, trust in Al tools, and other related features.
This is a binary classification problem where:
0-No
1-Yes

**b. Dataset Description**
Dataset Name: Al Tool Usage by Indian College Students 2025
Number of Records: 3614
Number of Features: 16
Target Variable: Willing_to_Pay_for_Access
Source:[
https://www.kaggle.com/datasets/iamsouravbanerjee/ai-tool-usage-by-indian-college-students-2025](url)
Features include:
Stream
Year of Study Al Tools Used
Daily Usage Hours Use Cases
Trust in Al Tools Impact on Grades Preferred AI_Tool Awareness Level State

**c. Observations**
Logistic Regression
Logistic Regression achieved the highest Accuracy, Recall, F1 Score, and MCC Score. It was the best- performing model overall.
Decision Tree
Decision Tree performed reasonably well and captured nonlinear patterns in student behavior. KNN
KNN produced performance similar to Decision Tree but was slightly less accurate.
Naive Bayes
Naive Bayes trained very quickly but achieved lower performance because of its independence assumption. Support Vector Machine
SVM achieved the highest AUC score but lower Accuracy and F1 Score.
Overall Winner
Logistic Regression was the best-performing model for this dataset.

**d. Technologies Used**
Python Pandas
NumPy
Scikit-Learn
Streamlit
Matplotlib Seabor

**STREAMLIT link :** [https://ai-tool-usage-classification.streamlit.app/](url)

**Author**
BHARGAV JANI
