# earthquake
#This project builds a machine learning pipeline to classify earthquake magnitudes into Minor, Moderate, and Strong categories using #XGBoost. It includes complete preprocessing, class balancing with SMOTE, model training with regularization, and evaluation with ROC curves and confusion matrices.

#🚀 Features
#Time-based feature extraction (year, month, day, hour)

#Missing value imputation (median strategy)

#Label encoding of categorical variables

#Magnitude classification into 3 categories

#Standardization of features

#SMOTE for class balancing

#XGBoost with multi-class support & early stopping

#ROC-AUC, confusion matrix, and classification report for evaluation

#📁 Dataset
#Input: Earthquake dataset in CSV format

#Target: Categorized magnitude (mag_category)

#Format: Make sure your dataset path is updated in the script

#🛠️ Libraries Used
#pandas, numpy

#scikit-learn

#imbalanced-learn (SMOTE)

#xgboost

#matplotlib, seaborn

#🧪 How to Run
#Clone the repository

#Update the dataset path in the script

#Run the Python script:

#bash
#Copy
#Edit
#python earthquake_classifier.py
#📊 Outputs
#ROC curves per class with AUC scores

#Confusion matrix heatmap

#Train vs Test classification reports

#🎯 Target Classes
#Class	Magnitude Range	Description
#0	< 4.0	Minor
#1	4.0 – 5.9	Moderate
#2	≥ 6.0	Strong
