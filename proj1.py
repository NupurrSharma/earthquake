import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/Users/nupursharma/Downloads/query (2).csv")

# Time feature engineering
df["time"] = pd.to_datetime(df["time"])
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day
df["hour"] = df["time"].dt.hour
df.drop(columns=["time", "id", "place", "updated"], inplace=True, errors="ignore")

# Handle numeric/categorical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Impute numeric columns
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Categorize magnitudes
def categorize_magnitude(mag):
    if mag < 4.0:
        return 0  # Minor
    elif 4.0 <= mag < 6.0:
        return 1  # Moderate
    else:
        return 2  # Strong

df["mag_category"] = df["mag"].apply(categorize_magnitude)
df.drop(columns=["mag"], inplace=True)

# Final check
if len(df.select_dtypes(include='object').columns) > 0:
    raise ValueError("❌ Non-numeric columns remain after preprocessing!")

# Features and target
X = df.drop(columns=["mag_category"])
y = df["mag_category"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance classes with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Create DMatrix objects for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters with regularization
params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.2,   # L1 regularization
    'reg_lambda': 1.5,  # L2 regularization
    'seed': 42
}

# Train model with early stopping
evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=150,
    evals=evals,
    early_stopping_rounds=15,
    verbose_eval=True
)

# Predict
y_prob = model.predict(dtest)
y_pred = np.argmax(y_prob, axis=1)

# ROC-AUC plot
fpr, tpr, roc_auc = {}, {}, {}
n_classes = len(np.unique(y))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 7))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=["Minor", "Moderate", "Strong"],
            yticklabels=["Minor", "Moderate", "Strong"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification report - Test
print("✅ Test Classification Report:\n", classification_report(y_test, y_pred))

# Train predictions and report for overfitting check
y_train_pred = np.argmax(model.predict(dtrain), axis=1)
print("📊 Train Classification Report:\n", classification_report(y_train, y_train_pred))
