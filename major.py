
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# Step 1: Import and explore the data
print("Loading data...")
try:
    # Update the file path if 'sensor-data.csv' is located elsewhere
    file_path = "sensor-data.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found in the current directory.")
    
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")

    print("Initial Data Overview:")
    print(data.head())
    print(data.info())
    print(data.describe())
except FileNotFoundError as e:
    print(e)
    print("Please ensure the dataset is in the correct path.")
    exit()

# Step 2: Data cleansing
print("\nData Cleansing...")
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values per Column:\n", missing_values[missing_values > 0])

# Drop columns with excessive missing values if necessary
# No columns dropped here due to lack of functional context
# Fill missing values with column mean
data.fillna(data.mean(), inplace=True)

# Verify missing values handled
print("Missing values after treatment:\n", data.isnull().sum().sum())

# Ensure 'Target' column exists in the dataset
if 'Target' not in data.columns:
    print("Error: Target column not found in the dataset.")
    exit()

# Step 3: Data analysis & visualization
print("\nPerforming Data Analysis...")
# Distribution of target variable
sns.countplot(data['Target'])
plt.title('Pass/Fail Distribution')
plt.show()

# Univariate analysis example: Histogram of a random feature
random_feature = data.columns[1]
sns.histplot(data[random_feature], kde=True, bins=30)
plt.title(f'Distribution of {random_feature}')
plt.show()

# Correlation heatmap (taking only numerical features for large datasets)
correlation_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Data preprocessing
print("\nData Preprocessing...")
X = data.drop(columns=['Target'])  # All columns except the target column
y = data['Target']

# Handle class imbalance with SMOTE
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"Class distribution after SMOTE: {np.bincount(y_resampled)}")

# Train-test split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# Standardize the data
print("Standardizing data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Model training, testing, and tuning
print("\nModel Training and Testing...")

def train_and_evaluate_model(model, param_grid=None):
    """Train and evaluate the model with optional hyperparameter tuning."""
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return best_model

# Random Forest
print("\nRandom Forest Classifier...")
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(random_state=42)
best_rf_model = train_and_evaluate_model(rf_model, rf_param_grid)

# Support Vector Machine
print("\nSupport Vector Machine...")
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}
svm_model = SVC(random_state=42)
best_svm_model = train_and_evaluate_model(svm_model, svm_param_grid)

# Naive Bayes
print("\nNaive Bayes Classifier...")
nb_model = GaussianNB()
best_nb_model = train_and_evaluate_model(nb_model)

# Compare models
print("\nModel Comparisons:")
rf_acc = cross_val_score(best_rf_model, X_resampled, y_resampled, cv=5).mean()
svm_acc = cross_val_score(best_svm_model, X_resampled, y_resampled, cv=5).mean()
nb_acc = cross_val_score(best_nb_model, X_resampled, y_resampled, cv=5).mean()

print(f"Random Forest Accuracy: {rf_acc}")
print(f"SVM Accuracy: {svm_acc}")
print(f"Naive Bayes Accuracy: {nb_acc}")

# Step 6: Conclusion and Improvisation
print("\nFinalizing the Best Model...")
final_model = best_rf_model if rf_acc >= max(svm_acc, nb_acc) else (best_svm_model if svm_acc > nb_acc else best_nb_model)
print(f"Selected Model: {type(final_model).__name__}")

# Save the final model
print("Saving the final model...")
joblib.dump(final_model, "final_model.pkl")
print("Model saved as 'final_model.pkl'.")
