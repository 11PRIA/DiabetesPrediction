# Importing essential libraries
import numpy as np
import pandas as pd
import joblib  # Using joblib instead of pickle for better compatibility

# Loading the dataset
try:
    df = pd.read_csv('kaggle_diabetes.csv')
except FileNotFoundError:
    print("Error: 'kaggle_diabetes.csv' file not found!")
    print("Please download the dataset from Kaggle:")
    print("https://www.kaggle.com/datasets/mathchi/diabetes-data-set")
    exit(1)

# Renaming DiabetesPedigreeFunction as DPF
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

# Calculate and store preprocessing values (mean/median) for use during prediction
glucose_mean = df_copy['Glucose'].mean()
bloodpressure_mean = df_copy['BloodPressure'].mean()
skinthickness_median = df_copy['SkinThickness'].median()
insulin_median = df_copy['Insulin'].median()
bmi_median = df_copy['BMI'].median()

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(glucose_mean, inplace=True)
df_copy['BloodPressure'].fillna(bloodpressure_mean, inplace=True)
df_copy['SkinThickness'].fillna(skinthickness_median, inplace=True)
df_copy['Insulin'].fillna(insulin_median, inplace=True)
df_copy['BMI'].fillna(bmi_median, inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
# FIXED: Use df_copy instead of df (the cleaned dataset)
X = df_copy.drop(columns='Outcome')
y = df_copy['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)

# Print accuracy
from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained successfully!")
print(f"Accuracy: {accuracy:.4f}")

# Save preprocessing parameters along with the model
preprocessing_params = {
    'glucose_mean': glucose_mean,
    'bloodpressure_mean': bloodpressure_mean,
    'skinthickness_median': skinthickness_median,
    'insulin_median': insulin_median,
    'bmi_median': bmi_median
}

# Creating pickle files for the classifier and preprocessing params
model_filename = 'diabetes-prediction-rfc-model.pkl'
preprocessing_filename = 'preprocessing_params.pkl'

joblib.dump(classifier, model_filename)
joblib.dump(preprocessing_params, preprocessing_filename)

print(f"Model saved to {model_filename}")
print(f"Preprocessing parameters saved to {preprocessing_filename}")


