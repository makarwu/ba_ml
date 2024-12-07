import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Read in the data
df = pd.read_excel('e-commerce-dataset.xlsx', sheet_name='E_Comm')
df.to_csv('e-commerce_churn.csv', index=False)
df = pd.read_csv('e-commerce_churn.csv')

##################################################################
######################## Data preparation ########################
##################################################################

print("\n---------------------------------------\n\
---------- Data Preparation -----------\n\
----------------------------------------\n")

print(df.info())
df = df.dropna()

df['Churn'] = df['Churn'].astype('category')

# Identify non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object']).columns

# Apply one-hot encoding to non-numeric columns
df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)

##################################################################
########################       a.)        ########################
##################################################################
print("\n---------------------------------------\n\
-------------- Part a.) -----------------\n\
----------------------------------------\n")

# Train a random forest model

# Calculate accuracy

##################################################################
########################       c.)        ########################
##################################################################
print("\n---------------------------------------\n\
-------------- Part c.) -----------------\n\
----------------------------------------\n")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

# Train-test split (60-20-20)
train, test = train_test_split(df, test_size=0.4, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)

# View proportions of Churn
print("Training Set Churn Proportions:")
print(train['Churn'].value_counts(normalize=True))
print("Validation Set Churn Proportions:")
print(val['Churn'].value_counts(normalize=True))
print("Test Set Churn Proportions:")
print(test['Churn'].value_counts(normalize=True))

# Define Model
model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

# Cross-validation
cross_val_scores = cross_val_score(model, train.drop('Churn', axis=1), train['Churn'], cv=4, scoring='accuracy')
print("Cross-validation Precision Scores:", cross_val_scores)
print("Mean Accuracy:", np.mean(cross_val_scores))

# Train the final model
model.fit(train.drop('Churn', axis=1), train['Churn'])

# Variable Importance Plot
importances = model.feature_importances_
plt.barh(train.drop('Churn', axis=1).columns, importances)
plt.title("Feature Importance")
plt.show()

# Apply on test set
test_predictions = model.predict(test.drop('Churn', axis=1))

# Confusion Matrix
conf_matrix = confusion_matrix(test['Churn'], test_predictions)
print("Confusion Matrix:\n", conf_matrix)

# Precision, accuracy, recall
precision = precision_score(test['Churn'], test_predictions)
accuracy = accuracy_score(test['Churn'], test_predictions)
recall = recall_score(test['Churn'], test_predictions)
print(f"Precision: {precision:.2f}, Accuracy: {accuracy:.2f}, Recall: {recall:.2f}")