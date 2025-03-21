import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the Titanic dataset
train_df = pd.read_csv('train.csv')

# Data Preprocessing
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Split the data into features and target
X = train_df.drop(['Survived','PassengerId'], axis=1)
y = train_df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(rf_model, 'titanic_rf_model.pkl')

print("Model training complete and saved as titanic_rf_model.pkl")
