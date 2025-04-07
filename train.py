import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv('data/rain.csv')

# Preprocess data
data.dropna(inplace=True)
X = data.drop('rainfall', axis=1)
y = data['rainfall']

# Encode categorical features (if any)
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'models/rainfall_model.pkl')
print("Model saved to models/rainfall_model.pkl")
