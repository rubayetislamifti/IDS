# Load dataset
import pandas as pd
df = pd.read_csv("WSN-IDS_Dataset.csv")

# Preprocess
df = df.dropna()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

X = df.drop('label', axis=1)
y = df['label']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.3, random_state=42)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, f1_score
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))
