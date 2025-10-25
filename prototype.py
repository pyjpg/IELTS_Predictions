import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("data/ielts_clean.csv")
X = df['Essay']
y = df['Overall']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mini model complete. MAE = {mae:.3f}")

joblib.dump(vectorizer, "vectorizer.pkl")

joblib.dump(model, "linear_model.pkl")