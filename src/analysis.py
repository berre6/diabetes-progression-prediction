import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset'i yükle
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

print("İlk 5 satır:")
print(df.head())

# 2. Feature ve target ayır
X = df.drop("target", axis=1)
y = df["target"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Tahmin
y_pred = model.predict(X_test)

# 6. Değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performansı:")
print("MSE:", mse)
print("R2 Score:", r2)
#I framed diabetes progression as a regression task and achieved an R² score of approximately 0.45 using a linear regression model on clinical features.

