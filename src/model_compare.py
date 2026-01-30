import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MSE": mse,
        "R2 Score": r2
    })

results_df = pd.DataFrame(results)
print("\nModel Karşılaştırması:")
print(results_df)
#Regularized regression models such as Ridge slightly improved generalization performance compared to standard linear regression.
#Sparser models generalize better in this dataset.
#Among the evaluated models, Lasso regression achieved the highest R² score, suggesting that feature selection plays an important role in modeling diabetes progression from clinical variables.

import matplotlib.pyplot as plt
import pandas as pd

# Lasso modelini tekrar eğitelim
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Feature importance (katsayılar)
importance = pd.Series(
    lasso.coef_,
    index=X.columns
).sort_values()

print("\nLasso Feature Importance:")
print(importance)

# Grafik
plt.figure()
importance.plot(kind="barh")
plt.title("Lasso Feature Importance (Diabetes Progression)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("lasso_feature_importance.png")
plt.show()
#Lasso regression highlights BMI and metabolic markers as the most influential predictors of diabetes progression.
