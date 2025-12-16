# %%
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
housing = fetch_california_housing(as_frame=True)
print(housing.data.shape, housing.target.shape)
print(housing.feature_names[0:6])

# %%
print(housing.DESCR)

# %% Find best single feature
feature_list = housing["data"].columns
best_feature = ""
best_R2 = 0
best_model = None

for each in feature_list:
    X = housing["data"][[each]].values
    y = housing.target
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    if r2 > best_R2:
        best_R2 = r2
        best_feature = each
        best_model = reg

print(f"Best single feature: {best_feature}")
print(f"Best R²: {best_R2:.3f}")

# %% Plot the best feature
X = housing["data"][[best_feature]].values
y = housing.target
reg = best_model

x_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_test = reg.predict(x_test)

plt.scatter(X, y, alpha=0.3)
plt.plot(x_test, y_test, color="red")
plt.xlabel(best_feature)
plt.ylabel("House Value ($100k)")
plt.annotate(
    f"R² = {best_R2:.2f}",
    xy=(0.5, 0.9),
    xycoords="axes fraction",
    fontsize=14,
    ha="center",
)
plt.show()

# %% R² with all features
X_all = housing["data"]
y = housing.target
reg_all = LinearRegression().fit(X_all, y)
print(f"R² with all features: {reg_all.score(X_all, y):.3f}")
