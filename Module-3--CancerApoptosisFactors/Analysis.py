# %%
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# %% --- LOAD DATA ---

data_folder = r"C:/Users/ugj3eb/Desktop/Third Sem/Computational BME/Comp BME/Module 3"
expr_file = os.path.join(data_folder, "Gene_subset_data.csv")
meta_file = os.path.join(data_folder, "patient_cancer_type.csv")

expr = pd.read_csv(expr_file, index_col=0, low_memory=False)
meta = pd.read_csv(meta_file)

# Keep only LUAD and THCA
meta = meta[meta["cancer_type"].isin(["LUAD", "THCA"])]

# Align intersection of samples
common_samples = expr.columns.intersection(meta["sample"])
expr = expr[common_samples]
meta = meta[meta["sample"].isin(common_samples)]

# Transpose to samples x genes
X = expr.T
X = X.dropna(axis=1, how='all')
X.columns = X.columns.astype(str)

# Labels
y = meta.set_index("sample").loc[X.index, "cancer_type"]

print(f"Data shape: {X.shape}")
print(y.value_counts())

# %% --- IMPUTE MISSING VALUES ---
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), index=X.index, columns=X.columns)

# %% --- STANDARDIZE ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# %% --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.5, random_state=42, stratify=y
)

# %% --- PCA (fit ONLY on training data) ---
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# %% --- LOGISTIC REGRESSION TRAINED ONLY ON TRAIN DATA ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

y_test_num = (y_test == "LUAD").astype(int)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Test accuracy: {acc:.3f}")

# %% --- LOGISTIC REGRESSION IN PCA SPACE (for visualization only) ---
model_2d = LogisticRegression(max_iter=1000)
model_2d.fit(X_train_pca, y_train)

# Create meshgrid for decision boundary
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = model_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# %% --- PLOT TRAINING DATA DECISION BOUNDARY ---
plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train,
                edgecolor='k', palette="Set1")
plt.title("Decision Boundary (Training Data, PCA space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %% --- PLOT TEST DATA DECISION BOUNDARY ---
plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test,
                edgecolor='k', palette="Set1")
plt.title("Decision Boundary (Test Data, PCA space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %% --- PLOT: Predicted probability vs true label (Model fit visualization) ---

plt.figure(figsize=(7,5))

# Sort by predicted probability for a clean curve
sorted_idx = np.argsort(y_proba)
probs_sorted = y_proba[sorted_idx]
truth_sorted = y_test_num.values[sorted_idx]

plt.plot(probs_sorted, label="Predicted probability of LUAD")
plt.scatter(range(len(truth_sorted)), truth_sorted,
            color="black", s=20, label="True label (0=THCA, 1=LUAD)")

plt.title("Logistic Regression Fit: Predicted Probability vs True Labels")
plt.ylabel("Probability / True Label")
plt.xlabel("Sorted Samples")
plt.legend()
plt.tight_layout()
plt.show()
