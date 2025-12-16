import pandas as pd
import numpy as np
from patient import Patient
import matplotlib.pyplot as plt
import re

Patient.combine_and_instantiate()

data = []

for patient in Patient.all_patients:
    data.append({
        'Donor ID': patient.get_id(),
        'Atherosclerosis': patient.get_atherosclerosis(),
        'THAL': patient.get_THAL(),
        'ABeta40': patient.get_ABeta40(),
        'ABeta42': patient.get_ABeta42()
    })

# Example DataFrame creation
df = pd.DataFrame(data, columns=['Donor ID', 'Atherosclerosis', 'THAL', 'THAL_score', 'ABeta40', 'ABeta42'])

# Convert severity levels to numeric (For ordering)
severity_order = ["mild", "moderate", "severe"]

# Extracts the numerical part of the THAL string
df["THAL_score"] = df["THAL"].apply(
    lambda x: int(re.search(r'\d+', str(x)).group())
)

# Boxplot of THAL score by Atherosclerosis severity
df.boxplot(column="THAL_score", by="Atherosclerosis", grid=False, positions=range(len(severity_order)))

plt.xlabel("Atherosclerosis Severity")
plt.ylabel("THAL Score")
plt.title("Distribution of THAL Score by Atherosclerosis Severity")
plt.suptitle("")  # removes the default pandas boxplot title
plt.show()

# T-test between mild and moderate groups (they have more samples)
# One sided because amyloid beta is expected to increase with atherosclerosis
# Unpaired because the patients aren't the same
from scipy import stats
mild_scores = df[df["Atherosclerosis"] == "Mild"]["THAL_score"].dropna()
moderate_scores = df[df["Atherosclerosis"] == "Moderate"]["THAL_score"].dropna()

# Are the variances equal?
print("mild score variance", np.var(mild_scores), " moderate score variance", np.var(moderate_scores)) # Pretty similar


t_statistic, p_value = stats.ttest_ind(mild_scores, moderate_scores, equal_var=False)
print(f"T-statistic: {t_statistic}, P-value: {p_value}")
