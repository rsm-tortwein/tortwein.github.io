# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import pandas as pd

df = pd.read_stata("karlan_list_2007.dta")

df.info()
df.head()
print(df.columns)


## Balance Test
print("Balance Test")

# Mean donation rates and amounts by treatment group
df.groupby("treatment")[["gave", "amount"]].mean()

import statsmodels.formula.api as smf

# Linear probability model
model_gave = smf.ols("gave ~ treatment", data=df).fit()
print(model_gave.summary())


# OLS on amount donated
model_amt = smf.ols("amount ~ treatment", data=df).fit()
print(model_amt.summary())
import scipy.stats as stats

# T-test: compare means of mrm2 by treatment
control = df[df["treatment"] == 0]["mrm2"]
treat = df[df["treatment"] == 1]["mrm2"]
t_stat, p_val = stats.ttest_ind(control, treat)
print("T-test result:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# Linear regression: mrm2 ~ treatment
model = smf.ols("mrm2 ~ treatment", data=df).fit()
print("\nLinear regression result:")
print(model.summary())

## Charitable Contribution Made


print("Charitable Contribution Made")
# q1

import matplotlib.pyplot as plt

# Calculate proportion of people who donated in each group
donation_rates = df.groupby("treatment")["gave"].mean()
donation_rates.index = ["Control", "Treatment"]

# Plot
donation_rates.plot(kind="bar")
plt.ylabel("Proportion Donated")
plt.title("Donation Rate by Treatment Group")
plt.ylim(0, 0.04)
plt.show()

#q2

# T-test
control = df[df["treatment"] == 0]["gave"]
treat = df[df["treatment"] == 1]["gave"]
t_stat, p_val = stats.ttest_ind(treat, control)
print("T-test:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# Linear regression
model = smf.ols("gave ~ treatment", data=df).fit()
print("\nLinear Regression:")
print(model.summary())

#q3

# Probit model: gave ~ treatment
probit_model = smf.probit("gave ~ treatment", data=df).fit()
print(probit_model.summary())

mfx = probit_model.get_margeff()
print(mfx.summary())

print("Differences between Match Rates")

treat_df = df[df["treatment"] == 1]

# Compare 2:1 vs 1:1
group_1_1 = treat_df[treat_df["ratio"] == 1]["gave"]
group_2_1 = treat_df[treat_df["ratio2"] == 1]["gave"]
t_2, p_2 = stats.ttest_ind(group_2_1, group_1_1)
print("2:1 vs 1:1 match rate:")
print("t-statistic:", t_2, "p-value:", p_2)

# Compare 3:1 vs 1:1
group_3_1 = treat_df[treat_df["ratio3"] == 1]["gave"]
t_3, p_3 = stats.ttest_ind(group_3_1, group_1_1)
print("\n3:1 vs 1:1 match rate:")
print("t-statistic:", t_3, "p-value:", p_3)

print("I tested whether increasing the match ratio from 1:1 to 2:1 or 3:1 significantly \n increased donation rates. The results showed no statistically significant difference in giving behavior \n between the groups — consistent with what Karlan and List suggest in the paper. \n This means that while the presence of a match offer itself increases giving, the size of the match offer does not. From a behavioral standpoint, \n this highlights that what matters most is the perception of support or urgency, not necessarily the financial efficiency of the match.")

# Q2


# Step 1: Filter only treatment group
treat_df = df[df["treatment"] == 1].copy()

# Step 2: Convert dummy columns to clean 0/1 integers
for col in ["ratio", "ratio2", "ratio3"]:
    treat_df[col] = pd.to_numeric(treat_df[col], errors='coerce').fillna(0).astype(int)

# Step 3: Create 'ratio1' (to mirror 'ratio2' and 'ratio3')
treat_df["ratio1"] = treat_df["ratio"]  # 1:1 match group

# Step 4: Run regression
model = smf.ols("gave ~ ratio1 + ratio2 + ratio3", data=treat_df).fit()
print(model.summary())

# Q3

# Direct response rates from the data
response_1_1 = treat_df[treat_df["ratio1"] == 1]["gave"].mean()
response_2_1 = treat_df[treat_df["ratio2"] == 1]["gave"].mean()
response_3_1 = treat_df[treat_df["ratio3"] == 1]["gave"].mean()

# Differences
diff_2v1_data = response_2_1 - response_1_1
diff_3v2_data = response_3_1 - response_2_1

print("From data:")
print("2:1 vs 1:1 response rate difference:", diff_2v1_data)
print("3:1 vs 2:1 response rate difference:", diff_3v2_data)

model = smf.ols("gave ~ ratio2 + ratio3", data=treat_df).fit()
print(model.summary())

# Extract coefficients and compute differences
params = model.params
diff_2v1_reg = params["ratio2"]              # 2:1 vs 1:1
diff_3v2_reg = params["ratio3"] - params["ratio2"]  # 3:1 vs 2:1

print("\nFrom regression coefficients:")
print("2:1 vs 1:1:", diff_2v1_reg)
print("3:1 vs 2:1:", diff_3v2_reg)

print(" \nSize of Charitable Contribution \n")

# Q1

# T-test: compare average donation amount across groups
amount_control = df[df["treatment"] == 0]["amount"]
amount_treatment = df[df["treatment"] == 1]["amount"]
t_stat, p_val = stats.ttest_ind(amount_treatment, amount_control)

print("T-test on donation amount:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# Regression: amount ~ treatment
model = smf.ols("amount ~ treatment", data=df).fit()
print("\nRegression on donation amount:")
print(model.summary())

# Q2

# Filter to only people who gave (positive amount)
df_gave = df[df["gave"] == 1].copy()

# Regression: amount ~ treatment (conditional on giving)
model_cond = smf.ols("amount ~ treatment", data=df_gave).fit()

print("Regression on donation amount (only for donors):")
print(model_cond.summary())

# Q3

import matplotlib.pyplot as plt

# Filter to people who donated
df_gave = df[df["gave"] == 1]

# Split into treatment and control donors
treat_donors = df_gave[df_gave["treatment"] == 1]["amount"]
control_donors = df_gave[df_gave["treatment"] == 0]["amount"]

# Calculate means
mean_treat = treat_donors.mean()
mean_control = control_donors.mean()

# Plot histograms
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Treatment group
axs[0].hist(treat_donors, bins=30, color='skyblue', edgecolor='black')
axs[0].axvline(mean_treat, color='red', linestyle='dashed', linewidth=2)
axs[0].set_title("Treatment Group")
axs[0].set_xlabel("Donation Amount")
axs[0].set_ylabel("Frequency")
axs[0].legend(["Mean", "Donors"])

# Control group
axs[1].hist(control_donors, bins=30, color='lightgreen', edgecolor='black')
axs[1].axvline(mean_control, color='red', linestyle='dashed', linewidth=2)
axs[1].set_title("Control Group")
axs[1].set_xlabel("Donation Amount")
axs[1].legend(["Mean", "Donors"])

plt.tight_layout()
plt.show()

# Law of Large Numbers

import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Parameters
n = 10000
p_treat = 0.022
p_control = 0.018
true_diff = p_treat - p_control

# Simulate
treatment = np.random.binomial(1, p_treat, n)
control = np.random.binomial(1, p_control, n)
diffs = treatment - control
cumulative_avg = np.cumsum(diffs) / np.arange(1, n + 1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(cumulative_avg, label="Cumulative Avg. Difference", color="steelblue", linewidth=2)
plt.axhline(true_diff, color='red', linestyle='--', linewidth=2, label=f"True Difference ({true_diff:.3f})")

# Style
plt.title("Law of Large Numbers Simulation", fontsize=16, weight='bold')
plt.xscale('log')

plt.xlabel("Number of Observations", fontsize=13)
plt.ylabel("Cumulative Average Difference", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Show
plt.show()


#
#
#
#
#
