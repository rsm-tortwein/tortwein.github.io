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


## Q2

# Mean donation rates and amounts by treatment group
df.groupby("treatment")[["gave", "amount"]].mean()

import statsmodels.formula.api as smf

# Linear probability model
model_gave = smf.ols("gave ~ treatment", data=df).fit()
print(model_gave.summary())


# OLS on amount donated
model_amt = smf.ols("amount ~ treatment", data=df).fit()
print(model_amt.summary())

"""
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
#
#
#
#
#
