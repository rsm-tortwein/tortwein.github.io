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

# q1
print("Charitable Contribution Made")

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

# q2

treat_df = df[df["treatment"] == 1]

# Regression with dummies for ratio types
model = smf.ols("gave ~ ratio + ratio2 + ratio3", data=treat_df).fit()
print(model.summary())


treat_df = df[df["treatment"] == 1].copy()

# Create a new variable to label match type
treat_df["ratio_type"] = (
    treat_df["ratio2"] * 2 +
    treat_df["ratio3"] * 3 +
    treat_df["ratio"] * 1
)

# Convert to categorical and set 1:1 as baseline
treat_df["ratio_type"] = treat_df["ratio_type"].astype("category")
treat_df["ratio_type"] = treat_df["ratio_type"].cat.rename_categories({1: "1:1", 2: "2:1", 3: "3:1"})

# Regression with 1:1 as reference
model = smf.ols("gave ~ C(ratio_type)", data=treat_df).fit()
print(model.summary())

#q3

response_rates = treat_df.groupby("ratio_type")["gave"].mean()
print(response_rates)

# Differences
diff_2v1 = response_rates["2:1"] - response_rates["1:1"]
diff_3v2 = response_rates["3:1"] - response_rates["2:1"]
print("\nResponse rate difference 2:1 vs 1:1:", diff_2v1)
print("Response rate difference 3:1 vs 2:1:", diff_3v2)


#
#
#
#
#
