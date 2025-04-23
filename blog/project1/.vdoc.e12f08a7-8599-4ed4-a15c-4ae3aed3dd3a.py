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
#| collapse: true
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import scipy.stats as stats

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

df = pd.read_stata("karlan_list_2007.dta")

print("Dataset shape:", df.shape)
print("\nFirst few rows of the dataset:")
df.head()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| collapse: true
# Perform balance tests comparing treatment and control groups across non key variables
def compare_groups(df, var_name):
    control = df[df['treatment'] == 0][var_name].dropna()
    treat = df[df['treatment'] == 1][var_name].dropna()
    
    t_stat = (treat.mean() - control.mean()) / np.sqrt(
        (treat.var() / len(treat)) + (control.var() / len(control))
    )
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(treat) + len(control) - 2))
    
    model = smf.ols(f"{var_name} ~ treatment", data=df).fit()
    
    print(f"\nResults for {var_name}:")
    print("-" * 50)
    print(f"Treatment mean: {treat.mean():.3f}")
    print(f"Control mean: {control.mean():.3f}")
    print(f"\nT-test results:")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.3f}")
    print(f"\nRegression results:")
    print(model.summary().tables[1])
    print("\n")

variables = ['mrm2', 'female', 'couple', 'freq']
for var in variables:
    compare_groups(df, var)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| collapse: true
# Analyze impact of matching grants on donation probability using multiple statistical approaches
donation_rates = df.groupby("treatment")["gave"].mean()
donation_rates.index = ["Control", "Treatment"]

plt.figure(figsize=(10, 6))
ax = donation_rates.plot(kind="bar", color=['blue', 'green'])
plt.ylabel("Proportion Donated", fontsize=12)
plt.title("Donation Rate by Treatment Group", fontsize=14)
plt.ylim(0, 0.04)

for i, v in enumerate(donation_rates):
    ax.text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=11)

plt.tight_layout()
plt.show()

control_gave = df[df["treatment"] == 0]["gave"]
treat_gave = df[df["treatment"] == 1]["gave"]
t_stat_gave, p_val_gave = stats.ttest_ind(treat_gave, control_gave)
print("T-test for donation rate:")
print(f"t-statistic: {t_stat_gave:.4f}")
print(f"p-value: {p_val_gave:.4f}")

model_gave = smf.ols("gave ~ treatment", data=df).fit()
print("\nLinear regression for donation rate:")
print(model_gave.summary().tables[1])

probit_model = smf.probit("gave ~ treatment", data=df).fit()
print("\nProbit regression for donation rate:")
print(probit_model.summary().tables[1])

mfx = probit_model.get_margeff()
print("\nMarginal effects:")
print(mfx.summary().tables[0])
#
#
#
#
#
#
#
#
#
#
#| collapse: true
# Compare effectiveness of different match ratios (1:1, 2:1, 3:1) on donation rates
treat_df = df[df["treatment"] == 1].copy()

for col in ["ratio", "ratio2", "ratio3"]:
    treat_df[col] = pd.to_numeric(treat_df[col], errors='coerce').fillna(0).astype(int)

treat_df["ratio1"] = treat_df["ratio"]

ratio_rates = treat_df.groupby(["ratio1", "ratio2", "ratio3"])["gave"].mean()
print("Donation rates by match ratio:")
print(ratio_rates)

group_1_1 = treat_df[treat_df["ratio1"] == 1]["gave"]
group_2_1 = treat_df[treat_df["ratio2"] == 1]["gave"]
t_2, p_2 = stats.ttest_ind(group_2_1, group_1_1)
print("\n2:1 vs 1:1 match rate:")
print(f"t-statistic: {t_2:.4f}")
print(f"p-value: {p_2:.4f}")

group_3_1 = treat_df[treat_df["ratio3"] == 1]["gave"]
t_3, p_3 = stats.ttest_ind(group_3_1, group_1_1)
print("\n3:1 vs 1:1 match rate:")
print(f"t-statistic: {t_3:.4f}")
print(f"p-value: {p_3:.4f}")

model_ratio = smf.ols("gave ~ ratio1 + ratio2 + ratio3", data=treat_df).fit()
print("\nRegression on match ratios:")
print(model_ratio.summary().tables[1])

response_1_1 = treat_df[treat_df["ratio1"] == 1]["gave"].mean()
response_2_1 = treat_df[treat_df["ratio2"] == 1]["gave"].mean()
response_3_1 = treat_df[treat_df["ratio3"] == 1]["gave"].mean()

diff_2v1 = response_2_1 - response_1_1
diff_3v2 = response_3_1 - response_2_1

print("\nResponse rate differences:")
print(f"2:1 vs 1:1: {diff_2v1:.4f}")
print(f"3:1 vs 2:1: {diff_3v2:.4f}")

ratio_data = pd.DataFrame({
    'Match Ratio': ['1:1', '2:1', '3:1'],
    'Donation Rate': [response_1_1, response_2_1, response_3_1]
})

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Match Ratio', y='Donation Rate', data=ratio_data, palette='viridis')
plt.title('Donation Rate by Match Ratio', fontsize=14)
plt.ylabel('Proportion Donated', fontsize=12)

for i, v in enumerate(ratio_data['Donation Rate']):
    ax.text(i, v + 0.001, f"{v:.3f}", ha='center', fontsize=11)

plt.tight_layout()
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#| collapse: true
# Analyzing the impact of matching grants on donation amounts and visualize distributions
amount_control = df[df["treatment"] == 0]["amount"]
amount_treatment = df[df["treatment"] == 1]["amount"]
t_stat_amt, p_val_amt = stats.ttest_ind(amount_treatment, amount_control)

print("T-test on donation amount:")
print(f"t-statistic: {t_stat_amt:.4f}")
print(f"p-value: {p_val_amt:.4f}")

model_amt = smf.ols("amount ~ treatment", data=df).fit()
print("\nRegression on donation amount:")
print(model_amt.summary().tables[1])

df_gave = df[df["gave"] == 1].copy()

model_cond = smf.ols("amount ~ treatment", data=df_gave).fit()
print("\nRegression on donation amount (only for donors):")
print(model_cond.summary().tables[1])

mean_treat = df_gave[df_gave["treatment"] == 1]["amount"].mean()
mean_control = df_gave[df_gave["treatment"] == 0]["amount"].mean()

print(f"\nMean donation amount (treatment): ${mean_treat:.2f}")
print(f"Mean donation amount (control): ${mean_control:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

treat_donors = df_gave[df_gave["treatment"] == 1]["amount"]
axs[0].hist(treat_donors, bins=30, color='blue', edgecolor='black', alpha=0.7)
axs[0].axvline(mean_treat, color='red', linestyle='dashed', linewidth=2)
axs[0].set_title("Treatment Group", fontsize=14)
axs[0].set_xlabel("Donation Amount ($)", fontsize=12)
axs[0].set_ylabel("Frequency", fontsize=12)

control_donors = df_gave[df_gave["treatment"] == 0]["amount"]
axs[1].hist(control_donors, bins=30, color='green', edgecolor='black', alpha=0.7)
axs[1].axvline(mean_control, color='red', linestyle='dashed', linewidth=2)
axs[1].set_title("Control Group", fontsize=14)
axs[1].set_xlabel("Donation Amount ($)", fontsize=12)

plt.suptitle("Distribution of Donation Amounts (Donors Only)", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| collapse: true
# Simulate Law of Large Numbers to demonstrate convergence of sample means to true difference
np.random.seed(42)

n = 10000
p_treat = 0.022
p_control = 0.018
true_diff = p_treat - p_control

treatment = np.random.binomial(1, p_treat, n)
control = np.random.binomial(1, p_control, n)
diffs = treatment - control
cumulative_avg = np.cumsum(diffs) / np.arange(1, n + 1)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_avg, label="Cumulative Avg. Difference", color="steelblue", linewidth=2)
plt.axhline(true_diff, color='red', linestyle='--', linewidth=2, label=f"True Difference ({true_diff:.3f})")

plt.title("Law of Large Numbers Simulation", fontsize=16, weight='bold')
plt.xscale('log')
plt.xlabel("Number of Observations", fontsize=13)
plt.ylabel("Cumulative Average Difference", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
#
#
#
#
#
#
#
#
#
#| collapse: true
# Demonstrate Central Limit Theorem using different sample sizes to show convergence to normal distribution
p_control = 0.018
p_treat = 0.022
n_sim = 1000
sample_sizes = [50, 200, 500, 1000]

np.random.seed(42)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, n in enumerate(sample_sizes):
    diff_means = []
    
    for _ in range(n_sim):
        control_sample = np.random.binomial(1, p_control, n)
        treat_sample = np.random.binomial(1, p_treat, n)
        diff = treat_sample.mean() - control_sample.mean()
        diff_means.append(diff)

    axs[i].hist(diff_means, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    axs[i].axvline(0, color='red', linestyle='--', linewidth=2)
    axs[i].set_title(f"Sample Size: {n}", fontsize=14)
    axs[i].set_xlabel("Mean Difference (Treatment - Control)")
    axs[i].set_ylabel("Frequency")

plt.suptitle("Central Limit Theorem Simulation: Distribution of Mean Differences", 
             fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
