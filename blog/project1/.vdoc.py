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
#| echo: false
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import scipy.stats as stats

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load the data
df = pd.read_stata("karlan_list_2007.dta")

# Display basic information about the dataset
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
#
#
#| echo: false
# Calculate summary statistics for key variables by treatment group
summary_stats = df.groupby("treatment")[["gave", "amount", "mrm2", "freq", "years", "female", "couple"]].mean()
print("Summary Statistics by Treatment Group:")
print(summary_stats)

# T-test for months since last donation (mrm2)
control_mrm2 = df[df["treatment"] == 0]["mrm2"]
treat_mrm2 = df[df["treatment"] == 1]["mrm2"]
t_stat_mrm2, p_val_mrm2 = stats.ttest_ind(control_mrm2, treat_mrm2)
print("\nT-test for months since last donation (mrm2):")
print(f"t-statistic: {t_stat_mrm2:.4f}")
print(f"p-value: {p_val_mrm2:.4f}")

# Linear regression: mrm2 ~ treatment
model_mrm2 = smf.ols("mrm2 ~ treatment", data=df).fit()
print("\nLinear regression for months since last donation:")
print(model_mrm2.summary().tables[1])  # Only show the coefficients table

# T-test for frequency of prior donations
control_freq = df[df["treatment"] == 0]["freq"]
treat_freq = df[df["treatment"] == 1]["freq"]
t_stat_freq, p_val_freq = stats.ttest_ind(control_freq, treat_freq)
print("\nT-test for frequency of prior donations:")
print(f"t-statistic: {t_stat_freq:.4f}")
print(f"p-value: {p_val_freq:.4f}")

# Linear regression: freq ~ treatment
model_freq = smf.ols("freq ~ treatment", data=df).fit()
print("\nLinear regression for frequency of prior donations:")
print(model_freq.summary().tables[1])  # Only show the coefficients table

# Additional balance tests for key demographic variables
# T-test for female
control_female = df[df["treatment"] == 0]["female"]
treat_female = df[df["treatment"] == 1]["female"]
t_stat_female, p_val_female = stats.ttest_ind(control_female, treat_female)
print("\nT-test for female:")
print(f"t-statistic: {t_stat_female:.4f}")
print(f"p-value: {p_val_female:.4f}")

# T-test for couple
control_couple = df[df["treatment"] == 0]["couple"]
treat_couple = df[df["treatment"] == 1]["couple"]
t_stat_couple, p_val_couple = stats.ttest_ind(control_couple, treat_couple)
print("\nT-test for couple:")
print(f"t-statistic: {t_stat_couple:.4f}")
print(f"p-value: {p_val_couple:.4f}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
# Calculate proportion of people who donated in each group
donation_rates = df.groupby("treatment")["gave"].mean()
donation_rates.index = ["Control", "Treatment"]

# Create a bar plot
plt.figure(figsize=(10, 6))
ax = donation_rates.plot(kind="bar", color=['#3498db', '#2ecc71'])
plt.ylabel("Proportion Donated", fontsize=12)
plt.title("Donation Rate by Treatment Group", fontsize=14, pad=20)
plt.ylim(0, 0.04)
plt.xticks(rotation=0)

# Add value labels on top of bars
for i, v in enumerate(donation_rates):
    ax.text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=11)

plt.tight_layout()
plt.show()

# T-test for donation rate
control_gave = df[df["treatment"] == 0]["gave"]
treat_gave = df[df["treatment"] == 1]["gave"]
t_stat_gave, p_val_gave = stats.ttest_ind(treat_gave, control_gave)
print("T-test for donation rate:")
print(f"t-statistic: {t_stat_gave:.4f}")
print(f"p-value: {p_val_gave:.4f}")

# Linear regression: gave ~ treatment
model_gave = smf.ols("gave ~ treatment", data=df).fit()
print("\nLinear regression for donation rate:")
print(model_gave.summary().tables[1])  # Only show the coefficients table

# Probit model: gave ~ treatment
probit_model = smf.probit("gave ~ treatment", data=df).fit()
print("\nProbit regression for donation rate:")
print(probit_model.summary().tables[1])  # Only show the coefficients table

# Calculate marginal effects
mfx = probit_model.get_margeff()
print("\nMarginal effects:")
print(mfx.summary().tables[0])  # Only show the marginal effects table
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
# Filter to only treatment group
treat_df = df[df["treatment"] == 1].copy()

# Convert dummy columns to clean 0/1 integers
for col in ["ratio", "ratio2", "ratio3"]:
    treat_df[col] = pd.to_numeric(treat_df[col], errors='coerce').fillna(0).astype(int)

# Create 'ratio1' (to mirror 'ratio2' and 'ratio3')
treat_df["ratio1"] = treat_df["ratio"]  # 1:1 match group

# Calculate donation rates by match ratio
ratio_rates = treat_df.groupby(["ratio1", "ratio2", "ratio3"])["gave"].mean()
print("Donation rates by match ratio:")
print(ratio_rates)

# Compare 2:1 vs 1:1
group_1_1 = treat_df[treat_df["ratio1"] == 1]["gave"]
group_2_1 = treat_df[treat_df["ratio2"] == 1]["gave"]
t_2, p_2 = stats.ttest_ind(group_2_1, group_1_1)
print("\n2:1 vs 1:1 match rate:")
print(f"t-statistic: {t_2:.4f}")
print(f"p-value: {p_2:.4f}")

# Compare 3:1 vs 1:1
group_3_1 = treat_df[treat_df["ratio3"] == 1]["gave"]
t_3, p_3 = stats.ttest_ind(group_3_1, group_1_1)
print("\n3:1 vs 1:1 match rate:")
print(f"t-statistic: {t_3:.4f}")
print(f"p-value: {p_3:.4f}")

# Regression: gave ~ ratio1 + ratio2 + ratio3
model_ratio = smf.ols("gave ~ ratio1 + ratio2 + ratio3", data=treat_df).fit()
print("\nRegression on match ratios:")
print(model_ratio.summary().tables[1])  # Only show the coefficients table

# Calculate response rate differences
response_1_1 = treat_df[treat_df["ratio1"] == 1]["gave"].mean()
response_2_1 = treat_df[treat_df["ratio2"] == 1]["gave"].mean()
response_3_1 = treat_df[treat_df["ratio3"] == 1]["gave"].mean()

diff_2v1 = response_2_1 - response_1_1
diff_3v2 = response_3_1 - response_2_1

print("\nResponse rate differences:")
print(f"2:1 vs 1:1: {diff_2v1:.4f}")
print(f"3:1 vs 2:1: {diff_3v2:.4f}")

# Create a bar plot of donation rates by match ratio
ratio_data = pd.DataFrame({
    'Match Ratio': ['1:1', '2:1', '3:1'],
    'Donation Rate': [response_1_1, response_2_1, response_3_1]
})

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Match Ratio', y='Donation Rate', data=ratio_data, palette='viridis')
plt.title('Donation Rate by Match Ratio', fontsize=14, pad=20)
plt.ylabel('Proportion Donated', fontsize=12)

# Add value labels on top of bars
for i, v in enumerate(ratio_data['Donation Rate']):
    ax.text(i, v + 0.001, f"{v:.3f}", ha='center', fontsize=11)

plt.tight_layout()
plt.show()

# Additional analysis: Test for differences in donation amounts by match ratio
# Filter to only people who gave (positive amount)
treat_gave = treat_df[treat_df["gave"] == 1].copy()

# Calculate mean donation amounts by match ratio
mean_amt_1_1 = treat_gave[treat_gave["ratio1"] == 1]["amount"].mean()
mean_amt_2_1 = treat_gave[treat_gave["ratio2"] == 1]["amount"].mean()
mean_amt_3_1 = treat_gave[treat_gave["ratio3"] == 1]["amount"].mean()

print("\nMean donation amounts by match ratio:")
print(f"1:1 match: ${mean_amt_1_1:.2f}")
print(f"2:1 match: ${mean_amt_2_1:.2f}")
print(f"3:1 match: ${mean_amt_3_1:.2f}")

# T-test for donation amounts: 2:1 vs 1:1
amt_1_1 = treat_gave[treat_gave["ratio1"] == 1]["amount"]
amt_2_1 = treat_gave[treat_gave["ratio2"] == 1]["amount"]
t_amt_2, p_amt_2 = stats.ttest_ind(amt_2_1, amt_1_1)
print("\nT-test for donation amounts (2:1 vs 1:1):")
print(f"t-statistic: {t_amt_2:.4f}")
print(f"p-value: {p_amt_2:.4f}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
# T-test for donation amount
amount_control = df[df["treatment"] == 0]["amount"]
amount_treatment = df[df["treatment"] == 1]["amount"]
t_stat_amt, p_val_amt = stats.ttest_ind(amount_treatment, amount_control)

print("T-test on donation amount:")
print(f"t-statistic: {t_stat_amt:.4f}")
print(f"p-value: {p_val_amt:.4f}")

# Regression: amount ~ treatment
model_amt = smf.ols("amount ~ treatment", data=df).fit()
print("\nRegression on donation amount:")
print(model_amt.summary().tables[1])  # Only show the coefficients table

# Filter to only people who gave (positive amount)
df_gave = df[df["gave"] == 1].copy()

# Regression: amount ~ treatment (conditional on giving)
model_cond = smf.ols("amount ~ treatment", data=df_gave).fit()
print("\nRegression on donation amount (only for donors):")
print(model_cond.summary().tables[1])  # Only show the coefficients table

# Calculate mean donation amounts
mean_treat = df_gave[df_gave["treatment"] == 1]["amount"].mean()
mean_control = df_gave[df_gave["treatment"] == 0]["amount"].mean()

print(f"\nMean donation amount (treatment): ${mean_treat:.2f}")
print(f"Mean donation amount (control): ${mean_control:.2f}")

# Create histograms of donation amounts
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Treatment group
treat_donors = df_gave[df_gave["treatment"] == 1]["amount"]
axs[0].hist(treat_donors, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
axs[0].axvline(mean_treat, color='red', linestyle='dashed', linewidth=2)
axs[0].set_title("Treatment Group", fontsize=14)
axs[0].set_xlabel("Donation Amount ($)", fontsize=12)
axs[0].set_ylabel("Frequency", fontsize=12)
axs[0].text(0.95, 0.95, f"Mean: ${mean_treat:.2f}", 
            transform=axs[0].transAxes, ha='right', va='top', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Control group
control_donors = df_gave[df_gave["treatment"] == 0]["amount"]
axs[1].hist(control_donors, bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
axs[1].axvline(mean_control, color='red', linestyle='dashed', linewidth=2)
axs[1].set_title("Control Group", fontsize=14)
axs[1].set_xlabel("Donation Amount ($)", fontsize=12)
axs[1].text(0.95, 0.95, f"Mean: ${mean_control:.2f}", 
            transform=axs[1].transAxes, ha='right', va='top', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

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
#
#
#| echo: false
# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n = 10000
p_treat = 0.022  # True probability for treatment group
p_control = 0.018  # True probability for control group
true_diff = p_treat - p_control

# Simulate
treatment = np.random.binomial(1, p_treat, n)
control = np.random.binomial(1, p_control, n)
diffs = treatment - control
cumulative_avg = np.cumsum(diffs) / np.arange(1, n + 1)

# Plot
plt.figure(figsize=(12, 7))
plt.plot(cumulative_avg, label="Cumulative Avg. Difference", color="#3498db", linewidth=2)
plt.axhline(true_diff, color='red', linestyle='--', linewidth=2, 
            label=f"True Difference ({true_diff:.3f})")

# Style
plt.title("Law of Large Numbers Simulation", fontsize=16, weight='bold')
plt.xscale('log')
plt.xlabel("Number of Observations", fontsize=13)
plt.ylabel("Cumulative Average Difference", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# Fix the y-axis limits to ensure the graph is not upside down
plt.ylim(0, 0.01)  # Set appropriate y-axis limits

# Add a horizontal line at y=0 for reference
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# Create a more detailed simulation to better illustrate convergence
# Multiple simulations to show variability
n_sims = 5
n_obs = 10000
plt.figure(figsize=(12, 7))

for i in range(n_sims):
    # Simulate
    treatment = np.random.binomial(1, p_treat, n_obs)
    control = np.random.binomial(1, p_control, n_obs)
    diffs = treatment - control
    cumulative_avg = np.cumsum(diffs) / np.arange(1, n_obs + 1)
    
    # Plot
    plt.plot(cumulative_avg, label=f"Simulation {i+1}", alpha=0.7, linewidth=1.5)

plt.axhline(true_diff, color='red', linestyle='--', linewidth=2, 
            label=f"True Difference ({true_diff:.3f})")

# Style
plt.title("Law of Large Numbers: Multiple Simulations", fontsize=16, weight='bold')
plt.xscale('log')
plt.xlabel("Number of Observations", fontsize=13)
plt.ylabel("Cumulative Average Difference", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(0, 0.01)  # Set appropriate y-axis limits
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)

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
#| echo: false
# Parameters
p_control = 0.018
p_treat = 0.022
n_sim = 1000
sample_sizes = [50, 200, 500, 1000]

# Set random seed for reproducibility
np.random.seed(42)

# Initialize figure
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, n in enumerate(sample_sizes):
    # Store average differences for this sample size
    diff_means = []
    
    for _ in range(n_sim):
        control_sample = np.random.binomial(1, p_control, n)
        treat_sample = np.random.binomial(1, p_treat, n)
        diff = treat_sample.mean() - control_sample.mean()
        diff_means.append(diff)

    # Plot histogram
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
#
#
