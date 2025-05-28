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
import numpy as np
import pandas as pd

# set seed for reproducibility
np.random.seed(123)

# define attributes
brand = ["N", "P", "H"]  # Netflix, Prime, Hulu
ad = ["Yes", "No"]
price = list(range(8, 33, 4))

# generate all possible profiles
profiles = pd.DataFrame([
    (b, a, p) for b in brand for a in ad for p in price
], columns=["brand", "ad", "price"])
m = len(profiles)

# assign part-worth utilities (true parameters)
b_util = {"N": 1.0, "P": 0.5, "H": 0}
a_util = {"Yes": -0.8, "No": 0.0}
def p_util(p): return -0.1 * p

# number of respondents, choice tasks, and alternatives per task
n_peeps = 100
n_tasks = 10
n_alts = 3

# function to simulate one respondent's data
def sim_one(id):
    datlist = []
    for t in range(1, n_tasks + 1):
        dat = profiles.sample(n=n_alts).copy()
        dat.insert(0, "task", t)
        dat.insert(0, "resp", id)
        dat["v"] = (
            dat["brand"].map(b_util) +
            dat["ad"].map(a_util) +
            dat["price"].apply(p_util)
        ).round(10)
        dat["e"] = -np.log(-np.log(np.random.uniform(size=n_alts)))
        dat["u"] = dat["v"] + dat["e"]
        dat["choice"] = (dat["u"] == dat["u"].max()).astype(int)
        datlist.append(dat)
    return pd.concat(datlist, ignore_index=True)

# simulate data for all respondents
conjoint_data = pd.concat([sim_one(i) for i in range(1, n_peeps + 1)], ignore_index=True)

# remove values unobservable to the researcher
conjoint_data = conjoint_data[["resp", "task", "brand", "ad", "price", "choice"]]

# clean up
for name in dir():
    if name != "conjoint_data":
        del globals()[name]

print(conjoint_data.head())
```
#
#
#
#
#
#
#
#
#
#

conjoint_data['brand_netflix'] = (conjoint_data['brand'] == 'N').astype(int)
conjoint_data['brand_prime'] = (conjoint_data['brand'] == 'P').astype(int)
conjoint_data['ads_yes'] = (conjoint_data['ad'] == 'Yes').astype(int)

X = conjoint_data[['brand_netflix', 'brand_prime', 'ads_yes', 'price']].values
y = conjoint_data['choice'].values

respondent = conjoint_data['resp'].values
task = conjoint_data['task'].values

print(conjoint_data.head())
#
#
#
#
#
#
#

import numpy as np

def mnl(beta, X, y, ids):
    v = X @ beta
    
    unique_tasks = np.unique(ids)
    log_likelihood = 0.0
    for t in unique_tasks:
        idx = (ids == t)
        v_t = v[idx]
        y_t = y[idx]

        denom = np.log(np.sum(np.exp(v_t)))
        log_likelihood += np.sum(y_t * (v_t - denom))

    return -log_likelihood

task_ids = conjoint_data['resp'].astype(str) + "_" + conjoint_data['task'].astype(str)
task_ids = task_ids.values
#
#
#
#
#
#
from scipy.optimize import minimize

# Minimize the negative log-likelihood
result = minimize(
    mnl,
    x0=np.zeros(4),
    args=(X, y, task_ids),
    method='BFGS'
)

print("$\beta_\text{netflix}$, $\beta_\text{prime}$, $\beta_\text{ads}$, $\beta_\text{price}$")
print("MLE estimates:", result.x)

# Standard errors from the Hessian
hessian_inv = result.hess_inv
se = np.sqrt(np.diag(hessian_inv))
print("Standard errors:", se)

# 95% confidence intervals
ci_lower = result.x - 1.96 * se
ci_upper = result.x + 1.96 * se
print("95% CI lower:", ci_lower)
print("95% CI upper:", ci_upper)

#
#
#
#
#
#
#
#
#
#
#
#
#
#

import numpy as np

# Log-prior function
def log_prior(beta):
    # beta: array of length 4
    # First 3: N(0, 5^2), Last: N(0, 1^2)
    lp = 0
    lp += -0.5 * ((beta[0] / 5) ** 2 + np.log(2 * np.pi * 25))
    lp += -0.5 * ((beta[1] / 5) ** 2 + np.log(2 * np.pi * 25))
    lp += -0.5 * ((beta[2] / 5) ** 2 + np.log(2 * np.pi * 25))
    lp += -0.5 * ((beta[3] / 1) ** 2 + np.log(2 * np.pi * 1))
    return lp

# Log-posterior function
def log_posterior(beta, X, y, task_ids):
    # log-likelihood (not negative)
    ll = -mnl(beta, X, y, task_ids)
    lp = log_prior(beta)
    return ll + lp

# MCMC parameters
n_steps = 11000
burn_in = 1000
np.random.seed(42)

# Start at MLE or zeros
beta_current = np.zeros(4)
samples = np.zeros((n_steps, 4))
log_post_current = log_posterior(beta_current, X, y, task_ids)

# Proposal std devs
proposal_sd = np.array([0.05, 0.05, 0.05, 0.005])

for step in range(n_steps):
    # Propose new beta
    beta_proposal = beta_current + np.random.normal(0, proposal_sd)
    log_post_proposal = log_posterior(beta_proposal, X, y, task_ids)
    # Acceptance probability
    accept_prob = np.exp(log_post_proposal - log_post_current)
    if np.random.rand() < accept_prob:
        beta_current = beta_proposal
        log_post_current = log_post_proposal
    samples[step, :] = beta_current
    if (step+1) % 1000 == 0:
        print(f"Step {step+1}/{n_steps}")

# Discard burn-in
posterior_samples = samples[burn_in:, :]

# Posterior summaries
means = posterior_samples.mean(axis=0)
stds = posterior_samples.std(axis=0)
ci_lower = np.percentile(posterior_samples, 2.5, axis=0)
ci_upper = np.percentile(posterior_samples, 97.5, axis=0)

print("Posterior means:", means)
print("Posterior stds:", stds)
print("95% credible intervals:")
print("Lower:", ci_lower)
print("Upper:", ci_upper)

#
#
#
#
#
#
#

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(posterior_samples[:,0])
plt.title("Trace plot: beta_Netflix")
plt.xlabel("Iteration")
plt.ylabel("Value")

plt.subplot(1,2,2)
plt.hist(posterior_samples[:,0], bins=30, density=True)
plt.title("Posterior: beta_Netflix")
plt.xlabel("Value")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

#
#
#
#
#

# Print MLE results
print("=== Maximum Likelihood Estimates ===")
print("Parameter   Estimate    Std.Err    95% CI Lower    95% CI Upper")
for i, name in enumerate(['Netflix', 'Prime', 'Ads', 'Price']):
    print(f"{name:10s}  {result.x[i]:10.3f}  {se[i]:8.3f}  {ci_lower[i]:13.3f}  {ci_upper[i]:13.3f}")

print("\n=== Bayesian Posterior Summaries ===")
print("Parameter   Mean        Std.Dev    95% CI Lower    95% CI Upper")
for i, name in enumerate(['Netflix', 'Prime', 'Ads', 'Price']):
    print(f"{name:10s}  {means[i]:10.3f}  {stds[i]:8.3f}  {ci_lower[i]:13.3f}  {ci_upper[i]:13.3f}")

# Optional: Show as a DataFrame for easier comparison
import pandas as pd

summary_df = pd.DataFrame({
    'Parameter': ['Netflix', 'Prime', 'Ads', 'Price'],
    'MLE_Estimate': result.x,
    'MLE_StdErr': se,
    'MLE_95CI_Lower': ci_lower,
    'MLE_95CI_Upper': ci_upper,
    'Bayes_Mean': means,
    'Bayes_StdDev': stds,
    'Bayes_95CI_Lower': ci_lower,
    'Bayes_95CI_Upper': ci_upper
})

print("\n=== Side-by-side Comparison ===")
print(summary_df)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
