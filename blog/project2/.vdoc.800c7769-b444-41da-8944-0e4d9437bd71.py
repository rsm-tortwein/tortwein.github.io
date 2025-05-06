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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

blueprinty_df = pd.read_csv('blueprinty.csv')
airbnb_df = pd.read_csv('airbnb.csv')
#
#
#
#
#
# Create histograms to compare patent counts by customer status
plt.figure(figsize=(12, 5))

# First subplot for non-customers
plt.subplot(1, 2, 1)
plt.hist(blueprinty_df[blueprinty_df['iscustomer'] == 0]['patents'], 
         bins=20, alpha=0.7, color='red')
plt.title('Patents Distribution - Non-Customers')
plt.xlabel('Number of Patents')
plt.ylabel('Frequency')

# Second subplot for customers
plt.subplot(1, 2, 2)
plt.hist(blueprinty_df[blueprinty_df['iscustomer'] == 1]['patents'], 
         bins=20, alpha=0.7, color='blue')
plt.title('Patents Distribution - Customers')
plt.xlabel('Number of Patents')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Calculate and display mean patents for each group
mean_patents = blueprinty_df.groupby('iscustomer')['patents'].mean()
print("\nMean Patents by Customer Status:")
print(f"Non-Customers (0): {mean_patents[0]:.2f}")
print(f"Customers (1): {mean_patents[1]:.2f}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# First, let's analyze regional distribution
plt.figure(figsize=(10, 5))
regional_dist = pd.crosstab(blueprinty_df['region'], blueprinty_df['iscustomer'], normalize='columns') * 100
regional_dist.plot(kind='bar', alpha=0.7)
plt.title('Regional Distribution by Customer Status')
plt.xlabel('Region')
plt.ylabel('Percentage')
plt.legend(['Non-Customers', 'Customers'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the actual percentages
print("\nRegional Distribution (%):")
print(regional_dist)

# Now, let's analyze age distribution
plt.figure(figsize=(12, 5))

# Age distribution for non-customers
plt.subplot(1, 2, 1)
sns.boxplot(data=blueprinty_df[blueprinty_df['iscustomer'] == 0], y='age')
plt.title('Age Distribution - Non-Customers')
plt.ylabel('Age (years)')

# Age distribution for customers
plt.subplot(1, 2, 2)
sns.boxplot(data=blueprinty_df[blueprinty_df['iscustomer'] == 1], y='age')
plt.title('Age Distribution - Customers')
plt.ylabel('Age (years)')

plt.tight_layout()
plt.show()

# Print summary statistics for age
print("\nAge Summary Statistics:")
print(blueprinty_df.groupby('iscustomer')['age'].describe())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
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
from scipy.special import factorial

def poisson_loglikelihood(lambda_, Y):

    Y = np.array(Y)
    
    n = len(Y)
    sum_Y = np.sum(Y)
    sum_log_factorial = np.sum(np.log(factorial(Y)))
    
    log_likelihood = -n * lambda_ + sum_Y * np.log(lambda_) - sum_log_factorial
    return log_likelihood
#
#
#
#
#

# Get the patents data from the dataframe
Y = blueprinty_df['patents'].values

# Create a range of lambda values to evaluate
lambda_values = np.linspace(0.1, 10, 100)

# Calculate log-likelihood for each lambda value
log_likelihoods = [poisson_loglikelihood(l, Y) for l in lambda_values]

# Plot the log-likelihood function
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, log_likelihoods)
plt.xlabel('λ (lambda)')
plt.ylabel('Log-likelihood')
plt.title('Log-likelihood Function for Poisson Model')
plt.grid(True)

# Add a vertical line at the sample mean (which is the MLE)
sample_mean = np.mean(Y)
plt.axvline(x=sample_mean, color='r', linestyle='--', 
            label=f'MLE (λ = {sample_mean:.2f})')
plt.legend()

plt.show()

# Print the MLE value
print(f"Maximum Likelihood Estimate (MLE) of λ: {sample_mean:.2f}")
#
#
#
#
#
# Use scipy's optimize function to find the MLE
from scipy import optimize

# Define the negative log-likelihood function (for minimization)
def neg_poisson_loglikelihood(lambda_, Y):
    return -poisson_loglikelihood(lambda_, Y)

# Find the MLE using scipy.optimize.minimize
result = optimize.minimize(neg_poisson_loglikelihood, 
                         x0=np.mean(Y),  # Use sample mean as starting value
                         args=(Y,),
                         method='BFGS')

# Print optimization results
print("\nOptimization Results:")
print(f"MLE of λ from optimization: {result.x[0]:.2f}")

#
#
#
#
#
#
#
#
#
#
def poisson_regression_loglikelihood(beta, Y, X):
    Y, X, beta = np.array(Y), np.array(X), np.array(beta)
    lambda_i = np.exp(X @ beta)
    return np.sum(Y * (X @ beta) - lambda_i - np.log(factorial(Y)))
#
#
#
#
#
# Prepare the design matrix X
# First create a column of 1's for the intercept
X = np.ones((len(blueprinty_df), 1))

# Add age and age squared
age = blueprinty_df['age'].values.reshape(-1, 1)
age_squared = np.square(age)
X = np.hstack([X, age, age_squared])

# Add region dummy variables (excluding one region as reference)
regions = pd.get_dummies(blueprinty_df['region'], drop_first=True)
X = np.hstack([X, regions])

# Add customer status
X = np.hstack([X, blueprinty_df['iscustomer'].values.reshape(-1, 1)])

# Response variable
Y = blueprinty_df['patents'].values

# Define negative log likelihood for optimization
def poisson_regression_loglikelihood(beta, Y, X):
    return -poisson_regression_loglikelihood(beta, Y, X)

# Initial guess for beta (zeros)
initial_beta = np.zeros(X.shape[1])

# Optimize to find MLE
result = optimize.minimize(neg_poisson_regression_loglikelihood,
                         x0=initial_beta,
                         args=(Y, X),
                         method='BFGS',
                         hess=True)

# Extract MLE estimates and Hessian
beta_mle = result.x
hessian = result.hess_inv

# Calculate standard errors from the Hessian
std_errors = np.sqrt(np.diag(hessian))

# Create a DataFrame to display results
param_names = ['Intercept', 'Age', 'Age^2'] + \
              [f'Region_{i}' for i in regions.columns] + \
              ['Is_Customer']

results_df = pd.DataFrame({
    'Parameter': param_names,
    'Estimate': beta_mle,
    'Std Error': std_errors,
    'z-value': beta_mle / std_errors,
    'p-value': 2 * (1 - stats.norm.cdf(abs(beta_mle / std_errors)))
})

print("\nPoisson Regression Results:")
print(results_df.to_string(index=False))

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
