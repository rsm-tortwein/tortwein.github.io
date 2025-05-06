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


_todo: Use your function to plot lambda on the horizontal axis and the likelihood (or log-likelihood) on the vertical axis for a range of lambdas (use the observed number of patents as the input for Y)._

```{python}

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
    """
    Calculate the log-likelihood for a Poisson regression model.
    
    Parameters:
    beta (array): Regression coefficients
    Y (array): Observed count data
    X (array): Design matrix of covariates
    
    Returns:
    float: Log-likelihood value
    """
    # Convert inputs to numpy arrays
    Y = np.array(Y)
    X = np.array(X)
    beta = np.array(beta)
    
    # Calculate λᵢ = exp(Xᵢ'β) for each observation
    lambda_i = np.exp(X @ beta)
    
    # Calculate the log-likelihood
    # ℓ(β) = ∑[Yᵢ(Xᵢ'β) - exp(Xᵢ'β) - log(Yᵢ!)]
    n = len(Y)
    sum_Y_log_lambda = np.sum(Y * (X @ beta))
    sum_lambda = np.sum(lambda_i)
    sum_log_factorial = np.sum(np.log(factorial(Y)))
    
    log_likelihood = sum_Y_log_lambda - sum_lambda - sum_log_factorial
    return log_likelihood

# Example usage:
# X = np.column_stack([np.ones(len(Y)),  # Intercept
#                     blueprinty_df['age'],
#                     blueprinty_df['age']**2,
#                     pd.get_dummies(blueprinty_df['region'], drop_first=True),
#                     blueprinty_df['iscustomer']])
# 
# beta_initial = np.zeros(X.shape[1])  # Initial guess for beta
# print("Log-likelihood:", poisson_regression_loglikelihood(beta_initial, Y, X))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
