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

print(blueprinty_df.head())
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
log_likelihoods = [poisson_loglikelihood(lambda_, Y) for lambda_ in lambda_values]

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
from scipy.special import gammaln

def poisson_regression_loglikelihood(beta, Y, X):
    Y, X, beta = np.array(Y), np.array(X), np.array(beta)
    lambda_i = np.exp(X @ beta)
    return np.sum(Y * (X @ beta) - lambda_i - gammaln(Y + 1))
#
#
#
#
#
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# === Step 1: Define the Negative Log-Likelihood Function ===
def neg_log_likelihood(beta, Y, X):
    beta = np.asarray(beta, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    eta = X @ beta
    lambda_ = np.exp(eta)
    log_likelihood = np.sum(Y * eta - lambda_ - gammaln(Y + 1))
    return -log_likelihood  # Negative for minimization

# === Step 2: Prepare the Data ===
# Outcome variable
Y = blueprintny_df['patents'].astype(float).values

# Design matrix: include age, age^2, customer, and region dummies
X = blueprintny_df[['age', 'region', 'iscustomer']].copy()
X['age_squared'] = X['age'] ** 2
region_dummies = pd.get_dummies(X['region'], prefix='region', drop_first=True)

# Combine all features and add intercept
X_final = pd.concat([
    pd.Series(1.0, index=X.index, name='intercept'),
    X[['age', 'age_squared', 'iscustomer']],
    region_dummies
], axis=1)

X_np = X_final.to_numpy(dtype=np.float64)

# === Step 3: Run Optimization ===
initial_beta = np.zeros(X_np.shape[1])
result = minimize(
    neg_log_likelihood,
    x0=initial_beta,
    args=(Y, X_np),
    method='BFGS'
)

if not result.success:
    print("Optimization failed:", result.message)
else:
    print("Optimization succeeded.")

# === Step 4: Extract Coefficients and Standard Errors ===
beta_hat = result.x
hessian_inv = result.hess_inv
if hasattr(hessian_inv, 'todense'):
    cov_matrix = hessian_inv.todense()
else:
    cov_matrix = hessian_inv
standard_errors = np.sqrt(np.diag(cov_matrix))

# === Step 5: Display Results ===
summary_df = pd.DataFrame({
    'Coefficient': beta_hat,
    'Std. Error': standard_errors
}, index=X_final.columns)

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
#
#
#
#
#
#
import numpy as np
import pandas as pd
from scipy import optimize, stats

# Example: Let's use 'neighbourhood_group', 'room_type', and 'minimum_nights' as covariates
# (You can adjust covariates as needed for your analysis)

# 1. Construct the design matrix X
ng_dummies = pd.get_dummies(airbnb_df['neighbourhood_group'], drop_first=True)
room_dummies = pd.get_dummies(airbnb_df['room_type'], drop_first=True)
X_airbnb = np.column_stack([
    np.ones(len(airbnb_df)),  # Intercept
    airbnb_df['minimum_nights'],
    ng_dummies,
    room_dummies
])

# 2. Response variable (let's use 'number_of_reviews' as an example count variable)
Y_airbnb = airbnb_df['number_of_reviews'].values
#
#
#
#
def neg_poisson_regression_loglikelihood_airbnb(beta, Y, X):
    return -poisson_regression_loglikelihood(beta, Y, X)
#
#
#
#
# Initial guess for beta (all zeros)
beta_initial_airbnb = np.zeros(X_airbnb.shape[1])

# Optimize to find MLE
result_airbnb = optimize.minimize(
    neg_poisson_regression_loglikelihood_airbnb,
    x0=beta_initial_airbnb,
    args=(Y_airbnb, X_airbnb),
    method='BFGS'
)

# Extract MLE estimates and Hessian
beta_mle_airbnb = result_airbnb.x
cov_matrix_airbnb = result_airbnb.hess_inv  # Inverse Hessian (covariance matrix)
se_airbnb = np.sqrt(np.diag(cov_matrix_airbnb))
#
#
#
#
variable_names_airbnb = (
    ['Intercept', 'Minimum_Nights'] +
    list(ng_dummies.columns) +
    list(room_dummies.columns)
)
results_airbnb = pd.DataFrame({
    'Variable': variable_names_airbnb,
    'Coefficient': beta_mle_airbnb,
    'Std. Error': se_airbnb
})

print("\nAirBnB Poisson Regression Results:")
print(results_airbnb.round(4))
#
#
#
#
