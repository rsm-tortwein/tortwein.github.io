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
    eta = np.clip(eta, -20, 20)  # Prevent overflow in exp()
    lambda_ = np.exp(eta)

    log_likelihood = np.sum(Y * eta - lambda_ - gammaln(Y + 1))
    return -log_likelihood  # Minimize negative log-likelihood

# === Step 2: Prepare the Data ===
# Extract the outcome variable
Y = blueprinty_df['patents'].astype(float).values

# Create features: standardize age and calculate age_squared
X = blueprinty_df[['age', 'region', 'iscustomer']].copy()
X['age'] = (X['age'] - X['age'].mean()) / X['age'].std()
X['age_squared'] = X['age'] ** 2

# Create region dummies (drop one to avoid multicollinearity)
region_dummies = pd.get_dummies(X['region'], prefix='region', drop_first=True)

# Final X matrix with intercept
X_final = pd.concat([
    pd.Series(1.0, index=X.index, name='intercept'),
    X[['age', 'age_squared', 'iscustomer']],
    region_dummies
], axis=1)

X_np = X_final.to_numpy(dtype=np.float64)

# === Step 3: Run the Optimization ===
initial_beta = np.zeros(X_np.shape[1])
result = minimize(
    neg_log_likelihood,
    x0=initial_beta,
    args=(Y, X_np),
    method='L-BFGS-B'
)

if not result.success:
    raise RuntimeError(f"Optimization failed: {result.message}")

# === Step 4: Extract Estimates and Standard Errors ===
beta_hat = result.x
hessian_inv = result.hess_inv
if hasattr(hessian_inv, 'todense'):
    cov_matrix = hessian_inv.todense()
else:
    cov_matrix = hessian_inv
standard_errors = np.sqrt(np.diag(cov_matrix))

# === Step 5: Display Summary Table ===
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
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Step 1: Prepare the data
df = blueprinty_df.copy()

# Standardize age and create age_squared
df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['age_squared'] = df['age'] ** 2

# Create dummy variables for region, drop the first for reference group
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

# Combine all predictors into a single DataFrame
model_df = pd.concat([
    df[['patents', 'age', 'age_squared', 'iscustomer']],
    region_dummies
], axis=1)

# Ensure all columns are numeric
model_df = model_df.apply(pd.to_numeric, errors='coerce')  # Coerce errors to NaN

# Drop any rows with NaNs in any column (this is the key fix)
model_df = model_df.dropna()

# Define response and predictor matrices
y = model_df['patents'].astype(float)
X = sm.add_constant(model_df.drop(columns='patents'))

# Step 2: Fit Poisson GLM
poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

# Step 3: Show summary
print(poisson_results.summary())

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
airbnb_df.head()
#
#
#
#
#
import matplotlib.pyplot as plt

# Replace 'room_type' and 'number_of_reviews' with your actual column names
plt.figure(figsize=(12, 5))

for i, group in enumerate(airbnb_df['room_type'].unique()):
    plt.subplot(1, len(airbnb_df['room_type'].unique()), i+1)
    plt.hist(airbnb_df[airbnb_df['room_type'] == group]['number_of_reviews'], bins=20, alpha=0.7)
    plt.title(f'Number of Reviews - {group}')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Calculate and display mean number of reviews for each group
mean_reviews = airbnb_df.groupby('room_type')['number_of_reviews'].mean()
print("\nMean Number of Reviews by Room Type:")
for group, mean in mean_reviews.items():
    print(f"{group}: {mean:.2f}")
#
#
#
#
#
#
import matplotlib.pyplot as plt
import seaborn as sns

# --- Regional distribution by room type (as a proxy for 'customer status') ---
plt.figure(figsize=(10, 5))
regional_dist = pd.crosstab(airbnb_df['neighbourhood_group'], airbnb_df['room_type'], normalize='columns') * 100
regional_dist.plot(kind='bar', alpha=0.7)
plt.title('Regional Distribution by Room Type')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Percentage')
plt.legend(title='Room Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nRegional Distribution (%):")
print(regional_dist)

# --- 'Age' distribution by room type (using minimum_nights as a proxy for 'age') ---
plt.figure(figsize=(12, 5))

# Minimum nights distribution for each room type
for i, room in enumerate(airbnb_df['room_type'].unique()):
    plt.subplot(1, len(airbnb_df['room_type'].unique()), i+1)
    sns.boxplot(data=airbnb_df[airbnb_df['room_type'] == room], y='minimum_nights')
    plt.title(f'Minimum Nights - {room}')
    plt.ylabel('Minimum Nights')

plt.tight_layout()
plt.show()

print("\nMinimum Nights Summary Statistics by Room Type:")
print(airbnb_df.groupby('room_type')['minimum_nights'].describe())
#
#
#
