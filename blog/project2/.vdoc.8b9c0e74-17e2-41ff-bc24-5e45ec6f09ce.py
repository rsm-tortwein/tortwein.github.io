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
#| code-fold: true
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import factorial
from scipy import optimize

blueprinty_df = pd.read_csv('blueprinty.csv')
airbnb_df = pd.read_csv('airbnb.csv')

print(blueprinty_df.head())
#
#
#
#
#
#| code-fold: true
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(blueprinty_df[blueprinty_df['iscustomer'] == 0]['patents'], 
         bins=20, color='red')
plt.title('Patents Distribution - Non-Customers')
plt.xlabel('Number of Patents')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(blueprinty_df[blueprinty_df['iscustomer'] == 1]['patents'], 
         bins=20, color='blue')
plt.title('Patents Distribution - Customers')
plt.xlabel('Number of Patents')
plt.ylabel('Frequency')

plt.show()

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
#| code-fold: true

# Analyze regional and age distribution by customer status
plt.figure(figsize=(10, 5))
regional_dist = pd.crosstab(blueprinty_df['region'], blueprinty_df['iscustomer'], normalize='columns') * 100
regional_dist.plot(kind='bar')
plt.title('Regional Distribution by Customer Status')
plt.xlabel('Region')
plt.ylabel('Percentage')
plt.legend(['Non-Customers', 'Customers'])
plt.show()

print("\nRegional Distribution (%):")
print(regional_dist)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=blueprinty_df[blueprinty_df['iscustomer'] == 0], y='age')
plt.title('Age Distribution - Non-Customers')
plt.ylabel('Age (years)')

plt.subplot(1, 2, 2)
sns.boxplot(data=blueprinty_df[blueprinty_df['iscustomer'] == 1], y='age')
plt.title('Age Distribution - Customers')
plt.ylabel('Age (years)')

plt.show()

print("\n Mean Age (Non)Customer:")
print(blueprinty_df.groupby('iscustomer')['age'].mean())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
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
#
#
#| code-fold: true
Y = blueprinty_df['patents'].values
lambda_values = np.linspace(0.1, 10, 100)

log_likelihoods = [poisson_loglikelihood(lambda_, Y) for lambda_ in lambda_values]

plt.figure(figsize=(10, 6))
plt.plot(lambda_values, log_likelihoods)
plt.xlabel('λ (lambda)')
plt.ylabel('Log-likelihood')
plt.title('Log-likelihood Function for Poisson Model')
plt.grid(True)
plt.show()

sample_mean = np.mean(Y)
print(f"Maximum Likelihood Estimate (MLE) of λ: {sample_mean:.2f}")
#
#
#
#
#
#| code-fold: true
def neg_poisson_loglikelihood(lambda_, Y):
    return -poisson_loglikelihood(lambda_, Y)

result = optimize.minimize(neg_poisson_loglikelihood, 
                         x0=np.mean(Y),
                         args=(Y,),
                         method='BFGS')

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
#| code-fold: true
from scipy.special import gammaln

def neg_log_likelihood(beta, Y, X):
    beta = np.asarray(beta, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    eta = X @ beta
    eta = np.clip(eta, -20, 20)
    lambda_ = np.exp(eta)

    log_likelihood = np.sum(Y * eta - lambda_ - gammaln(Y + 1))
    return -log_likelihood
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
from scipy.optimize import minimize

Y = blueprinty_df['patents'].astype(float).values
X = blueprinty_df[['age', 'region', 'iscustomer']].copy()

X['age'] = (X['age'] - X['age'].mean()) / X['age'].std()
X['age_squared'] = X['age'] ** 2

region_dummies = pd.get_dummies(X['region'], prefix='region', drop_first=True)

X_final = pd.concat([
    pd.Series(1.0, index=X.index, name='intercept'),
    X[['age', 'age_squared', 'iscustomer']],
    region_dummies], axis=1)

X_np = X_final.to_numpy(dtype=np.float64)

initial_beta = np.zeros(X_np.shape[1])
result = minimize(
    neg_log_likelihood,
    x0=initial_beta,
    args=(Y, X_np),
    method='L-BFGS-B'
)

beta_hat = result.x
hessian_inv = result.hess_inv
cov_matrix = hessian_inv.todense()
standard_errors = np.sqrt(np.diag(cov_matrix))

summary_df = pd.DataFrame({
    'Coefficient': beta_hat,
    'Std. Error': standard_errors
}, index=X_final.columns)

summary_df = summary_df.round(4)
print("\nPoisson Regression Results:")
print(summary_df.to_string())
#
#
#
#
#
#| code-fold: true
import statsmodels.api as sm

df = blueprinty_df.copy()
df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['age_squared'] = df['age'] ** 2

region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)

model_df = pd.concat([
    df[['patents', 'age', 'age_squared', 'iscustomer']],
    region_dummies], axis=1)

y = model_df['patents'].astype(float)
X = sm.add_constant(model_df.drop(columns='patents')).astype(float)

poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

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
#| code-fold: true

X_0 = X.copy()
X_1 = X.copy()
X_0['iscustomer'] = 0
X_1['iscustomer'] = 1

y_pred_0 = poisson_results.predict(X_0)
y_pred_1 = poisson_results.predict(X_1)

avg_effect = (y_pred_1 - y_pred_0).mean()
print(f"Average effect of Blueprinty's software (iscustomer): {avg_effect:.3f} additional patents per firm")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
df = pd.read_csv("airbnb.csv")
print(df.head())
#
#
#
#

df_clean = df.dropna(subset=["number_of_reviews"])

selected_columns = [
    "number_of_reviews",
    "price",
    "room_type",
    "instant_bookable",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value"
]

df_model = df_clean[selected_columns].copy()
df_model.dropna(inplace=True)

df_model["instant_bookable"] = df_model["instant_bookable"].map({"t": 1, "f": 0})

df_model = pd.get_dummies(df_model, columns=["room_type"], drop_first=True)
print(df_model.head())

#
#
#
X = df_model.drop(columns=["number_of_reviews"]).copy()
X.insert(0, "intercept", 1)
X = X.values

# Response variable
y = df_model["number_of_reviews"].values

# Define the Poisson log-likelihood function
def poisson_log_likelihood(beta, X, y):
    """
    Calculates the negative log-likelihood for Poisson regression.
    
    Parameters:
    - beta: Coefficient vector
    - X: Design matrix
    - y: Observed counts
    
    Returns:
    - Negative log-likelihood (to minimize)
    """
    Xb = np.dot(X, beta)
    lambda_ = np.exp(Xb)
    
    # To avoid overflow in factorial, use gammaln(y+1) = log(y!)
    from scipy.special import gammaln
    log_likelihood = np.sum(y * Xb - lambda_ - gammaln(y + 1))
    
    return -log_likelihood  # Negative because we use minimization

# Check dimensions
print("X shape:", X.shape)
print("y shape:", y.shape)


#
#
#
#
