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

# # Optionally, keep respondent and task info for later grouping
# respondent = conjoint_data['resp'].values
# task = conjoint_data['task'].values

print(conjoint_data.head())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
