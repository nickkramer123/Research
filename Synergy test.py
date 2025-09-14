from sklearn.feature_selection import mutual_info_regression


import numpy as np
import pickle


# TODO fix function. Now we can test it yay!
def synergy(X, f_x):
    back = 0 # initialize back half of function
    front = mutual_info_regression(X, f_x.ravel())[0] # raval flattens an array to 1D
    for j in range(X.shape[1]):   # for each column
        x = X[:, j].reshape(-1, 1) # grabs one, arbitrary column of x
        back += mutual_info_regression(x, f_x.ravel())[0]
    final = front - np.sum(back)
    return final


# call X from Rossler Attractor
with open("./Research/data/values.pkl", "rb") as file:
    values_dict = pickle.load(file)

# CHECK placing X values at a specific Beta
beta = list(values_dict.keys())[-1]   # index is changeable?

# convert dict to 2D array
values_array = np.array(values_dict[beta], dtype=float)  # 3000 rows, 3 cols

# # generate sample data for f(x)
# random_fx = list(range(1, 3001))
# random_fx = np.array(random_fx).reshape(-1, 1)
# random_fx = np.array(random_fx).ravel()

# print(synergy(values_array, random_fx))

# TESTS:
n=2000
x = np.random.randn(n)
y = np.random.randn(n)
z = np.random.randn(n)
X = np.column_stack([x,y,z])
f = np.random.randn(n)

print("Independent:", synergy(X,f))


n=2000
x = np.random.randn(n)
y = x + 0.001*np.random.randn(n)
z = x + 0.002*np.random.randn(n)
X = np.column_stack([x,y,z])
f = x + 0.1*np.random.randn(n)

print("Redundant:", synergy(X,f))





np.random.seed(0)
X = np.random.rand(3000, 3)         # 3000 samples, 3 features
f_x = np.random.rand(3000, 1)       # target

# --- Original synergy ---
print("Original synergy:", synergy(X, f_x))

