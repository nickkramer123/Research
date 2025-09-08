from sklearn.feature_selection import mutual_info_regression


import numpy as np
import pickle


# TODO fix function. Now we can test it yay!
# TODO does this work for discrete data?
def synergy(X, f_x):
    back = 0
    front = mutual_info_regression(X, f_x.ravel())[0]
    for j in range(X.shape[1]):   # for each column
        x = X[:, j].reshape(-1, 1)
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

# CHAT GPT TESTS:
n=2000
x = np.random.randn(n)
y = np.random.randn(n)
z = np.random.randn(n)
X = np.column_stack([x,y,z])
f = np.random.randn(n)   # independent of X

print("Independent:", synergy(X,f))


n=2000
x = np.random.randn(n)
y = x + 0.001*np.random.randn(n)  # nearly duplicate
z = x + 0.002*np.random.randn(n)
X = np.column_stack([x,y,z])
f = x + 0.1*np.random.randn(n)

print("Redundant:", synergy(X,f))



# WRONG????? MAYBE IDK
n=2000
x = np.random.randint(0,2,n)
y = np.random.randint(0,2,n)
f = np.logical_xor(x,y).astype(int)
X = np.column_stack([x,y])
print("XOR discrete (expected positive):", synergy(X,f))


# --- Generate synthetic continuous data ---
np.random.seed(0)
X = np.random.rand(3000, 3)         # 3000 samples, 3 features
f_x = np.random.rand(3000, 1)       # target

# --- Original synergy ---
print("Original synergy:", synergy(X, f_x))

# --- Permuted target (breaks dependence) ---
f_x_perm = np.random.permutation(f_x)
print("Permuted synergy (should be ~0):", synergy(X, f_x_perm))