# Rossler Attractor

import os
import numpy as np
import pickle
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
from scipy import stats

# parameters
a = .2
# ADJUSTABLE to an extent
b_list = np.linspace(.1, 2, 1000) # 1000 values between 0 and 2
g = 5.7

# initial variable values
# good default, may need to change later
initVars = [1, 1, 1]

# generate a solution at 3000 evenly spaced samples in the interval 0 <= t <= 250.
tset = np.linspace(0, 250, 3000) # ADJUSTABLE

# FIRST ORDER OF BUISNESS
def rossler(initVars, t, alpha, beta, gamma):
    x = initVars[0]
    y = initVars[1]
    z = initVars[2]
    dxdt = - y - z # CHANGE MADE
    dydt = x + alpha * y
    dzdt = beta + z * (x - gamma) # THERE MAY BE TYPO HERE, CHANGED NOW
    return [dxdt, dydt, dzdt]


# func: the function that computes the derivatives
# y0: array, initial condition of x, y, z
# t: array, sequence of time points to solve for x, y, z
# args: extra arguments to pass through to function (alpha, beta, gamma)

# solutions in a dictionary, a list for each b value
sol_dict = {}

# create empty lists for bifurcation diagram
b_vals_x = []
b_vals_z = []
z_steady_state =[] # long term (post-transient) values of z
x_steady_state =[] # long term (post-transient) values of x


for i, b in enumerate(b_list):
    sol = odeint(
        func = rossler, 
        y0 = initVars,
        t = tset,
        args = (a, b, g),
        atol=1e-10,  # handle bigger workload
        rtol=1e-10)
    sol_dict[b] = sol


    # get rid of pre-transient values
    post_trans_dict = {}

    for b, arr in sol_dict.items():
        # slice away first half
        half = int(0.5 * len(arr))
        post_trans_arr = arr[half:, :]  # keep rows after half, all columns
        post_trans_dict[b] = post_trans_arr


    # extract values for graphs
    x_values = post_trans_dict[b][:, 0]
    y_values = post_trans_dict[b][:, 1]
    z_values = post_trans_dict[b][:, 2]


    




    # 3D Plots
    # For 5 different indexes
    if i % 225 == 0: 
        fig = plt.figure() # create figure
        ax = fig.add_subplot(111, projection='3d') # add new axis
        ax.scatter(xs = x_values, ys = y_values, zs = z_values, zdir='z', s=20, c=None, depthshade=True)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title('Attractor at β = ' + f'{b:.3f}' + ':')

        plt.show()

    # ADJUSTABLE
    # collect z values
    # zPostTrans =  z_values[int(.5*len(z_values)):] # getting rid off pre-transient behavior
    zPeaksInd, _ = find_peaks(z_values) # find local maximum indices
    zPeaks = z_values[zPeaksInd] # find values correlated to indices
    z_steady_state.extend(zPeaks) # store values
    b_vals_z.extend([b] * len(zPeaks))# store corresponding b values



    # collect x values
 #   zPostTrans =  z_values[int(.5*len(z_values)):] # getting rid off pre-transient behavior
    xPeaksInd, _ = find_peaks(x_values) # find local maximum indices
    xPeaks = x_values[xPeaksInd] # find values correlated to indices
    x_steady_state.extend(xPeaks) # store values
    b_vals_x.extend([b] * len(xPeaks))# store corresponding b values



# # make bifurcation diagram for z
plt.scatter(x = b_vals_z, y = z_steady_state, s = .3, alpha=.3, color="black")
plt.xlabel("β (beta)")
plt.ylabel("z (Steady-State Values)")
plt.title("Bifurcation Diagram of the Rössler System for z")
plt.grid(True)
plt.show()

# # make bifurcation diagram for x
plt.scatter(x = b_vals_x, y = x_steady_state, s = .3, alpha=.3, color="brown")
plt.xlabel("β (beta)")
plt.ylabel("x (Steady-State Values)")
plt.title("Bifurcation Diagram of the Rössler System for x")
plt.grid(True)
plt.show()

# for nn data collection
# TODO save data in a file to be used by NN
print(np.shape(x_values)) # dict for 1500 b values


# global normalization
# warning: each beta value has its own standardization
z_sol_dict = {}
for key, arr in post_trans_dict.items():
    if np.std(arr) == 0:
        z_sol_dict[key] = np.zeros_like(arr) # should never happen
    else:
        z_sol_dict[key] = stats.zscore(arr)


os.makedirs("./Research/data", exist_ok=True)  # creates the folders if missing
with open("./Research/data/values.pkl", "wb") as file:
    pickle.dump(z_sol_dict, file)
